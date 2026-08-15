"""W4A16 executor GEMM path unit tests (02 §6 / 04 §1.2 / int4-plan Q4b).

Covers the executor's INT4 data path introduced for W4A16:
  * `unpack_int4` nibble order + sign extension (authoritative packing order);
  * `read_matrix` INT4 unpack round-trip (packed via the executor's own
    `_write_vector`, the authoritative packer);
  * `_gemm_dequant(w4=True)` fp32 in-group accumulation vs an fp64 reference
    with the SAME per-128-group BF16 scales (implementation correctness), and
    the full `_matrix` BF16 writeback within 1 ULP of the fp64 reference
    rounded to BF16;
  * a regression that BF16 activations are NOT truncated through
    `int8_group_partials`' astype(int32) (the bug fixed in Q4b).

Runs standalone (`python3 qsim/test_int4.py`) and under pytest.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = np.float16

from compiler.isa import isa as I
from qsim.executor import Executor, unpack_int4


def _quantize_int4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Test-side symmetric per-128-K-group INT4 quantize (mirrors qforge/quant.py
    `quantize_weight_int4`, divisor 7, clip [-7, 7]). Returns (wqi, sw)."""
    N, K = w.shape
    G = K // 128
    wr = w.reshape(N, G, 128)
    sw = np.abs(wr).max(axis=2) / 7.0
    wqi = np.clip(np.round(wr / sw[:, :, None]), -7, 7).astype(np.int8)
    return wqi.reshape(N, K), sw.astype(np.float32)


def _bf16_ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b))
    ulp = np.zeros_like(mag)
    nz = mag > 0
    ulp[nz] = np.exp2(np.floor(np.log2(mag[nz])) - 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ulp > 0, d / ulp, np.where(d > 0, np.inf, 0.0))


# --------------------------------------------------------------------------- #
# nibble order
# --------------------------------------------------------------------------- #
def test_unpack_int4_nibble_order():
    # 0x9A = 1001_1010b: lo nibble 0xA -> -6, hi nibble 0x9 -> -7
    # 0x0F = 0000_1111b: lo nibble 0xF -> -1, hi nibble 0x0 ->  0
    out = unpack_int4(bytes([0x9A, 0x0F]), 4)
    assert out.dtype == np.int8
    assert out.tolist() == [-6, -7, -1, 0]
    # full positive range
    assert unpack_int4(bytes([0x07, 0x12]), 4).tolist() == [7, 0, 2, 1]


def test_int4_read_matrix_roundtrip():
    rng = np.random.default_rng(0)
    exe = Executor()
    # non-transposed [N, K]
    N, K = 8, 256
    wqi = rng.integers(-7, 8, size=(N, K)).astype(np.int8)
    exe._write_vector("sram", 0x1000, I.DT_INT4, wqi.reshape(-1))
    got = exe.read_matrix("sram", 0x1000, I.DT_INT4, N, K, K // 2, False)
    assert np.array_equal(got, wqi)
    # transposed storage: logical [K, N] == storage [N, K] = wqi
    got_t = exe.read_matrix("sram", 0x1000, I.DT_INT4, K, N, K // 2, True)
    assert np.array_equal(got_t, wqi.T)


# --------------------------------------------------------------------------- #
# W4A16 GEMM dequant
# --------------------------------------------------------------------------- #
# non-overlapping SRAM byte regions (16B aligned)
_A_ADDR = 0x00000   # [M, K] BF16, <= 0x2000 bytes
_B_ADDR = 0x10000   # [N, K] INT4 packed, <= 0x10000 bytes
_C_ADDR = 0x30000   # [M, N] BF16 output
_S_ADDR = 0x31000   # [N, G] BF16 per-128-group scale


def test_qforge_pack_to_executor_unpack_lock():
    """Bit-order lock (int4-plan §2 test ①): qforge.quant.pack_int4 output must
    unpack identically in the executor (read_matrix INT4 path)."""
    import qforge.quant as Q
    rng = np.random.default_rng(3)
    N, K = 16, 512
    wqi = rng.integers(-7, 8, size=(N, K)).astype(np.int8)
    packed = Q.pack_int4(wqi)                    # [N, K//2] uint8
    exe = Executor()
    exe.write_bytes("sram", 0x1000, packed.tobytes())
    got = exe.read_matrix("sram", 0x1000, I.DT_INT4, N, K, K // 2, False)
    assert np.array_equal(got, wqi)
    got_t = exe.read_matrix("sram", 0x1000, I.DT_INT4, K, N, K // 2, True)
    assert np.array_equal(got_t, wqi.T)
    # executor's own _write_vector must reproduce qforge.quant.pack_int4 bytes
    exe2 = Executor()
    exe2._write_vector("sram", 0x2000, I.DT_INT4, wqi.reshape(-1))
    assert exe2.read_bytes("sram", 0x2000, packed.nbytes) == packed.tobytes()


def _setup_w4a16(rng, M, N, K):
    """Build executor state + fp64 reference for one W4A16 GEMM."""
    G = K // 128
    A = (rng.standard_normal((M, K)) * 0.7).astype(BF16).astype(np.float32)
    W = rng.standard_normal((N, K)).astype(np.float32)
    wqi, sw = _quantize_int4(W)          # [N,K] int8, [N,G] fp32
    cd = sw.astype(BF16)                  # bf16 per-128-group scale

    exe = Executor()
    exe.write_bytes("sram", _A_ADDR, np.asarray(A, dtype=BF16).tobytes())
    exe._write_vector("sram", _B_ADDR, I.DT_INT4, wqi.reshape(-1))
    exe.write_bytes("sram", _S_ADDR, cd.tobytes())

    exe.AR[0] = _A_ADDR // 16            # A (word addr)
    exe.AR[1] = _B_ADDR // 16            # B (word addr)
    exe.AR[2] = _C_ADDR // 16            # C (word addr)
    exe.C[0] = (K * 2) << 16             # CA: row_stride = K*2 (BF16 A)
    exe.C[1] = (K // 2) << 16            # CB: row_stride = K//2 (INT4 B)
    exe.C[2] = (N * 2) << 16             # CC: row_stride = N*2 (BF16 C)
    exe.C[3] = (1 << 20) | (0 << 19) | (_S_ADDR // 16)  # CD: per-group, BF16

    # fp64 reference: sum_g cd[n,g] * (A_g @ wqi_g.T)  (same bf16 scales)
    ref = np.zeros((M, N), np.float64)
    for g in range(G):
        partial = (A.astype(np.float64)[:, g * 128:(g + 1) * 128]
                   @ wqi.astype(np.float64)[:, g * 128:(g + 1) * 128].T)
        ref += cd.astype(np.float64)[:, g][None, :] * partial
    return exe, A, wqi, cd, ref, (M, N, K, G)


def _run_ge(exe, M, N, K):
    word = I.encode_inst(
        "GEMM", srcA=I.DT_BF16, srcB=I.DT_INT4, acc=I.ACC_FP32,
        ARa=0, ARb=1, ARc=2, M=M, N=N, K=K, batch=1, CA=0, CB=1, CC=2, CD=3,
        acc_init=1, bsrc=0, dequant=1, transpose_A=0, transpose_B=1)
    exe._exec(I.decode_inst(word))


def _read_bf16_out(exe, M, N):
    out = np.empty((M, N), dtype=np.float32)
    for r in range(M):
        raw = exe.read_bytes("sram", _C_ADDR + r * N * 2, N * 2)
        out[r] = np.frombuffer(raw, dtype=BF16).astype(np.float32)
    return out


def test_w4a16_gemm_dequant_fp32():
    rng = np.random.default_rng(1)
    M, N, K = 4, 128, 1024
    exe, A, wqi, cd, ref, (M, N, K, G) = _setup_w4a16(rng, M, N, K)

    # fp32 dequant (before BF16 writeback) — the implementation-correctness
    # comparison against the fp64 reference with the same scales.
    B = exe.read_matrix("sram", _B_ADDR, I.DT_INT4, K, N, K // 2, True)
    assert np.array_equal(B, wqi.T), "INT4 unpack must match the source weights"
    C_f32 = exe._gemm_dequant(A.astype(np.float32), B, M, N, K, G, 0,
                              _S_ADDR, w4=True)
    abs_err = float(np.abs(C_f32.astype(np.float64) - ref).max())
    rel_err = float(abs_err / np.abs(ref).max())
    # fp32 in-group accumulation (04 §1.2): ~fp32-eps relative vs the fp64
    # reference with the same BF16 scales. The astype(int32) truncation bug
    # would push this to O(1) relative error.
    assert rel_err < 1e-5, f"W4A16 fp32 dequant rel err {rel_err:.3e}"

    # full _matrix path writes BF16; verify within 1 ULP of fp64 ref -> BF16.
    _run_ge(exe, M, N, K)
    out = _read_bf16_out(exe, M, N)
    golden_bf16 = ref.astype(BF16).astype(np.float32)
    max_ulp = float(np.ceil(_bf16_ulp_dist(out, golden_bf16).max()))
    assert max_ulp <= 1, f"W4A16 BF16 writeback {max_ulp} ULP"


def test_w4a16_no_activation_truncation():
    """Regression: fractional BF16 activations must NOT be truncated to int32.

    Activations in [-1.5, 1.5) all round to {-1, 0, 1} under astype(int32);
    the fp32 path must preserve them exactly (bf16 is a subset of fp32)."""
    rng = np.random.default_rng(2)
    M, N, K = 2, 128, 512
    # small fractional magnitudes, deliberately within int32-truncation range
    A = (rng.uniform(-1.5, 1.5, size=(M, K))).astype(BF16).astype(np.float32)
    W = rng.standard_normal((N, K)).astype(np.float32)
    wqi, sw = _quantize_int4(W)
    cd = sw.astype(BF16)

    G = K // 128
    exe = Executor()
    exe.write_bytes("sram", _A_ADDR, np.asarray(A, dtype=BF16).tobytes())
    exe._write_vector("sram", _B_ADDR, I.DT_INT4, wqi.reshape(-1))
    exe.write_bytes("sram", _S_ADDR, cd.tobytes())
    exe.AR[0], exe.AR[1], exe.AR[2] = 0, _B_ADDR // 16, _C_ADDR // 16
    exe.C[0] = (K * 2) << 16
    exe.C[1] = (K // 2) << 16
    exe.C[2] = (N * 2) << 16
    exe.C[3] = (1 << 20) | (0 << 19) | (_S_ADDR // 16)

    B = exe.read_matrix("sram", _B_ADDR, I.DT_INT4, K, N, K // 2, True)
    C_f32 = exe._gemm_dequant(A, B, M, N, K, G, 0, _S_ADDR, w4=True)

    ref = np.zeros((M, N), np.float64)
    for g in range(G):
        partial = (A.astype(np.float64)[:, g * 128:(g + 1) * 128]
                   @ wqi.astype(np.float64)[:, g * 128:(g + 1) * 128].T)
        ref += cd.astype(np.float64)[:, g][None, :] * partial
    abs_err = float(np.abs(C_f32.astype(np.float64) - ref).max())
    rel_err = float(abs_err / np.abs(ref).max())
    assert rel_err < 1e-5, f"truncation regression: rel err {rel_err:.3e}"


def test_w4a16_rejects_w4a8_combo():
    exe = Executor()
    exe.C[0] = exe.C[1] = exe.C[2] = 0
    exe.C[3] = (1 << 20) | (0 << 19) | 0
    # W4A8 (srcA=INT8, srcB=INT4, acc=INT32) is out of v0 scope: must reject.
    word = I.encode_inst(
        "GEMM", srcA=I.DT_INT8, srcB=I.DT_INT4, acc=I.ACC_INT32,
        ARa=0, ARb=1, ARc=2, M=1, N=128, K=128, batch=1, CA=0, CB=1, CC=2, CD=3,
        acc_init=1, bsrc=0, dequant=1, transpose_A=0, transpose_B=1)
    try:
        exe._exec(I.decode_inst(word))
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "W4A8 (INT8 activation) must be rejected in v0"


if __name__ == "__main__":
    for name in sorted(n for n in globals() if n.startswith("test_")):
        fn = globals()[name]
        fn()
        print(f"PASS {name}")
    print("test_int4: all pass")
