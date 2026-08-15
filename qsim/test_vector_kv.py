"""VECTOR (18) + KV (4) numeric semantics and field-level tests.

Two layers of coverage:

1. Field-level (encoding/decoding/opcode/illegal-reject) against 02-isa §3/§7/§8.
2. Numeric unit tests — primitive ops against numpy references (fp32 internal,
   bf16 writeback) and single-op golden comparison against P1 golden
   (RMSNorm / QK-norm / RoPE / softmax / SwiGLU / attention-score scale).
   Criterion (plan §1): bf16-domain <= 1 ULP + argmax match.

Runs standalone (`python3 qsim/test_vector_kv.py`) and under pytest.
"""
from __future__ import annotations
from pathlib import Path

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = np.float16

from compiler.isa import isa as I
from qsim.executor import Executor

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "golden", "qwen3-0.6b")
CACHE = "decode_seq1_cache1024"          # pos=1024, seq=1025 (non-trivial)
MODEL_SAFETENSORS = (
    os.environ.get("MODEL_SAFETENSORS",
                   str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/model.safetensors")))
LAYER = 27
EPS = 1e-6
ROPE_THETA = 1_000_000.0
HEAD_DIM = 128


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bf16b(x) -> bytes:
    return np.asarray(x, dtype=np.float32).astype(BF16).tobytes()


def _read_bf16(exe: Executor, word_addr: int, n: int) -> np.ndarray:
    return np.frombuffer(exe.read_bytes("sram", word_addr * 16, n * 2),
                         dtype=BF16).astype(np.float32)


def _fp32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _ulp(a, b) -> np.ndarray:
    """bf16-domain ULP distance (plan §1 / test_m2a convention)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b))
    ulp = np.zeros_like(mag)
    nz = mag > 0
    ulp[nz] = np.exp2(np.floor(np.log2(mag[nz])) - 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ulp > 0, d / ulp, np.where(d > 0, np.inf, 0.0))


def _run(exe: Executor, op: str, srcA: int = 0, srcB: int = 0, acc: int = 0,
         ARa: int = 0, ARb: int = 0, ARd: int = 0, length: int = 0,
         CV: int = 0, imm: int = 0, Cval: int | None = None, **operands):
    if Cval is not None:
        exe.C[CV] = Cval
    if op.startswith("KV."):
        w = I.encode_inst(op, srcA=srcA, srcB=srcB, acc=acc, **operands)
    else:
        operands.update(ARa=ARa, ARb=ARb, ARd=ARd, len=length, CV=CV, imm=imm)
        w = I.encode_inst(op, srcA=srcA, srcB=srcB, acc=acc, **operands)
    exe._exec(I.decode_inst(w))


def _load_golden(op: str):
    d = os.path.join(GOLDEN, CACHE, f"L{LAYER:02d}_{op}")
    return (np.load(os.path.join(d, "inputs.npz")),
            np.load(os.path.join(d, "outputs.npz")))


def _hf_tensor(key: str) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(MODEL_SAFETENSORS, framework="pt") as f:
        return f.get_tensor(key).float().numpy()


# --------------------------------------------------------------------------- #
# field-level assertions (02-isa §3 / §7 / §8)
# --------------------------------------------------------------------------- #
VECTOR_OPS = {
    "VADD": 0x80, "VSUB": 0x81, "VMUL": 0x82, "VDIV": 0x83, "VRECIP": 0x84,
    "VEXP": 0x85, "VRSQRT": 0x86, "VSILU": 0x87, "VMAX": 0x88, "VMOV": 0x89,
    "VSCALE": 0x8A, "VMASK": 0x8B, "VREDUCE_SUM": 0x8C, "VREDUCE_MAX": 0x8D,
    "ROPE": 0x8E, "RMSNORM": 0x8F, "QUANT": 0x90, "DEQUANT": 0x91,
}
KV_OPS = {
    "KV.APPEND": 0xC0, "KV.STORE_BLOCK": 0xC1, "KV.LOAD": 0xC2,
    "KV.GATHER": 0xC3,
}


def test_vector_opcodes():
    for mn, op in VECTOR_OPS.items():
        assert I.OPCODE_BY_NAME[mn] == op
        assert I._engine_for_opcode(op) == 0x03, mn
        d = I.decode_inst(I.encode_inst(mn))
        assert d["engine"] == 0x03 and d["engine_tag_valid"], mn
    assert len(VECTOR_OPS) == 18


def test_kv_opcodes():
    for mn, op in KV_OPS.items():
        assert I.OPCODE_BY_NAME[mn] == op
        assert I._engine_for_opcode(op) == 0x04, mn
        d = I.decode_inst(I.encode_inst(mn))
        assert d["engine"] == 0x04 and d["engine_tag_valid"], mn
    assert len(KV_OPS) == 4


def test_vector_field_layout():
    # §7.1: [103:98] ARa, [97:92] ARb, [91:86] ARd, [85:70] len,
    #       [69:65] CV, [64:33] imm
    d = I.decode_inst(I.encode_inst(
        "VADD", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
        ARa=3, ARb=4, ARd=5, len=4096, CV=7, imm=0xDEADBEEF))
    assert (d["ARa"], d["ARb"], d["ARd"]) == (3, 4, 5)
    assert d["len"] == 4096 and d["CV"] == 7 and d["imm"] == 0xDEADBEEF
    assert (d["srcA"], d["srcB"], d["acc"]) == (I.DT_BF16, I.DT_BF16, I.ACC_FP32)


def test_kv_append_field_layout():
    # §8.1: srcK(6@98) srcV(6@92) layer(6@86) head(3@83)
    d = I.decode_inst(I.encode_inst(
        "KV.APPEND", srcK=1, srcV=2, layer=27, head=5))
    assert (d["srcK"], d["srcV"], d["layer"], d["head"]) == (1, 2, 27, 5)


def test_kv_store_block_field_layout():
    # §8.2: + pos_start(13@70) count(14@56)
    d = I.decode_inst(I.encode_inst(
        "KV.STORE_BLOCK", srcK=1, srcV=2, layer=27, head=5,
        pos_start=8191, count=128))
    assert (d["pos_start"], d["count"]) == (8191, 128)


def test_kv_load_field_layout():
    # §8.3: dstK dstV layer head sel(2@81) pos_start(13@68) count(14@54)
    d = I.decode_inst(I.encode_inst(
        "KV.LOAD", dstK=1, dstV=2, layer=27, head=5,
        sel=2, pos_start=100, count=2048))
    assert (d["dstK"], d["dstV"], d["layer"], d["head"]) == (1, 2, 27, 5)
    assert (d["sel"], d["pos_start"], d["count"]) == (2, 100, 2048)


def test_kv_gather_field_layout():
    # §8.4: dst(6@98) dst2(6@92) layer head sel(1@82) broadcast(1@81)
    #       pos_start(13@68) count(14@54) Cstride(5@49)
    d = I.decode_inst(I.encode_inst(
        "KV.GATHER", dst=3, layer=27, head=2, sel=1, broadcast=1,
        pos_start=100, count=512, Cstride=7))
    assert (d["dst"], d["layer"], d["head"]) == (3, 27, 2)
    assert (d["sel"], d["broadcast"]) == (1, 1)
    assert (d["pos_start"], d["count"], d["Cstride"]) == (100, 512, 7)


def test_vector_kv_roundtrip():
    for mn in list(VECTOR_OPS) + list(KV_OPS):
        w = I.encode_inst(mn, srcA=1, srcB=2, acc=1)
        d = I.decode_inst(w)
        assert d["mnemonic"] == mn and d["engine_tag_valid"]


def test_illegal_opcode_rejected():
    # reserved VECTOR (0x92) and KV (0xC4) opcodes -> decode raises
    for op in (0x92, 0xC4):
        word = (0x03 << 120) | (op << 112)
        try:
            I.decode_inst(word)
            raise AssertionError(f"reserved opcode 0x{op:02X} not rejected")
        except ValueError:
            pass


def test_engine_tag_mismatch_flagged():
    # VECTOR opcode with DMA engine tag -> decode flags disagreement
    w = I.encode_inst("VADD")
    w = (w & ~(0xFF << 120)) | (0x01 << 120)  # force DMA engine tag
    d = I.decode_inst(w)
    assert not d["engine_tag_valid"]


def test_field_out_of_range_rejected():
    # len is 16b: 65536 must be rejected by the encoder
    try:
        I.encode_inst("VADD", ARa=0, ARb=0, ARd=0, len=1 << 16)
        raise AssertionError("len overflow not rejected")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# primitive numeric tests (numpy reference, fp32 internal + bf16 writeback)
# --------------------------------------------------------------------------- #
def _vec_primitive(op, srcA, srcB, acc, a, b, **kw):
    """Run a single VECTOR op on bf16 inputs, return bf16-exact fp32 output."""
    exe = Executor()
    n = a.size
    exe.AR[0] = 0x1000
    exe.AR[1] = 0x2000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(a))
    if b is not None:
        exe.write_bytes("sram", 0x20000, _bf16b(b))
    _run(exe, op, srcA, srcB, acc, 0, 1, 2, n, **kw)
    return _read_bf16(exe, 0x3000, n)


def test_binary_elementwise():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)
    a = a.astype(BF16).astype(np.float32)
    b = b.astype(BF16).astype(np.float32)
    for op, ref in [("VADD", a + b), ("VSUB", a - b), ("VMUL", a * b),
                    ("VDIV", a / b), ("VMAX", np.maximum(a, b))]:
        got = _vec_primitive(op, I.DT_BF16, I.DT_BF16, I.ACC_FP32, a, b)
        ref_bf16 = ref.astype(BF16).astype(np.float32)
        assert _ulp(got, ref_bf16).max() <= 1, op


def test_binary_scalar_broadcast():
    """cv field != 0 -> ARb scalar b[0] broadcast across all len lanes (the
    softmax program-emitted form: VSUB es = scores - mrun / VMUL ctx *= alpha,
    where ARb holds a per-head scalar and cv points at a non-zero broadcast C
    reg). Dispatch is on the cv FIELD, independent of the C register value."""
    rng = np.random.default_rng(6)
    a = rng.standard_normal(64).astype(np.float32)
    a = a.astype(BF16).astype(np.float32)
    s = a[3]  # scalar operand (e.g. running max / exp scale)
    for op, ref in [("VSUB", a - s), ("VMUL", a * s),
                    ("VADD", a + s), ("VMAX", np.maximum(a, s))]:
        exe = Executor()
        exe.AR[0] = 0x1000
        exe.AR[1] = 0x2000
        exe.AR[2] = 0x3000
        exe.write_bytes("sram", 0x10000, _bf16b(a))
        # ARb holds a single scalar element; cv field != 0 selects broadcast.
        # C[CV] is left 0 to prove the dispatch keys on the field, not the value.
        exe.write_bytes("sram", 0x20000, _bf16b(np.array([s], np.float32)))
        _run(exe, op, I.DT_BF16, I.DT_BF16, I.ACC_FP32, 0, 1, 2, a.size,
             CV=7, Cval=0)
        got = _read_bf16(exe, 0x3000, a.size)
        ref_bf16 = ref.astype(BF16).astype(np.float32)
        assert _ulp(got, ref_bf16).max() <= 1, op


def test_binary_cv0_continuous_with_nonzero_creg():
    """Regression: cv field == 0 selects contiguous len-element ARb even when
    the pointed-at C[0] register holds a non-zero value. (Pre-arbitration code
    keyed on C[cv] value and would wrongly broadcast here.)"""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)
    a = a.astype(BF16).astype(np.float32)
    b = b.astype(BF16).astype(np.float32)
    for op, ref in [("VADD", a + b), ("VSUB", a - b), ("VMUL", a * b),
                    ("VDIV", a / b), ("VMAX", np.maximum(a, b))]:
        exe = Executor()
        exe.AR[0] = 0x1000
        exe.AR[1] = 0x2000
        exe.AR[2] = 0x3000
        exe.write_bytes("sram", 0x10000, _bf16b(a))
        # ARb holds len contiguous elements; cv field == 0 selects contiguous,
        # and C[0] is deliberately non-zero to guard against value-based dispatch.
        exe.write_bytes("sram", 0x20000, _bf16b(b))
        _run(exe, op, I.DT_BF16, I.DT_BF16, I.ACC_FP32, 0, 1, 2, a.size,
             CV=0, Cval=12345)
        got = _read_bf16(exe, 0x3000, a.size)
        ref_bf16 = ref.astype(BF16).astype(np.float32)
        assert _ulp(got, ref_bf16).max() <= 1, op


def test_unary_elementwise():
    rng = np.random.default_rng(1)
    a = (np.abs(rng.standard_normal(64).astype(np.float32)) * 2.0 + 0.5)
    a = a.astype(BF16).astype(np.float32)  # strictly positive (VRSQRT domain)
    for op, ref in [("VRECIP", 1.0 / a), ("VEXP", np.exp(a)),
                    ("VRSQRT", 1.0 / np.sqrt(a)),
                    ("VSILU", a * (1.0 / (1.0 + np.exp(-a)))),
                    ("VMOV", a)]:
        got = _vec_primitive(op, I.DT_BF16, 0, I.ACC_FP32, a, None)
        ref_bf16 = ref.astype(BF16).astype(np.float32)
        assert _ulp(got, ref_bf16).max() <= 1, op


def test_vscale_bf16_scalar():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(32).astype(np.float32)
    a = a.astype(BF16).astype(np.float32)
    s = np.float32(0.08838834764831845)  # ATTN_SCALE
    s_bits = struct.unpack("<H", np.array([s], dtype=BF16).tobytes())[0]
    exe = Executor()
    exe.AR[0] = 0x1000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(a))
    _run(exe, "VSCALE", I.DT_BF16, 0, I.ACC_INT32, 0, 0, 2, a.size, imm=s_bits)
    got = _read_bf16(exe, 0x3000, a.size)
    ref = (a * np.float32(s)).astype(BF16).astype(np.float32)
    assert _ulp(got, ref).max() <= 1


def test_vscale_fp32_scalar():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(32).astype(np.float32)
    a = a.astype(BF16).astype(np.float32)
    s = np.float32(1.0 / 128.0)
    exe = Executor()
    exe.AR[0] = 0x1000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(a))
    _run(exe, "VSCALE", I.DT_BF16, 0, I.ACC_FP32, 0, 0, 2, a.size,
         CV=4, Cval=_fp32_bits(s))
    got = _read_bf16(exe, 0x3000, a.size)
    ref = (a * s).astype(BF16).astype(np.float32)
    assert _ulp(got, ref).max() <= 1


def test_vmask_causal():
    exe = Executor()
    exe.AR[2] = 0x3000
    # C[5] = [31:16] col_base, [15:0] row_base
    exe.C[5] = (2 << 16) | 3   # col_base=2, row_base=3
    rows, cols = 4, 5
    imm = (rows << 16) | cols
    _run(exe, "VMASK", I.DT_BF16, 0, I.ACC_FP32, 0, 0, 2, 0, CV=5, imm=imm)
    got = _read_bf16(exe, 0x3000, rows * cols).reshape(rows, cols)
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    ref = np.where((2 + j) <= (3 + i), 0.0, -np.inf).astype(BF16).astype(np.float32)
    assert np.array_equal(got, ref)


def test_reduce_sum_max():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((6, 128)).astype(np.float32)
    a = a.astype(BF16).astype(np.float32)
    exe = Executor()
    exe.AR[0] = 0x1000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(a.reshape(-1)))
    for op, ref in [("VREDUCE_SUM", a.sum(axis=1)),
                    ("VREDUCE_MAX", a.max(axis=1))]:
        _run(exe, op, I.DT_BF16, 0, I.ACC_FP32, 0, 0, 2, 128, CV=6, Cval=6)
        got = _read_bf16(exe, 0x3000, 6)
        ref_bf16 = ref.astype(BF16).astype(np.float32)
        assert _ulp(got, ref_bf16).max() <= 1, op


def test_quant_dequant_roundtrip():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(256).astype(np.float32) * 0.5
    a = a.astype(BF16).astype(np.float32)
    s = np.float32(0.01)
    s_stored = float(np.array([s], dtype=BF16)[0])  # what the executor reads back
    # scale descriptor CD: per-tensor (mode=0), BF16 scale (scale_dtype=0),
    # scale_base word addr -> write s as bf16 at SRAM 0x8000 (word)
    exe = Executor()
    exe.write_bytes("sram", 0x80000, np.array([s], dtype=BF16).tobytes())
    cd = (0 << 20) | (0 << 19) | 0x8000  # mode=0, bf16, scale_base=0x8000 words
    exe.AR[0] = 0x1000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(a))
    # QUANT: fp -> INT8 (srcB=0 -> INT8)
    _run(exe, "QUANT", I.DT_BF16, I.DT_INT8, I.ACC_INT32, 0, 0, 2, a.size, CV=5, Cval=cd)
    q = np.frombuffer(exe.read_bytes("sram", 0x30000, a.size), dtype=np.int8)
    ref_q = np.clip(np.round(a / s_stored), -127, 127).astype(np.int8)
    assert np.array_equal(q, ref_q)
    # DEQUANT: INT8 -> BF16 (srcB=0 -> BF16)
    exe.AR[0] = 0x3000
    exe.AR[2] = 0x5000
    # "落盘 BF16" guard: canary right after the output; a fp32 (4 B/elem)
    # writeback would clobber it, so this asserts DEQUANT writes 2 B/elem.
    canary = 0x50000 + a.size * 2
    exe.write_bytes("sram", canary, b"\xDE\xAD\xBE\xEF")
    _run(exe, "DEQUANT", I.DT_INT8, I.DT_BF16, I.ACC_FP32, 0, 0, 2, a.size, CV=5, Cval=cd)
    assert exe.read_bytes("sram", canary, 4) == b"\xDE\xAD\xBE\xEF"
    dq = _read_bf16(exe, 0x5000, a.size)
    ref_dq = (ref_q.astype(np.float32) * s_stored).astype(BF16).astype(np.float32)
    assert _ulp(dq, ref_dq).max() <= 1


# --------------------------------------------------------------------------- #
# single-op golden comparisons (plan §1 ULP criterion)
# --------------------------------------------------------------------------- #
def test_rmsnorm_normal_vs_golden():
    inp, out = _load_golden("rmsnorm_in")
    x = inp["x"].astype(np.float32)          # [1,1024]
    y = out["y"].astype(np.float32)
    gamma = _hf_tensor(f"model.layers.{LAYER}.input_layernorm.weight")
    exe = Executor()
    exe.AR[0] = 0x1000
    exe.AR[1] = 0x2000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(x.reshape(-1)))
    exe.write_bytes("sram", 0x20000, _bf16b(gamma))
    _run(exe, "RMSNORM", I.DT_BF16, I.DT_BF16, I.ACC_FP32, 0, 1, 2, x.size,
         CV=5, Cval=_fp32_bits(EPS), imm=0)  # mode=normal
    got = _read_bf16(exe, 0x3000, x.size)
    d = _ulp(got, y.reshape(-1))
    assert d.max() <= 1, f"RMSNorm max_ulp={d.max():.2f}"


def test_qknorm_per_head_vs_golden():
    inp, out = _load_golden("attn_qknorm")
    q = inp["q"].astype(np.float32)          # [1,2048]
    k = inp["k"].astype(np.float32)          # [1,1024]
    q_out = out["q"].astype(np.float32)      # [16,1,128]
    k_out = out["k"].astype(np.float32)      # [8,1,128]
    qg = _hf_tensor(f"model.layers.{LAYER}.self_attn.q_norm.weight")  # [128]
    kg = _hf_tensor(f"model.layers.{LAYER}.self_attn.k_norm.weight")  # [128]
    exe = Executor()
    exe.AR[0] = 0x1000
    exe.AR[1] = 0x2000
    exe.AR[2] = 0x3000
    # per-head RMSNorm: gamma tiled per 128-element head
    exe.write_bytes("sram", 0x10000, _bf16b(q.reshape(-1)))
    exe.write_bytes("sram", 0x20000, _bf16b(np.tile(qg, 16)))
    _run(exe, "RMSNORM", I.DT_BF16, I.DT_BF16, I.ACC_FP32, 0, 1, 2, q.size,
         CV=5, Cval=_fp32_bits(EPS), imm=(1 << 31))  # mode=per-head
    got_q = _read_bf16(exe, 0x3000, q.size).reshape(16, 128)
    assert _ulp(got_q, q_out.reshape(16, 128)).max() <= 1

    exe.write_bytes("sram", 0x10000, _bf16b(k.reshape(-1)))
    exe.write_bytes("sram", 0x20000, _bf16b(np.tile(kg, 8)))
    _run(exe, "RMSNORM", I.DT_BF16, I.DT_BF16, I.ACC_FP32, 0, 1, 2, k.size,
         CV=5, Cval=_fp32_bits(EPS), imm=(1 << 31))
    got_k = _read_bf16(exe, 0x3000, k.size).reshape(8, 128)
    assert _ulp(got_k, k_out.reshape(8, 128)).max() <= 1


def test_rope_vs_golden():
    inp, out = _load_golden("attn_rope")
    q = inp["q"].astype(np.float32)          # [16,1,128]
    k = inp["k"].astype(np.float32)          # [8,1,128]
    q_out = out["q"].astype(np.float32)
    k_out = out["k"].astype(np.float32)
    pos = 1024
    exe = Executor()
    exe.AR[0] = 0x1000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(q.reshape(-1)))
    _run(exe, "ROPE", I.DT_BF16, 0, I.ACC_FP32, 0, 0, 2, q.size,
         CV=6, Cval=_fp32_bits(ROPE_THETA), imm=pos)
    got_q = _read_bf16(exe, 0x3000, q.size).reshape(16, 128)
    assert _ulp(got_q, q_out.reshape(16, 128)).max() <= 1

    exe.write_bytes("sram", 0x10000, _bf16b(k.reshape(-1)))
    _run(exe, "ROPE", I.DT_BF16, 0, I.ACC_FP32, 0, 0, 2, k.size,
         CV=6, Cval=_fp32_bits(ROPE_THETA), imm=pos)
    got_k = _read_bf16(exe, 0x3000, k.size).reshape(8, 128)
    assert _ulp(got_k, k_out.reshape(8, 128)).max() <= 1


def test_softmax_vs_golden():
    inp, out = _load_golden("attn_softmax")
    scores = inp["scores"].astype(np.float32).reshape(16, 1025)
    mask = inp["mask"].astype(np.float32).reshape(1, 1025)
    probs = out["probs"].astype(np.float32).reshape(16, 1025)
    assert np.all(mask == 0), "decode mask must be all-zero"

    exe = Executor()
    exe.AR[0] = 0x1000   # scores
    exe.AR[1] = 0x2000   # maxv (16)
    exe.AR[2] = 0x3000   # es
    exe.AR[3] = 0x4000   # sumv (16)
    exe.AR[4] = 0x5000   # rinv (16)
    exe.AR[5] = 0x6000   # softmax
    exe.AR[6] = 0x7000   # maxv broadcast
    exe.AR[7] = 0x8000   # rinv broadcast
    exe.write_bytes("sram", 0x10000, _bf16b(scores.reshape(-1)))

    _run(exe, "VREDUCE_MAX", I.DT_BF16, 0, I.ACC_FP32, 0, 0, 1, 1025, CV=9, Cval=16)
    maxv = _read_bf16(exe, 0x2000, 16)
    maxv_b = np.repeat(maxv[:, None], 1025, axis=1)
    exe.write_bytes("sram", 0x70000, _bf16b(maxv_b.reshape(-1)))
    _run(exe, "VSUB", I.DT_BF16, I.DT_BF16, I.ACC_FP32, 0, 6, 0, 16 * 1025)
    _run(exe, "VEXP", I.DT_BF16, 0, I.ACC_FP32, 0, 0, 2, 16 * 1025)
    _run(exe, "VREDUCE_SUM", I.DT_BF16, 0, I.ACC_FP32, 2, 0, 3, 1025, CV=9, Cval=16)
    _run(exe, "VRECIP", I.DT_BF16, 0, I.ACC_FP32, 3, 0, 4, 16)
    rinv = _read_bf16(exe, 0x5000, 16)
    rinv_b = np.repeat(rinv[:, None], 1025, axis=1)
    exe.write_bytes("sram", 0x80000, _bf16b(rinv_b.reshape(-1)))
    _run(exe, "VMUL", I.DT_BF16, I.DT_BF16, I.ACC_FP32, 2, 7, 5, 16 * 1025)

    got = _read_bf16(exe, 0x6000, 16 * 1025).reshape(16, 1025)
    # argmax must match exactly (sampling correctness)
    assert (np.argmax(got, axis=1) == np.argmax(probs, axis=1)).all()
    # top-1 (dominant) probability within 1 ULP
    am = np.argmax(probs, axis=1)
    top_ulp = _ulp(got[np.arange(16), am], probs[np.arange(16), am])
    assert top_ulp.max() <= 1, f"top-1 prob ulp={top_ulp.max():.2f}"


def test_swiglu_vs_golden():
    inp, out = _load_golden("mlp_silu")
    gate = inp["gate"].astype(np.float32)    # [1,3072]
    up = inp["up"].astype(np.float32)
    y = out["y"].astype(np.float32)
    exe = Executor()
    exe.AR[0] = 0x1000   # gate
    exe.AR[1] = 0x4000   # up
    exe.AR[2] = 0x5000   # h = silu(gate)
    exe.AR[3] = 0x6000   # y = h*up
    exe.write_bytes("sram", 0x10000, _bf16b(gate.reshape(-1)))
    exe.write_bytes("sram", 0x40000, _bf16b(up.reshape(-1)))
    _run(exe, "VSILU", I.DT_BF16, 0, I.ACC_FP32, 0, 0, 2, gate.size)
    _run(exe, "VMUL", I.DT_BF16, I.DT_BF16, I.ACC_FP32, 2, 1, 3, gate.size)
    got = _read_bf16(exe, 0x6000, gate.size)
    assert _ulp(got, y.reshape(-1)).max() <= 1


def test_attention_score_scale_vs_golden():
    # attention scores = QK^T (BMM, MATRIX) then ×ATTN_SCALE (VSCALE, VECTOR).
    # Here we validate the VECTOR VSCALE leg against golden: raw fp32 scores
    # (scores_ref / scale) -> bf16 -> VSCALE -> bf16 scores.
    inp, out = _load_golden("attn_score")
    scores = out["scores"].astype(np.float32)        # [16,1,1025] bf16
    scores_ref = out["scores_ref"].astype(np.float32)  # fp32
    scale = np.float32(128.0 ** -0.5)
    raw = (scores_ref / scale).astype(BF16).astype(np.float32)
    s_bits = struct.unpack("<H", np.array([scale], dtype=BF16).tobytes())[0]

    exe = Executor()
    exe.AR[0] = 0x1000
    exe.AR[2] = 0x3000
    exe.write_bytes("sram", 0x10000, _bf16b(raw.reshape(-1)))
    _run(exe, "VSCALE", I.DT_BF16, 0, I.ACC_INT32, 0, 0, 2, raw.size, imm=s_bits)
    got = _read_bf16(exe, 0x3000, raw.size)
    assert _ulp(got, scores.reshape(-1)).max() <= 1


# --------------------------------------------------------------------------- #
# KV numeric tests
# --------------------------------------------------------------------------- #
def _kv_setup(exe: Executor, slab_shift: int = 21):
    exe.C[exe.C_SLAB_SHIFT] = slab_shift
    exe.AR[exe.AR_KV_BASE] = (1 << 63) | 0  # AR_KV_BASE = HBM addr 0


def test_kv_append_load_roundtrip():
    exe = Executor()
    _kv_setup(exe)
    exe.C[exe.C_KV_POS] = 3
    k = np.arange(128, dtype=np.float32).astype(BF16)
    v = (np.arange(128, dtype=np.float32) + 1000).astype(BF16)
    exe.AR[3] = 0x2000
    exe.AR[4] = 0x2010
    exe.AR[5] = 0x3000
    exe.AR[6] = 0x3010
    exe.write_bytes("sram", 0x20000, _bf16b(k))
    exe.write_bytes("sram", 0x20100, _bf16b(v))
    _run(exe, "KV.APPEND", srcK=3, srcV=4, layer=27, head=5)
    _run(exe, "KV.LOAD", dstK=5, dstV=6, layer=27, head=5, sel=2,
         pos_start=3, count=1)
    assert np.array_equal(_read_bf16(exe, 0x3000, 128), k.astype(np.float32))
    assert np.array_equal(_read_bf16(exe, 0x3010, 128), v.astype(np.float32))


def test_kv_store_block_roundtrip():
    exe = Executor()
    _kv_setup(exe)
    count = 8
    k = np.random.default_rng(6).standard_normal((count, 128)).astype(np.float32)
    v = np.random.default_rng(7).standard_normal((count, 128)).astype(np.float32)
    k = k.astype(BF16)
    v = v.astype(BF16)
    exe.AR[3] = 0x2000
    exe.AR[4] = 0x4000
    exe.AR[5] = 0x6000
    exe.AR[6] = 0x7000
    exe.write_bytes("sram", 0x20000, k.tobytes())
    exe.write_bytes("sram", 0x40000, v.tobytes())
    _run(exe, "KV.STORE_BLOCK", srcK=3, srcV=4, layer=3, head=7,
         pos_start=4, count=count)
    _run(exe, "KV.LOAD", dstK=5, dstV=6, layer=3, head=7, sel=2,
         pos_start=4, count=count)
    kk = np.frombuffer(exe.read_bytes("sram", 0x60000, count * 256),
                       dtype=BF16).reshape(count, 128)
    vv = np.frombuffer(exe.read_bytes("sram", 0x70000, count * 256),
                       dtype=BF16).reshape(count, 128)
    assert np.array_equal(kk, k)
    assert np.array_equal(vv, v)


def test_kv_gather_broadcast():
    exe = Executor()
    _kv_setup(exe)
    count = 4
    k = np.random.default_rng(8).standard_normal((count, 128)).astype(np.float32)
    k = k.astype(BF16)
    # write K window via STORE_BLOCK first
    exe.AR[3] = 0x2000
    exe.write_bytes("sram", 0x20000, k.tobytes())
    _run(exe, "KV.STORE_BLOCK", srcK=3, srcV=3, layer=9, head=1,
         pos_start=0, count=count)
    # GATHER broadcast ×4: dst at word 0x3000; Cstride = count×16 words
    # (Q-head copy stride = count tokens × 256 B, so the 4 copies don't overlap)
    exe.C[10] = count * 16
    exe.AR[7] = 0x3000
    _run(exe, "KV.GATHER", dst=7, layer=9, head=1, sel=0,
         broadcast=1, pos_start=0, count=count, Cstride=10)
    for i in range(4):
        got = np.frombuffer(
            exe.read_bytes("sram", 0x30000 + i * count * 256, count * 256),
            dtype=BF16).reshape(count, 128)
        assert np.array_equal(got, k), f"broadcast copy {i}"


def test_kv_slab_address_formula():
    # 05 §1.3: slab_index = (layer<<4)|(head<<1)|kv; addr = base + slab_index<<shift
    exe = Executor()
    exe.C[exe.C_SLAB_SHIFT] = 21
    exe.AR[exe.AR_KV_BASE] = (1 << 63) | (4 << 20)  # 4 MiB-aligned base
    assert exe._kv_slab_base(27, 5, 0) == (4 << 20) + (((27 << 4) | (5 << 1) | 0) << 21)
    assert exe._kv_slab_base(27, 5, 1) == (4 << 20) + (((27 << 4) | (5 << 1) | 1) << 21)
    # SLAB_SHIFT=20 (BF16 4K mode): slab step 2^20
    exe.C[exe.C_SLAB_SHIFT] = 20
    assert exe._kv_slab_base(0, 0, 0) == (4 << 20) + 0
    assert exe._kv_slab_base(0, 1, 0) == (4 << 20) + (2 << 20)
    # SLAB_SHIFT=22 (4 MiB slab, 16K token capacity, pos still 13b): slab step 2^22
    exe.C[exe.C_SLAB_SHIFT] = 22
    assert exe._kv_slab_base(0, 0, 0) == (4 << 20) + 0
    assert exe._kv_slab_base(0, 1, 0) == (4 << 20) + (2 << 22)
    assert exe._kv_slab_base(27, 5, 0) == (4 << 20) + (((27 << 4) | (5 << 1) | 0) << 22)
    assert exe._kv_slab_base(27, 5, 1) == (4 << 20) + (((27 << 4) | (5 << 1) | 1) << 22)
    # pos=8192 (the 8193rd slot) is addressable under slab_shift=22 without
    # wrapping: KV.LOAD's 13-bit pos_start still caps at 8191, but the slab
    # itself holds 16K tokens. Roundtrip a K row through KV.APPEND at pos 8192.
    exe.AR[3] = 0x2000
    exe.AR[4] = 0x2010
    k22 = np.arange(128, dtype=np.float32).astype(BF16)
    v22 = (np.arange(128, dtype=np.float32) + 7).astype(BF16)
    exe.write_bytes("sram", 0x20000, _bf16b(k22))
    exe.write_bytes("sram", 0x20100, _bf16b(v22))
    exe.C[exe.C_KV_POS] = 8192
    _run(exe, "KV.APPEND", srcK=3, srcV=4, layer=0, head=0)
    got_k = np.frombuffer(
        exe.read_bytes("hbm", exe._kv_slab_base(0, 0, 0) + (8192 << 8), 256),
        dtype=BF16).astype(np.float32)
    got_v = np.frombuffer(
        exe.read_bytes("hbm", exe._kv_slab_base(0, 0, 1) + (8192 << 8), 256),
        dtype=BF16).astype(np.float32)
    assert np.array_equal(got_k, k22.astype(np.float32))
    assert np.array_equal(got_v, v22.astype(np.float32))




def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} vector/KV assertion groups passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
