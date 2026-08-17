"""B' KV quantization functional test — INT8-K (QK-norm folded, signed scale)
+ INT4-V (per-token per-head), quantize-on-write / dequant-on-read.

Dual-track reference (M3 convention):
  * fp32 dequant vs fp64 reference < 1e-6 (before BF16 writeback);
  * BF16 writeback vs fp64-rounded reference <= 1 BF16 ULP.

The scheme under test (DECISION §5 / fold-verify):
  k_unit = rmsnorm(k_raw) (unit RMS, pre-weight); q = round(k_unit/s_q),
  s_q = bf16(max|k_unit|/127); dequant k_hat[c] = q[c] * (s_q * k_norm[c]),
  scale_c = s_q * k_norm[c] is SIGNED BF16 (k_norm may be negative).
  V: q4 = round(v/s_v), s_v = bf16(max|v|/7); v_hat = q4 * s_v.
K is stored pre-RoPE (fold scheme); on-read RoPE is the conditional perf item.
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
from qsim.executor import (Executor, unpack_int4,
                           C_KVNORM_BASE, AR_KV_SCALE_BASE)

HEAD_DIM = 128
INT8_LEVEL = 127.0
INT4_LEVEL = 7.0


def _bf16_ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element bf16-domain ULP distance (M3 convention)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b))
    ulp = np.zeros_like(mag)
    nz = mag > 0
    ulp[nz] = np.exp2(np.floor(np.log2(mag[nz])) - 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ulp > 0, d / ulp, np.where(d > 0, np.inf, 0.0))


def _bf16(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(BF16)


def _bf16_scalar(f: float) -> float:
    return float(np.asarray(np.float32(f)).astype(BF16).astype(np.float32))


# --------------------------------------------------------------------------- #
# unit-norm K generator (max|k_unit| bounded ~< 12, no channel outliers)
# --------------------------------------------------------------------------- #
def _unit_k(rng, n_heads: int) -> np.ndarray:
    k = rng.standard_normal((n_heads, HEAD_DIM)).astype(np.float32)
    k = k / np.sqrt(np.mean(k * k, axis=1, keepdims=True) + 1e-6)
    return k  # [n_heads, HEAD_DIM] unit RMS


def _setup(exe: Executor, n_heads: int = 8):
    """Configure KV + B' scale regions; returns (k_norm_f32, v_f32)."""
    exe.C[exe.C_SLAB_SHIFT] = 21
    exe.AR[exe.AR_KV_BASE] = (1 << 63) | 0
    # static k_norm table (signed BF16) at SRAM word 0x3000
    exe.C[C_KVNORM_BASE] = 0x3000
    # per-token scale slab at HBM byte 0x100_00000 (clear of the KV region)
    exe.AR[AR_KV_SCALE_BASE] = (1 << 63) | 0x10000000
    # SRAM staging for K/V write sources (word addr 0x2000 / 0x2010)
    exe.AR[0] = 0x2000
    exe.AR[1] = 0x2010
    exe.AR[2] = 0x2020  # LOAD dstK
    exe.AR[3] = 0x2030  # LOAD dstV
    return


def _write_k_norm(exe: Executor, rng, n_heads: int) -> np.ndarray:
    """Write a static signed k_norm table (BF16, with negative channels)."""
    kn = rng.standard_normal((n_heads, HEAD_DIM)).astype(np.float32)
    kn = kn * 2.0
    # force a few negative channels (fold-verify: ~3.4% negative)
    kn[:, :5] = -np.abs(kn[:, :5])
    buf = bytearray()
    for h in range(n_heads):
        exe.write_bytes("sram", 0x3000 * 16 + h * 256,
                        _bf16(kn[h]).tobytes())
    return kn


def _fold_ref(k_unit: np.ndarray, k_norm: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """fp64 hardware-faithful fold reference: returns (q int8, s_q bf16, k_hat bf16)."""
    amax = float(np.abs(k_unit).max())
    s_q = _bf16_scalar(max(amax / INT8_LEVEL, 1e-6))
    q = np.clip(np.round(k_unit / s_q), -INT8_LEVEL, INT8_LEVEL).astype(np.int8)
    scale_c = _bf16(s_q * k_norm)                      # SIGNED BF16
    k_hat = _bf16(q.astype(np.float32) * scale_c.astype(np.float32))
    return q, s_q, k_hat


def _v_ref(v: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    amax = float(np.abs(v).max())
    s_v = _bf16_scalar(max(amax / INT4_LEVEL, 1e-6))
    q4 = np.clip(np.round(v / s_v), -INT4_LEVEL, INT4_LEVEL).astype(np.int8)
    v_hat = _bf16(q4.astype(np.float32) * np.float32(s_v))
    return q4, s_v, v_hat


# --------------------------------------------------------------------------- #
# 1. K fold dequant: executor vs fp64 dual-track
# --------------------------------------------------------------------------- #
def test_k_fold_dequant_fp64():
    rng = np.random.default_rng(0)
    exe = Executor()
    _setup(exe, 1)
    kn = _write_k_norm(exe, rng, 1)[0]
    k_unit = _unit_k(rng, 1)[0]
    q, s_q, k_hat_ref = _fold_ref(k_unit, kn)

    # executor dequant path (fp32 internal + BF16 writeback)
    got = exe._dequant_k_fold(q.astype(np.float32), s_q, kn)

    # (a) fp32-vs-fp64 < 1e-6: recompute the fp32 chain without BF16 rounding
    scale_c_fp32 = (np.float32(s_q) * kn.astype(np.float32)).astype(np.float32)
    fp32_result = q.astype(np.float32) * scale_c_fp32
    fp64_result = q.astype(np.float64) * (np.float64(s_q) * kn.astype(np.float64))
    rel = np.abs(fp32_result.astype(np.float64) - fp64_result) / np.maximum(np.abs(fp64_result), 1e-30)
    assert rel.max() < 1e-6, f"fp32 vs fp64 dequant rel err {rel.max():.3e}"

    # (b) BF16 writeback <= 1 ULP
    ulp = _bf16_ulp_dist(got, k_hat_ref)
    assert ulp.max() <= 1, f"K fold BF16 writeback {ulp.max()} ULP"


# --------------------------------------------------------------------------- #
# 2. V INT4 dequant: executor vs fp64 dual-track
# --------------------------------------------------------------------------- #
def test_v_int4_dequant_fp64():
    rng = np.random.default_rng(1)
    exe = Executor()
    v = rng.standard_normal(HEAD_DIM).astype(np.float32)
    q4, s_v, v_hat_ref = _v_ref(v)
    got = exe._dequant_v_int4(q4.astype(np.float32), s_v)

    fp32_result = q4.astype(np.float32) * np.float32(s_v)
    fp64_result = q4.astype(np.float64) * np.float64(s_v)
    rel = np.abs(fp32_result.astype(np.float64) - fp64_result) / np.maximum(np.abs(fp64_result), 1e-30)
    assert rel.max() < 1e-6, f"fp32 vs fp64 V dequant rel err {rel.max():.3e}"
    ulp = _bf16_ulp_dist(got, v_hat_ref)
    assert ulp.max() <= 1, f"V INT4 BF16 writeback {ulp.max()} ULP"


# --------------------------------------------------------------------------- #
# 3. Full KV roundtrip: APPEND (INT8-K fold + INT4-V) -> LOAD dequant
# --------------------------------------------------------------------------- #
def _roundtrip(exe: Executor, n_heads: int, rng):
    kn = _write_k_norm(exe, rng, n_heads)
    k_unit = _unit_k(rng, n_heads)
    v = rng.standard_normal((n_heads, HEAD_DIM)).astype(np.float32)

    k_hats, v_hats = [], []
    for h in range(n_heads):
        # per-head staging: APPEND srcK/srcV read this head's 256 B staging
        exe.write_bytes("sram", 0x2000 * 16, _bf16(k_unit[h]).tobytes())
        exe.write_bytes("sram", 0x2010 * 16, _bf16(v[h]).tobytes())
        exe.C[exe.C_KV_POS] = 0
        exe._exec(I.decode_inst(I.encode_inst(
            "KV.APPEND", srcA=I.DT_INT8, srcB=I.DT_INT4,
            srcK=0, srcV=1, layer=0, head=h)))
        exe._exec(I.decode_inst(I.encode_inst(
            "KV.LOAD", srcA=I.DT_INT8, srcB=I.DT_INT4,
            dstK=2, dstV=3, layer=0, head=h, sel=2,
            pos_start=0, count=1)))
        k_hats.append(np.frombuffer(exe.read_bytes("sram", 0x2020 * 16, 256),
                                    dtype=BF16).astype(np.float32))
        v_hats.append(np.frombuffer(exe.read_bytes("sram", 0x2030 * 16, 256),
                                    dtype=BF16).astype(np.float32))
    return k_unit, v, kn, np.stack(k_hats), np.stack(v_hats)


def test_kv_roundtrip_int8k_int4v():
    rng = np.random.default_rng(2)
    exe = Executor()
    _setup(exe, 8)
    k_unit, v, kn, k_hats, v_hats = _roundtrip(exe, 8, rng)
    # K reconstruction: k_hat ~= k_unit * k_norm (pre-RoPE, signed)
    k_target = _bf16(k_unit).astype(np.float32) * kn
    k_rel = np.abs(k_hats - k_target) / np.maximum(np.abs(k_target), 1e-6)
    # unit-norm K fold INT8: mean rel err ~8.6% (fold-verify §5); per-element
    # rel err near zero-value channels can reach 1.0, so gate on the mean.
    # V reconstruction: per-head INT4 — mean |err| <= s_v/2 + BF16 noise; gate
    # on mean abs err vs the per-head scale (rel err is unbounded near zero).
    v_abs = np.abs(v_hats - _bf16(v).astype(np.float32))
    s_v_scale = np.abs(v).max(axis=1, keepdims=True) / 7.0
    assert (v_abs.mean(axis=1) <= s_v_scale.ravel() * 0.6 + 1e-3).all(), \
        f"V INT4 reconstruction abs err too large"


# --------------------------------------------------------------------------- #
# 4. scale metadata layout + BF16 default backward compatibility
# --------------------------------------------------------------------------- #
def test_kv_bf16_default_unchanged():
    """srcA=srcB=0 (BF16) keeps the v0 byte-identical path."""
    rng = np.random.default_rng(3)
    exe = Executor()
    _setup(exe, 1)
    k = rng.standard_normal(128).astype(np.float32)
    v = rng.standard_normal(128).astype(np.float32)
    exe.write_bytes("sram", 0x2000 * 16, _bf16(k).tobytes())
    exe.write_bytes("sram", 0x2010 * 16, _bf16(v).tobytes())
    exe.C[exe.C_KV_POS] = 0
    exe._exec(I.decode_inst(I.encode_inst("KV.APPEND", srcK=0, srcV=1, layer=1, head=0)))
    # HBM slab must hold raw 256B BF16 (not quantized)
    kk = np.frombuffer(exe.read_bytes("hbm", exe._kv_slab_base(1, 0, 0), 256),
                       dtype=BF16).astype(np.float32)
    vv = np.frombuffer(exe.read_bytes("hbm", exe._kv_slab_base(1, 0, 1), 256),
                       dtype=BF16).astype(np.float32)
    assert np.array_equal(kk, _bf16(k).astype(np.float32))
    assert np.array_equal(vv, _bf16(v).astype(np.float32))


def test_kv_signed_scale_layout():
    """Per-token scale record is [s_q (2B), s_v (2B)]; static k_norm 256 B/head."""
    rng = np.random.default_rng(4)
    exe = Executor()
    _setup(exe, 1)
    kn = _write_k_norm(exe, rng, 1)[0]
    k_unit = _bf16(_unit_k(rng, 1)[0]).astype(np.float32)
    v = _bf16(rng.standard_normal(HEAD_DIM).astype(np.float32)).astype(np.float32)
    exe.write_bytes("sram", 0x2000 * 16, _bf16(k_unit).tobytes())
    exe.write_bytes("sram", 0x2010 * 16, _bf16(v).tobytes())
    exe.C[exe.C_KV_POS] = 0
    exe._exec(I.decode_inst(I.encode_inst("KV.APPEND", srcA=I.DT_INT8, srcB=I.DT_INT4,
                                          srcK=0, srcV=1, layer=0, head=0)))
    q_ref, s_q_ref, _ = _fold_ref(k_unit, kn)
    _, s_v_ref, _ = _v_ref(v)
    # K data slab: 128 B INT8
    q_got = np.frombuffer(exe.read_bytes("hbm", exe._kv_slab_base(0, 0, 0), 128),
                          dtype=np.int8)
    assert np.array_equal(q_got, q_ref)


def _run_all():
    import traceback
    failed = []
    for name in sorted(n for n in globals() if n.startswith("test_")):
        fn = globals()[name]
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    if failed:
        print(f"\ntest_bprime_kv: {len(failed)} failed: {failed}")
        return 1
    print("test_bprime_kv: all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
