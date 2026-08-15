"""W8A8 symmetric quantization (per-128-K-group, weight side).

Weight: per-128-K-group symmetric scale sw = max|w_group|/127, INT8 clipped to
[-127, 127] (zero is representable; -128 reserved). Activation: per-tensor
symmetric scale sx = max|x|/127 (compile-time constant; the executor folds it
into the CD dequant scale as sx * sw). Reference: qsim/test_m2a.py (M2a INT8).

Output scale tensor layout: [N, G] row-major BF16 (G = K/128), matching the
executor's _gemm_dequant read (scale[n*G+g]).
"""

from __future__ import annotations

import numpy as np

try:
    import ml_dtypes
    _BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    _BF16 = np.float16

GROUP = 128


def quantize_weight(w: np.ndarray, group: int = GROUP) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-group weight quantize.

    w: [N, K] fp32. Returns (wqi [N,K] int8, sw [N,G] fp32).
    """
    N, K = w.shape
    assert K % group == 0, "K must be a multiple of group"
    G = K // group
    wr = w.reshape(N, G, group)
    sw = np.abs(wr).max(axis=2) / 127.0
    wqi = np.clip(np.round(wr / sw[:, :, None]), -127, 127).astype(np.int8)
    return wqi.reshape(N, K), sw.astype(np.float32)


def quantize_activation(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric activation quantize.

    x: [M, K] fp32. Returns (xq int8, sx float).
    """
    sx = float(np.abs(x).max() / 127.0)
    xq = np.clip(np.round(x / sx), -127, 127).astype(np.int8)
    return xq, sx


def combine_scale(sw: np.ndarray, sx: float) -> np.ndarray:
    """Fold activation scale into the per-group weight scale.

    sw: [N,G] fp32, sx: scalar. Returns cd [N,G] bf16 (= sx*sw).
    """
    return (sx * sw).astype(_BF16)


def scale_bytes(N: int, K: int, group: int = GROUP) -> int:
    """Per-group scale tensor byte size: [N, K/group] BF16."""
    return N * (K // group) * 2


def int8_group_partials(a_i8: np.ndarray, b_i8: np.ndarray,
                        group: int = GROUP) -> list[np.ndarray]:
    """Per-group INT32 partial sums (exact). a: [M,K], b: [K,N] (both int8)."""
    A = a_i8.astype(np.int32)
    B = b_i8.astype(np.int32)
    G = a_i8.shape[1] // group
    return [np.matmul(A[:, g * group:(g + 1) * group],
                      B[g * group:(g + 1) * group, :], dtype=np.int32)
            for g in range(G)]


# ---------------------------------------------------------------------------
# W4A16: symmetric per-128-K-group INT4 weight quantization + 2b packing.
# (plans/int4-plan.md §3 Q4a)
#
# Weight: per-128-K-group symmetric scale sw = max|w_group|/7, INT4 clipped to
# [-7, 7] (zero representable; -8 reserved). Activation stays BF16 (W4A16, no
# runtime QUANT). Packing bit-order is authoritative in qsim/executor.py
# _read_vector/_write_vector: even element -> low nibble, odd -> high nibble,
# two's-complement 4-bit (-8..7).
# ---------------------------------------------------------------------------

INT4_LEVEL = 7.0


def quantize_weight_int4(w: np.ndarray, group: int = GROUP
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-group INT4 weight quantize (round-to-nearest).

    w: [N, K] fp32. Returns (wqi [N,K] int8 in [-7,7], sw [N,G] fp32),
    sw = max|w_group| / 7.
    """
    N, K = w.shape
    assert K % group == 0, "K must be a multiple of group"
    G = K // group
    wr = w.reshape(N, G, group)
    sw = np.abs(wr).max(axis=2) / INT4_LEVEL
    wqi = np.clip(np.round(wr / sw[:, :, None]), -7, 7).astype(np.int8)
    return wqi.reshape(N, K), sw.astype(np.float32)


def pack_int4(wqi: np.ndarray) -> np.ndarray:
    """Pack [N,K] int8 (values -8..7) -> [N, K//2] uint8 (row-major).

    Even column -> low nibble, odd column -> high nibble, two's-complement
    4-bit. Bit-order matches executor _read_vector/_write_vector (authority).
    """
    wqi = np.asarray(wqi, np.int8)
    assert wqi.ndim == 2, "pack_int4 expects a [N,K] matrix"
    N, K = wqi.shape
    assert K % 2 == 0, "K must be even to pack 2 elements per byte"
    v = np.clip(wqi.astype(np.int32), -8, 7) & 0x0F
    return (v[:, 0::2] | (v[:, 1::2] << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack flat uint8 nibbles -> int8 [n] (sign-extended). Inverse of the
    row-major flat layout produced by pack_int4 (even index -> low nibble)."""
    packed = np.asarray(packed, np.uint8).reshape(-1)
    lo = (packed & 0x0F).astype(np.int32)
    hi = ((packed >> 4) & 0x0F).astype(np.int32)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    out = np.empty(len(packed) * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:n]


def unpack_int4_rows(packed: np.ndarray, N: int, K: int) -> np.ndarray:
    """Unpack [N, K//2] uint8 -> [N, K] int8."""
    return unpack_int4(np.asarray(packed, np.uint8).reshape(-1),
                       N * K).reshape(N, K)


def dequant_weight_int4(wqi: np.ndarray, sw: np.ndarray,
                        group: int = GROUP) -> np.ndarray:
    """fp64 weight dequant reference: wqi * sw (per group). Returns [N,K] fp64."""
    N, K = wqi.shape
    G = K // group
    wr = wqi.reshape(N, G, group).astype(np.float64)
    return (wr * sw.astype(np.float64)[:, :, None]).reshape(N, K)


def quantize_weight_int4_awq(w: np.ndarray, x: np.ndarray, group: int = GROUP
                             ) -> tuple[np.ndarray, np.ndarray]:
    """AWQ-style activation-aware per-group INT4 quantize (plans/int4-plan.md
    §5 fallback: "AWQ 式 per-group 搜索").

    Searches a per-(n,g) scale s = beta * max|w_group|/7 (beta in a grid) that
    minimizes the activation-weighted reconstruction error
        sum_k ( (w[n,k] - dequant(wq[n,k])) * a[k] )^2
    where a[k] = RMS over tokens of the calibration activation x[:, k]
    (protects salient K-columns; the classic AWQ per-channel weighting folded
    into the per-128-group scale, since the CD descriptor is per-(n,g)).

    w: [N,K] fp32. x: [M,K] fp32 calibration activation. Returns
    (wqi [N,K] int8 in [-7,7], sw [N,G] fp32). Same pack/unpack/lowering/qbin
    contract as quantize_weight_int4 — only the scale VALUES differ.
    """
    N, K = w.shape
    assert K % group == 0, "K must be a multiple of group"
    M = x.shape[0]
    assert x.shape[1] == K, "x and w reduction dims must match"
    G = K // group
    a = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=0))
    a = np.maximum(a, 1e-12).astype(np.float32)
    wr = w.reshape(N, G, group).astype(np.float32)
    ar = a.reshape(G, group)
    s0 = np.abs(wr).max(axis=2) / INT4_LEVEL        # symmetric anchor [N,G]
    grid = np.linspace(0.4, 1.2, 33, dtype=np.float32)
    sw = np.empty((N, G), dtype=np.float32)
    wqi = np.empty((N, K), dtype=np.int8)
    # chunk over N so the [chunk,G,group,ncand] search tensor stays small
    for n0 in range(0, N, 128):
        n1 = min(n0 + 128, N)
        wb = wr[n0:n1]                              # [c,G,128]
        s = s0[n0:n1][:, :, None] * grid[None, None, :]   # [c,G,ncand]
        wq = np.clip(np.round(wb[:, :, :, None] / s[:, :, None, :]),
                     -7, 7)                          # [c,G,128,ncand]
        deq = wq * s[:, :, None, :]
        err = (((wb[:, :, :, None] - deq) * ar[None, :, :, None]) ** 2
               ).sum(axis=2)                        # [c,G,ncand]
        bi = err.argmin(axis=2)                     # [c,G]
        c = n1 - n0
        sbest = s[np.arange(c)[:, None], np.arange(G)[None, :], bi]
        sw[n0:n1] = sbest
        wqi[n0:n1] = np.clip(np.round(wb / sbest[:, :, None]), -7, 7
                             ).reshape(c, -1).astype(np.int8)
    return wqi.reshape(N, K), sw.astype(np.float32)
