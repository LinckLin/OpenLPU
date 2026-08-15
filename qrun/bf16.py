"""BF16 encode/decode helpers for qrun (host side).

The executor stores BF16 as little-endian 2-byte values via `ml_dtypes.bfloat16`.
qrun uses the same dtype so host-written BF16 bytes are bit-identical to what the
executor reads/writes. All helpers are dtype-agnostic for fp32 <-> bf16:
  * fp32 -> bf16 rounds-to-nearest-even (same as ml_dtypes / numpy astype)
  * bf16 -> fp32 is exact (bf16 is a subset of fp32)
"""
from __future__ import annotations

import numpy as np

try:
    import ml_dtypes
    _BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    _BF16 = np.float16

BF16_NP = np.dtype(_BF16)


def fp32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """fp32 (or any float) ndarray -> little-endian BF16 bytes (flat)."""
    return np.asarray(arr, dtype=BF16_NP).tobytes()

def bf16_bytes_to_fp32(data: bytes, count: int | None = None) -> np.ndarray:
    """little-endian BF16 bytes -> fp32 ndarray (exact)."""
    arr = np.frombuffer(data, dtype=BF16_NP) if count is None else \
        np.frombuffer(data, dtype=BF16_NP, count=count)
    return arr.astype(np.float32)


def read_bf16_vec(data: bytes) -> np.ndarray:
    """BF16 bytes -> fp32 (whole buffer)."""
    return bf16_bytes_to_fp32(data)


def ulp_bf16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ULP distance between two fp32 arrays measured on the BF16 lattice.

    Both inputs are treated as BF16 values carried in fp32. The distance is
    counted in BF16 ULP steps (1 ULP = the spacing of the BF16 grid around the
    larger-magnitude value's binade). Values that differ only by bf16 rounding
    of the same fp32 real are 0..1 ULP apart.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    # bf16 has 8 mantissa bits -> unit in last place for a binade = 2^(e-7)
    def ulp_bits(x: np.ndarray) -> np.ndarray:
        xb = x.astype(BF16_NP).view(np.uint16)
        return xb.astype(np.int64)

    # Measure on the integer BF16 bit lattice (monotonic except the sign wrap).
    ua = ulp_bits(a)
    ub = ulp_bits(b)
    # Map sign-magnitude BF16 bits to a signed linear order.
    def lin(bits: np.ndarray) -> np.ndarray:
        sign = bits >> 15
        mag = bits & 0x7FFF
        return np.where(sign == 0, mag.astype(np.int64), (-mag).astype(np.int64))

    return np.abs(lin(ua) - lin(ub)).astype(np.float64)


def bf16_roundtrip(arr: np.ndarray) -> np.ndarray:
    """Round an fp32 array to BF16 and back to fp32 (lossy)."""
    return np.asarray(arr, dtype=BF16_NP).astype(np.float32)
