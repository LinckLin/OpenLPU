"""Minimal safetensors reader (header parse + lazy byte slice + bf16->fp32).

Self-contained: reads the 8-byte header length + JSON header, then slices the
data section directly (no torch / framework conversion). BF16 tensors are
returned as float32 via ml_dtypes.bfloat16 (numpy 2.x native).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

try:
    import ml_dtypes
    _BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    _BF16 = np.float16

_BF16_NP = np.dtype(_BF16)


@dataclass
class SafeTensors:
    path: str
    header: dict
    _data: bytes
    _base: int  # data section start offset

    @staticmethod
    def open(path: str) -> "SafeTensors":
        with open(path, "rb") as f:
            head = f.read(8)
        (hlen,) = struct.unpack("<Q", head)
        with open(path, "rb") as f:
            f.seek(8)
            header = json.loads(f.read(hlen))
            f.seek(8 + hlen)
            data = f.read()
        return SafeTensors(path, header, data, 8 + hlen)

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def shape(self, name: str) -> list[int]:
        return list(self.header[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.header[name]["dtype"]

    def _slice_bytes(self, name: str) -> bytes:
        start, end = self.header[name]["data_offsets"]
        return self._data[start:end]

    def get_float32(self, name: str) -> np.ndarray:
        """Load a BF16/F32 tensor as float32 (row-major, PyTorch layout)."""
        e = self.header[name]
        dt = e["dtype"]
        raw = self._slice_bytes(name)
        n = 1
        for s in e["shape"]:
            n *= s
        if dt == "BF16":
            arr = np.frombuffer(raw, dtype=_BF16_NP, count=n)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32, count=n)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16, count=n)
        else:
            raise ValueError(f"unsupported safetensors dtype {dt!r}")
        return arr.astype(np.float32).reshape(e["shape"])
