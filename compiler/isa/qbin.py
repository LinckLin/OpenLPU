"""QCore .qbin container writer/reader.

Authoritative reference: docs/spec-src/00-container.md §2 (frozen P0).

Layout (little-endian, sections 64B aligned):
  0        magic       4B  "NLPU"
  4        version     u32  1
  8        flags       u32  bits[1:0] default dtype (0=BF16,2=INT8,3=INT4), bit2 dual-mode
  12       header_size u32  header region bytes (incl. u32 length prefix)
  16       header      var  u32 json_len + JSON
  ...      .weights    var  quantized weights at file offset == tensor hbm_off
  ...      .pf_program var  prefill Q-ISA program (raw 128-bit inst bytes)
  ...      .dc_program var  decode Q-ISA program
  ...      .end        8B   "ENDQ" + u32 total file length (validation)
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

MAGIC = b"NLPU"
SENTINEL = b"ENDQ"
VERSION = 1
ALIGN = 64

# flags bits
FLAG_DTYPE_MASK = 0x3
FLAG_DUAL_MODE = 0x4

DTYPE_FLAG = {"BF16": 0, "FP16": 1, "INT8": 2, "INT4": 3}


@dataclass
class Tensor:
    name: str
    shape: list[int]
    dtype: str                 # "BF16" | "INT8" | "INT4"
    hbm_off: int
    data: bytes
    scales_hbm_off: int | None = None
    scales: bytes | None = None   # per-128-group scale bytes (BF16/FP16)
    scale_dtype: str = "BF16"

    @property
    def bytes(self) -> int:
        return len(self.data)


def _align(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def build_header(model: str, cfg: dict, quant: dict, tensors: list[Tensor],
                 pf_entry: int, dc_entry: int, pf_len: int, dc_len: int) -> dict:
    tlist = []
    for t in tensors:
        e = {
            "name": t.name, "shape": t.shape, "dtype": t.dtype,
            "hbm_off": t.hbm_off, "bytes": t.bytes,
        }
        if t.scales is not None:
            e["scales_hbm_off"] = t.scales_hbm_off
            e["scale_bytes"] = len(t.scales)
            e["scale_dtype"] = t.scale_dtype
        tlist.append(e)
    tlist.sort(key=lambda e: e["hbm_off"])  # tensors table by ascending HBM addr
    return {
        "model": model,
        "cfg": cfg,
        "quant": quant,
        "tensors": tlist,
        "pf_entry": pf_entry,
        "dc_entry": dc_entry,
        "pf_len": pf_len,
        "dc_len": dc_len,
    }


def write_qbin(path: str, model: str, cfg: dict, quant: dict,
               tensors: list[Tensor], pf_program: bytes, dc_program: bytes,
               flags: int | None = None) -> dict:
    """Write a qbin file. Returns the final header dict (with resolved offsets)."""
    if flags is None:
        dtype_code = DTYPE_FLAG.get(quant.get("mode"), 0)
        flags = (dtype_code & FLAG_DTYPE_MASK) | FLAG_DUAL_MODE

    hbm_off_min = min(t.hbm_off for t in tensors)

    # placeholder header to size the region
    header = build_header(model, cfg, quant, tensors, 0, 0,
                          len(pf_program), len(dc_program))
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_region = struct.pack("<I", len(header_json)) + header_json
    header_size = len(header_region)

    weights_base = hbm_off_min
    min_weights_base = _align(16 + header_size, ALIGN)
    if weights_base < min_weights_base:
        raise ValueError(
            f"tensor hbm_off {weights_base} < header end {min_weights_base}; "
            f"raise hbm_off (header region is {header_size} bytes)")

    weights_end = weights_base
    for t in sorted(tensors, key=lambda x: x.hbm_off):
        end = t.hbm_off + len(t.data)
        if t.scales is not None:
            end = max(end, t.scales_hbm_off + len(t.scales))
        weights_end = max(weights_end, end)
    weights_end = _align(weights_end, ALIGN)

    pf_entry = weights_end
    dc_entry = _align(pf_entry + len(pf_program), ALIGN)
    end_off = _align(dc_entry + len(dc_program), ALIGN)

    header = build_header(model, cfg, quant, tensors, pf_entry, dc_entry,
                          len(pf_program), len(dc_program))
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_region = struct.pack("<I", len(header_json)) + header_json
    header_size = len(header_region)

    # Pre-size a zero buffer then place each section at its absolute offset.
    # (bytearray slice assignment past the end appends instead of extending —
    # so build the full buffer first.)
    total = end_off + 8
    out = bytearray(total)
    out[0:4] = MAGIC
    out[4:8] = struct.pack("<I", VERSION)
    out[8:12] = struct.pack("<I", flags)
    out[12:16] = struct.pack("<I", header_size)
    out[16:16 + len(header_region)] = header_region
    for t in sorted(tensors, key=lambda x: x.hbm_off):
        out[t.hbm_off:t.hbm_off + len(t.data)] = t.data
        if t.scales is not None:
            out[t.scales_hbm_off:t.scales_hbm_off + len(t.scales)] = t.scales
    out[pf_entry:pf_entry + len(pf_program)] = pf_program
    out[dc_entry:dc_entry + len(dc_program)] = dc_program
    out[end_off:end_off + 4] = SENTINEL
    out[end_off + 4:end_off + 8] = struct.pack("<I", total)
    with open(path, "wb") as f:
        f.write(out)
    return header


@dataclass
class Qbin:
    path: str
    magic: bytes
    version: int
    flags: int
    header_size: int
    header: dict
    tensors: list[Tensor]
    pf_program: bytes
    dc_program: bytes
    data: bytes = field(repr=False)


def read_qbin(path: str) -> Qbin:
    with open(path, "rb") as f:
        data = f.read()

    magic = data[0:4]
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
    version = struct.unpack_from("<I", data, 4)[0]
    flags = struct.unpack_from("<I", data, 8)[0]
    header_size = struct.unpack_from("<I", data, 12)[0]
    if header_size < 4:
        raise ValueError("bad header_size")
    json_len = struct.unpack_from("<I", data, 16)[0]
    header_json = data[20:20 + json_len].decode("utf-8")
    header = json.loads(header_json)

    tensors = []
    for e in header.get("tensors", []):
        t = Tensor(
            name=e["name"], shape=e["shape"], dtype=e["dtype"],
            hbm_off=e["hbm_off"], data=data[e["hbm_off"]:e["hbm_off"] + e["bytes"]],
        )
        if "scales_hbm_off" in e:
            t.scales_hbm_off = e["scales_hbm_off"]
            t.scales = data[e["scales_hbm_off"]:e["scales_hbm_off"] + e["scale_bytes"]]
            t.scale_dtype = e.get("scale_dtype", "BF16")
        tensors.append(t)

    pf_entry = header["pf_entry"]
    dc_entry = header["dc_entry"]
    sentinel_off = len(data) - 8
    if data[sentinel_off:sentinel_off + 4] != SENTINEL:
        raise ValueError("missing ENDQ sentinel")
    stored_len = struct.unpack_from("<I", data, sentinel_off + 4)[0]
    if stored_len != len(data):
        raise ValueError(
            f"length check failed: stored {stored_len}, actual {len(data)}")

    pf_program = data[pf_entry:dc_entry]
    dc_program = data[dc_entry:sentinel_off]
    return Qbin(path, magic, version, flags, header_size, header, tensors,
                pf_program, dc_program, data)
