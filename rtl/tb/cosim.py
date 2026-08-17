"""QCore RTL co-sim harness (M6).

Drives the Verilator-compiled qcore_top against the frozen qsim baseline
(rtl/ref/qsim_baseline/): same instruction program + same initial memory, then
compares (a) per-instruction cycle trace and (b) final memory bytes.

Cycle model mirrors command_processor.sv 1:1 (see its header comment).
"""
from __future__ import annotations

import os
import subprocess
import sys
import struct
from math import ceil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE = os.path.join(REPO, "rtl", "ref", "qsim_baseline")
SIM = os.path.join(HERE, "obj_dir", "Vqcore_top")

sys.path.insert(0, BASELINE)
sys.path.insert(0, REPO)

from executor import Executor, load_qbin_into_executor, int8_group_partials  # noqa: E402
from timing import (  # noqa: E402
    matrix_pf_cycles, matrix_dc_batch_cycles, VECTOR_LATENCY,
    hbm_write_cycles, sram_write_cycles, T_FIRST, MODE_SWITCH_CYCLES,
)
from compiler.isa import isa as I  # noqa: E402

T_FIRST = T_FIRST
MODE_SWITCH_CYCLES = MODE_SWITCH_CYCLES


# --------------------------------------------------------------------------
# Per-instruction cycle model (mirrors command_processor.sv)
# --------------------------------------------------------------------------
def inst_latency(d: dict, mode: int) -> int:
    mn = d["mnemonic"]
    op = d["opcode"]
    if mn in ("CONFIG", "BARRIER", "WAIT", "NOP"):
        return 1
    if mn == "MODE":
        return 1 if d["mode"] == mode else MODE_SWITCH_CYCLES
    if mn in ("DMA.LOAD", "DMA.PREFETCH"):
        nb = d["RowBytes"] * (d["NumRows"] if d["mode"] == 1 else 1)
        return T_FIRST + sram_write_cycles(nb)
    if mn == "DMA.STORE":
        nb = d["RowBytes"] * (d["NumRows"] if d["mode"] == 1 else 1)
        return T_FIRST + hbm_write_cycles(nb)
    if mn in ("GEMM", "GEMV", "BMM"):
        if mode == 1:
            return matrix_dc_batch_cycles(d["K"])
        return matrix_pf_cycles(d["M"], d["K"])
    if mn == "KV.APPEND":
        # B' quantized: K INT8 128 + V INT4 64 + scale 4 = 196 B write
        if d["srcA"] == 2 or d["srcB"] == 3:
            return T_FIRST + hbm_write_cycles(196)
        return T_FIRST + hbm_write_cycles(512)
    if mn == "KV.STORE_BLOCK":
        if d["srcA"] == 2 or d["srcB"] == 3:
            return T_FIRST + hbm_write_cycles(d["count"] * 196)
        return T_FIRST + hbm_write_cycles(d["count"] * 256 * 2)
    if mn == "KV.LOAD":
        w = d["count"] * 256 * (2 if d["sel"] == 2 else 1)
        return T_FIRST + sram_write_cycles(w)
    if mn == "KV.GATHER":
        w = d["count"] * 256 * (4 if d["broadcast"] else 1)
        return T_FIRST + sram_write_cycles(w)
    if op >= 0x80:  # VECTOR
        ln = d["len"]
        lat = VECTOR_LATENCY.get(mn, 16)  # RMSNORM=16 (qcore_pkg vector_latency)
        return lat * max(1, ceil(ln / 128))
    return 1


def expected_trace(program: bytes):
    """Decode the program and return (list of (index, latency), total)."""
    out = []
    total = 0
    mode = 0  # PF
    for off in range(0, len(program), 16):
        word = int.from_bytes(program[off:off + 16], "little")
        d = I.decode_inst(word)
        lat = inst_latency(d, mode)
        if d["mnemonic"] == "MODE":
            mode = d["mode"]
        out.append((off // 16, lat))
        total += lat
    return out, total


# --------------------------------------------------------------------------
# RTL runner
# --------------------------------------------------------------------------
def _rec(sel: int, addr: int, data: bytes) -> bytes:
    return struct.pack("<BQ", sel, addr) + struct.pack("<I", len(data)) + data


def dump_preload(exe: Executor) -> bytes:
    """Serialize the executor's current memory to the preload record stream."""
    parts = bytearray()
    # SRAM: non-zero contiguous runs
    buf = exe.sram.buf
    i = 0
    n = len(buf)
    while i < n:
        if buf[i] == 0:
            i += 1
            continue
        j = i
        while j < n and buf[j] != 0:
            j += 1
        parts += _rec(0, i, bytes(buf[i:j]))
        i = j
    # HBM: touched blocks
    for idx, blk in sorted(exe.hbm._blk.items()):
        parts += _rec(1, idx * 4096, bytes(blk))
    return bytes(parts)


def dump_requests(regions) -> bytes:
    """regions: list of (sel, addr, nbytes)."""
    out = bytearray()
    for sel, addr, nb in regions:
        out += struct.pack("<BQ", sel, addr) + struct.pack("<I", nb)
    return bytes(out)


def run_rtl(program: bytes, preload: bytes, regions, max_cycles=2_000_000_000):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        prog_p = os.path.join(td, "prog.bin")
        pre_p = os.path.join(td, "preload.bin")
        req_p = os.path.join(td, "dump_req.bin")
        tr_p = os.path.join(td, "trace.bin")
        tot_p = os.path.join(td, "total.bin")
        dump_p = os.path.join(td, "dump.bin")
        with open(prog_p, "wb") as f:
            f.write(program)
        with open(pre_p, "wb") as f:
            f.write(preload)
        with open(req_p, "wb") as f:
            f.write(dump_requests(regions))
        r = subprocess.run(
            [SIM, prog_p, pre_p, req_p, tr_p, tot_p, dump_p, str(max_cycles)],
            capture_output=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError("RTL sim failed:\n" + r.stderr.decode())
        with open(tr_p, "rb") as f:
            trace_raw = f.read()
        with open(tot_p, "rb") as f:
            total = struct.unpack("<Q", f.read())[0]
        with open(dump_p, "rb") as f:
            dump = f.read()
    # parse trace (6-byte records)
    trace = []
    for i in range(0, len(trace_raw), 6):
        idx = struct.unpack("<H", trace_raw[i:i + 2])[0]
        cyc = struct.unpack("<I", trace_raw[i + 2:i + 6])[0]
        trace.append((idx, cyc))
    return trace, total, dump


def mem_bytes(exe: Executor, sel: int, addr: int, n: int) -> bytes:
    return exe.read_bytes("hbm" if sel else "sram", addr, n)


def compare_trace(trace, total, expected_tr, expected_total):
    ok = True
    if total != expected_total:
        ok = False
    if [t for t in trace] != expected_tr:
        ok = False
    return ok, total, expected_total


def compare_mem(exe: Executor, regions, dump: bytes):
    """Compare dumped RTL bytes against the executor's final memory."""
    ref = bytearray()
    for sel, addr, nb in regions:
        ref += mem_bytes(exe, sel, addr, nb)
    ref = bytes(ref)
    mismatches = 0
    first = None
    for i in range(len(ref)):
        if ref[i] != dump[i]:
            mismatches += 1
            if first is None:
                first = i
    return mismatches, first, len(ref)


def bf16_ulp_dist(a, b):
    import numpy as np
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b))
    ulp = np.zeros_like(mag)
    nz = mag > 0
    ulp[nz] = np.exp2(np.floor(np.log2(mag[nz])) - 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ulp > 0, d / ulp, np.where(d > 0, np.inf, 0.0))
