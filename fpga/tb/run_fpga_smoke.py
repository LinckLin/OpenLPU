"""P9 FPGA integration smoke (M6 co-sim criterion).

Drives the Verilator-compiled qcore_fpga_top (host_if + ddr_if + clock_reset
wrapping the unmodified Command Processor) against the frozen qsim baseline
(rtl/ref/qsim_baseline/): the same instruction program + same initial memory,
then compares (a) the per-instruction cycle trace (exact equality) and (b) the
final memory bytes (bf16 <= 1 ULP, |y| >= 1e-3 M4 normal-magnitude criterion).

The whole run goes through the host_if register interface (config / cmd-queue
load / qbin load / logits readback) — no testbench backdoor — proving the P9
board-independent control plane end to end.

Coverage (one representative instruction sequence per engine/memory space):
  1. VADD            — vector engine, SRAM operand marshalling
  2. RMSNORM         — vector engine (reduce + scale), SRAM
  3. KV.APPEND+LOAD  — DMA + KV address generator, HBM/DDR write+read round trip
  4. GEMM BF16       — matrix engine, SRAM (M=4, N=128, K=128)
"""
from __future__ import annotations

import os
import subprocess
import sys
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE = os.path.join(REPO, "rtl", "ref", "qsim_baseline")
SIM = os.environ.get("FPGA_SIM", os.path.join(HERE, "obj_dir", "Vqcore_fpga_top"))

sys.path.insert(0, BASELINE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "rtl", "tb"))

from executor import Executor  # noqa: E402
from cosim import (  # noqa: E402
    dump_preload, dump_requests, expected_trace, bf16_ulp_dist, mem_bytes,
)
from compiler.isa import isa as I  # noqa: E402

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = np.float16


def _cfg(prog: bytearray, reg: int, cls: int, val: int):
    prog += I.encode_inst("CONFIG", REG=reg, reg_class=cls,
                          IMM64=int(val)).to_bytes(16, "little")


# --------------------------------------------------------------------------
# FPGA runner (same file protocol as cosim.run_rtl, but through host_if)
# --------------------------------------------------------------------------
def run_fpga(program: bytes, preload: bytes, regions, max_cycles=2_000_000_000):
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
            raise RuntimeError("FPGA sim failed:\n" + r.stderr.decode())
        with open(tr_p, "rb") as f:
            trace_raw = f.read()
        with open(tot_p, "rb") as f:
            total = struct.unpack("<Q", f.read())[0]
        with open(dump_p, "rb") as f:
            dump = f.read()
    trace = []
    for i in range(0, len(trace_raw), 6):
        idx = struct.unpack("<H", trace_raw[i:i + 2])[0]
        cyc = struct.unpack("<I", trace_raw[i + 2:i + 6])[0]
        trace.append((idx, cyc))
    return trace, total, dump


def _compare(exe: Executor, prog: bytes, regions, name: str):
    """regions: list of (sel, addr, nb) — bf16 ULP compare (nb even) else byte."""
    preload = dump_preload(exe)
    etr, etot = expected_trace(prog)
    trace, total, dump = run_fpga(prog, preload, regions)
    exe.run(prog)
    trace_ok = (trace == etr) and (total == etot)
    max_ulp = 0.0
    byte_ok = True
    off = 0
    for sel, addr, nb in regions:
        ref = mem_bytes(exe, sel, addr, nb)
        got = dump[off:off + nb]
        off += nb
        if nb % 2 == 0 and len(ref) == len(got):
            a = np.frombuffer(got, dtype=BF16).astype(np.float32)
            b = np.frombuffer(ref, dtype=BF16).astype(np.float32)
            if a.size:
                dist = bf16_ulp_dist(a, b)
                max_ulp = max(max_ulp, float(dist.max()))
                normal = np.abs(b) >= 1e-3
                max_ulp = max(max_ulp,
                              float(np.ceil(dist[normal].max())) if normal.any() else 0.0)
        else:
            byte_ok = byte_ok and (got == ref)
    return {
        "name": name, "trace_ok": trace_ok, "total": total, "expected_total": etot,
        "max_ulp": max_ulp, "byte_ok": byte_ok,
        "pass": trace_ok and byte_ok and max_ulp <= 1,
    }


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------
def case_vadd(rng):
    n = 128
    a = (rng.standard_normal(n) * 2 + 1).astype(np.float32).astype(BF16)
    b = (rng.standard_normal(n) * 2 + 1).astype(np.float32).astype(BF16)
    exe = Executor()
    exe.write_bytes("sram", 0x1000 * 16, a.tobytes())
    exe.write_bytes("sram", 0x1010 * 16, b.tobytes())
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x1010); _cfg(prog, 2, 1, 0x1020)
    prog += I.encode_inst("VADD", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARd=2, len=n, CV=0).to_bytes(16, "little")
    return exe, bytes(prog), [(0, 0x1020 * 16, n * 2)]


def case_rmsnorm(rng):
    n = 128
    a = (rng.standard_normal(n) * 2).astype(np.float32).astype(BF16)
    g = (rng.standard_normal(n) + 1).astype(np.float32).astype(BF16)
    exe = Executor()
    exe.write_bytes("sram", 0x1000 * 16, a.tobytes())
    exe.write_bytes("sram", 0x1010 * 16, g.tobytes())
    eps_bits = np.array([1e-6], dtype=np.float32).view(np.uint32)[0]
    exe.C[6] = eps_bits
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x1010); _cfg(prog, 2, 1, 0x1020)
    _cfg(prog, 6, 0, eps_bits)
    prog += I.encode_inst("RMSNORM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARd=2, len=n, CV=6, imm=0).to_bytes(16, "little")
    return exe, bytes(prog), [(0, 0x1020 * 16, n * 2)]


def case_kv(rng):
    exe = Executor()
    exe.C[exe.C_SLAB_SHIFT] = 21
    exe.AR[exe.AR_KV_BASE] = (1 << 63) | 0
    exe.C[exe.C_KV_POS] = 0
    k = (rng.standard_normal(128)).astype(np.float32).astype(BF16)
    v = (rng.standard_normal(128)).astype(np.float32).astype(BF16)
    exe.write_bytes("sram", 0x2000 * 16, k.tobytes())
    exe.write_bytes("sram", 0x2010 * 16, v.tobytes())
    prog = bytearray()
    for i, vv in enumerate((0x2000, 0x2010, 0x2020, 0x2030)):
        _cfg(prog, i, 1, vv)
    _cfg(prog, 31, 0, 21)          # C[31] = slab shift
    _cfg(prog, 63, 1, 1 << 63)     # AR[63] = KV base (HBM, addr 0)
    prog += I.encode_inst("KV.APPEND", srcK=0, srcV=1, layer=3, head=5).to_bytes(16, "little")
    prog += I.encode_inst("KV.LOAD", dstK=2, dstV=3, layer=3, head=5, sel=2,
                          pos_start=0, count=1).to_bytes(16, "little")
    regions = [(0, 0x2020 * 16, 256), (0, 0x2030 * 16, 256)]
    return exe, bytes(prog), regions


def case_gemm(rng):
    M, N, K = 4, 128, 128
    A = (rng.standard_normal(M * K) * 2).astype(np.float32).astype(BF16)
    B = (rng.standard_normal(K * N) * 2).astype(np.float32).astype(BF16)
    exe = Executor()
    # SRAM byte addresses: A @ 0x1000, B @ 0x10000, C @ 0x30000
    exe.write_bytes("sram", 0x1000, A.tobytes())
    exe.write_bytes("sram", 0x10000, B.tobytes())
    row_a = K * 2   # 256 B/row
    row_b = N * 2   # 256 B/row
    row_c = N * 2
    prog = bytearray()
    _cfg(prog, 0, 1, 0x100)     # AR0 = A word addr (byte 0x1000)
    _cfg(prog, 1, 1, 0x1000)    # AR1 = B word addr (byte 0x10000)
    _cfg(prog, 2, 1, 0x3000)    # AR2 = C word addr (byte 0x30000)
    _cfg(prog, 10, 0, row_a << 16)   # C10 = row_stride_a
    _cfg(prog, 11, 0, row_b << 16)   # C11 = row_stride_b
    _cfg(prog, 12, 0, row_c << 16)   # C12 = row_stride_c
    prog += I.encode_inst("GEMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARc=2, M=M, N=N, K=K, batch=0,
                          CA=10, CB=11, CC=12, acc_init=1,
                          transpose_A=0, transpose_B=0).to_bytes(16, "little")
    regions = [(0, 0x30000, M * N * 2)]
    return exe, bytes(prog), regions


def main():
    rng = np.random.default_rng(0)
    cases = [
        ("VADD", case_vadd),
        ("RMSNORM", case_rmsnorm),
        ("KV_APPEND_LOAD", case_kv),
        ("GEMM_BF16", case_gemm),
    ]
    results = []
    all_pass = True
    for name, fn in cases:
        exe, prog, regions = fn(rng)
        r = _compare(exe, prog, regions, name)
        results.append(r)
        mark = "PASS" if r["pass"] else "FAIL"
        all_pass = all_pass and r["pass"]
        print(f"[{mark}] {name:16s} trace={r['trace_ok']} "
              f"max_ulp={r['max_ulp']:.1f} cycles={r['total']} "
              f"(expected {r['expected_total']})")
    print(f"\nP9 FPGA smoke: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
