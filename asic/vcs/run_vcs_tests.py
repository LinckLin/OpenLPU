"""VCS vs Verilator byte-exact co-sim cross-validation (P10/M9 ASIC flow).

Runs the QCore co-sim instruction cases (vector subset, KV subset, plus
small matrix single-tile GEMM/GEMV) through BOTH the frozen Verilator harness
(rtl/tb) and the VCS harness (asic/vcs/simv), and asserts byte-exact agreement
of:

  * trace.bin   — per-instruction (index, cycles) records
  * total.bin   — total cycle count
  * dump.bin    — final memory bytes (concatenated requested regions)

Criterion (plans/asic-dc-plan.md §3 VcsRun): trace 逐记录一致 + 最终内存逐字节
一致.  Since the Verilator reference is already validated against the qsim
executor by rtl/tb/run_cosim.py, byte-exact VCS == Verilator proves VCS
correctness.  The executor-level result (trace_ok, max ULP) is also reported
per case (secondary, unchanged from the co-sim acceptance).

Matrix scope note: the co-sim matrix engine advances ONE MAC per clock, so the
full-size M2A linear (128x1024x2048 = 16 tiles -> ~268M cycles) is impractical
in 4-state VCS (~10k cycles/s).  Per the plan's "matrix 子集 + 单 tile GEMM"
scope, the matrix path is exercised with single-tile GEMM/GEMV cases (M<=128,
K=128, one N=128 tile), which are byte-exact-comparable and cover PF/DC and
BF16/INT8 matrix datapaths.

Run:  python3 asic/vcs/run_vcs_tests.py
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE = os.path.join(REPO, "rtl", "ref", "qsim_baseline")
TB = os.path.join(REPO, "rtl", "tb")

sys.path.insert(0, BASELINE)
sys.path.insert(0, REPO)
sys.path.insert(0, TB)

VERILATOR = os.path.join(TB, "obj_dir", "Vqcore_top")
VCS = os.path.join(HERE, "simv")

import cosim  # noqa: E402
import run_cosim  # noqa: E402
from executor import Executor  # noqa: E402

MAX_CYCLES = 2_000_000_000

# CentOS 7 compatibility namespace: simv is linked against glibc 2.17 and
# references __pthread_unwind@GLIBC_PRIVATE (removed in glibc >= 2.34), so it
# must run inside the Synopsys compat rootfs, exactly like the `vcs` compile.
SNPS_NS = "/home/public/app/synopsys/compat/bin/snps-centos7"


def run_generic(sim, program, preload, regions, max_cycles, vcs):
    """Invoke one simulator; return (trace, total, dump).  Binary formats are
    identical to cosim.run_rtl (trace: 6-byte records; total: 8-byte LE;
    dump: concatenated bytes)."""
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
            f.write(cosim.dump_requests(regions))
        if vcs:
            cmd = [
                SNPS_NS, sim,
                f"+prog={prog_p}", f"+preload={pre_p}", f"+dump_req={req_p}",
                f"+trace={tr_p}", f"+total={tot_p}", f"+dump={dump_p}",
                f"+max_cycles={max_cycles}",
                "+vcs+initreg+0", "+vcs+initmem+0",
            ]
        else:
            cmd = [sim, prog_p, pre_p, req_p, tr_p, tot_p, dump_p, str(max_cycles)]
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
        if r.returncode != 0:
            tag = "VCS" if vcs else "Verilator"
            raise RuntimeError(
                f"{tag} sim failed (rc={r.returncode}):\n"
                + r.stderr.decode(errors="replace")[-4000:])
        with open(tr_p, "rb") as f:
            trace_raw = f.read()
        with open(tot_p, "rb") as f:
            total = struct.unpack("<Q", f.read())[0]
        with open(dump_p, "rb") as f:
            dump = f.read()
    trace = [(struct.unpack("<H", trace_raw[i:i + 2])[0],
              struct.unpack("<I", trace_raw[i + 2:i + 6])[0])
             for i in range(0, len(trace_raw), 6)]
    return trace, total, dump


def make_runner(sim, vcs):
    capture = []

    def runner(program, preload, regions, max_cycles=MAX_CYCLES):
        trace, total, dump = run_generic(sim, program, preload, regions,
                                         max_cycles, vcs)
        capture.append((trace, total, dump))
        return trace, total, dump

    return runner, capture


def _byte_equal(a, b):
    tv, tot_v, dv = a
    tc, tot_c, dc = b
    return (tv == tc) and (tot_v == tot_c) and (dv == dc)


def _run_vector_kv():
    """Run the frozen vector + KV instruction suites through both simulators
    and cross-check byte-exactly.  (The full-size M2A linear is excluded: it is
    ~268M co-sim cycles and impractical in 4-state VCS; see module docstring.)"""
    v_runner, vcap = make_runner(VERILATOR, vcs=False)
    c_runner, ccap = make_runner(VCS, vcs=True)

    def collect(runner):
        results = []
        results += run_cosim.run_vector_tests()
        results += run_cosim.run_kv_tests()
        return results

    run_cosim.run_rtl = v_runner
    vres = collect(v_runner)
    run_cosim.run_rtl = c_runner
    cres = collect(c_runner)

    rows = []
    all_byte_ok = True
    for i, r in enumerate(vres):
        byte_ok = _byte_equal(vcap[i], ccap[i])
        all_byte_ok &= byte_ok
        rows.append({
            "name": r["name"],
            "byte_exact": byte_ok,
            "trace_ok": r.get("trace_ok"),
            "max_ulp": r.get("max_ulp"),
            "n_trace": len(vcap[i][0]),
            "total_cycles": vcap[i][1],
        })
    return rows, all_byte_ok


def _matrix_case(name, mode, quant, M, N, K, v_runner, c_runner, rng):
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    wq = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    exe, prog, plan = run_cosim._build_linear(mode, quant, x, wq)
    preload = cosim.dump_preload(exe)
    regions = [(0, plan.io["out_sram"], plan.io["out_bytes"])]
    etr, etot = cosim.expected_trace(prog)

    tv, tot_v, dv = v_runner(prog, preload, regions)
    tc, tot_c, dc = c_runner(prog, preload, regions)
    byte_ok = (tv == tc) and (tot_v == tot_c) and (dv == dc)

    exe.run(prog)
    ref = exe.read_bytes("sram", plan.io["out_sram"], plan.io["out_bytes"])
    a = np.frombuffer(dc, dtype=run_cosim.BF16).astype(np.float32)
    b = np.frombuffer(ref, dtype=run_cosim.BF16).astype(np.float32)
    max_ulp = (float(np.ceil(cosim.bf16_ulp_dist(a, b).max()))
               if (a.size and len(dc) == len(ref)) else float("inf"))
    trace_ok = (tc == etr) and (tot_c == etot)
    return {
        "name": name,
        "byte_exact": byte_ok,
        "trace_ok": trace_ok,
        "max_ulp": max_ulp,
        "n_trace": len(tc),
        "total_cycles": tot_c,
    }, byte_ok


def run_matrix_cases():
    """Single-tile GEMM/GEMV through the matrix engine (small enough for 4-state
    VCS, byte-exact comparable)."""
    v_runner, _ = make_runner(VERILATOR, vcs=False)
    c_runner, _ = make_runner(VCS, vcs=True)
    rng = np.random.default_rng(42)
    cases = [
        # name, mode, quant, M, N, K
        ("GEMM_1tile_BF16_M128", "PF", "BF16", 128, 128, 128),
        ("GEMM_1tile_INT8_M8",   "PF", "INT8",   8, 128, 128),
        ("GEMV_1tile_BF16_DC",   "DC", "BF16",   1, 128, 128),
    ]
    rows = []
    all_byte_ok = True
    for name, mode, quant, M, N, K in cases:
        row, ok = _matrix_case(name, mode, quant, M, N, K, v_runner, c_runner, rng)
        rows.append(row)
        all_byte_ok &= ok
    return rows, all_byte_ok


def main():
    rows, ok1 = _run_vector_kv()
    rows2, ok2 = run_matrix_cases()
    rows += rows2
    all_ok = ok1 and ok2

    for r in rows:
        mark = "PASS" if r["byte_exact"] else "FAIL"
        print(f"[{mark}] {r['name']:24s} byte_exact={r['byte_exact']} "
              f"trace_ok={r['trace_ok']} max_ulp={r['max_ulp']} "
              f"n_trace={r['n_trace']} cycles={r['total_cycles']}")

    print(f"\nVCS vs Verilator byte-exact result: "
          f"{'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    with open(os.path.join(HERE, "vcs_results.json"), "w") as f:
        json.dump({"results": rows, "all_byte_exact": all_ok}, f, indent=2)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
