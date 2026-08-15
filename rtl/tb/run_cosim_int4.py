"""QCore RTL W4A16 dedicated co-sim (Q4c).

Validates the matrix_engine.sv W4A16 data path (srcA=BF16/FP16, srcB=INT4,
acc=FP32, dequant=1) against the INT4-merged qsim baseline (qsim/executor.py
W4A16 GEMM path — live module, not the frozen rtl/ref snapshot):

  1. INT4 GEMM (PF) / GEMV (DC) vs executor at <= 1 bf16 ULP on
     normal-magnitude elements (same carve-out as the BF16 M2a co-sim), plus
     per-instruction cycle trace equality.
  2. Bit-order round-trip lock: executor pack (`_write_vector`, even -> low
     nibble, odd -> high nibble) -> RTL unpack == original matrix (via an
     identity-activation GEMM with unit scale).

Dedicated INT4 cases — NOT part of the default regression set
(plans/int4-plan.md §2/§4).

Run: python3 rtl/tb/run_cosim_int4.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(HERE, "obj_dir", "Vqcore_top")

# Live INT4-merged qsim baseline (Q4b writes qsim/, not the frozen snapshot).
sys.path.insert(0, os.path.join(REPO, "qsim"))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = np.float16

from executor import Executor, unpack_int4  # noqa: E402
import cosim  # noqa: E402
from cosim import run_rtl, dump_preload, expected_trace, bf16_ulp_dist  # noqa: E402
from compiler.isa import isa as I  # noqa: E402
from compiler.lowering import (  # noqa: E402
    WQ_HBM, SCALES_HBM, INPUT_HBM, OUTPUT_HBM,
)

GROUP = 128

# -- memory plan (single 128-column tile; mirrors compiler.lowering) ----------
AR_X_SRAM, AR_X_HBM = 0, 1
AR_OUT_SRAM, AR_OUT_HBM = 2, 3
AR_WQ_HBM = 4
AR_SCALE_SRAM, AR_SCALE_HBM = 5, 6
AR_TILE_BASE = 10
AR_TILE_B_BASE = 34
C_CA, C_CB, C_CC = 0, 1, 2
C_DMA_X, C_DMA_OUT = 4, 5
C_CD_BASE = 6


def _sram_addr(byte: int) -> int:
    assert byte % 16 == 0
    return byte // 16


def _hbm_addr(byte: int) -> int:
    return (1 << 63) | byte


def _stride_val(row: int, batch: int = 0) -> int:
    return (row << 16) | batch


def build_w4a16_program(mode: str, M: int, N: int, K: int):
    """Lower a single W4A16 linear tile x[M,K] @ W^T[N,K] to Q-ISA asm.

    srcA=BF16, srcB=INT4 (packed 2/byte, transpose_B=1 storage [N,K]),
    acc=FP32, dequant=1 (per-128-group, scale layout [N,G] row-major).
    """
    assert mode in ("PF", "DC")
    assert N == 128, "dedicated test uses a single 128-column tile"
    assert K % GROUP == 0
    if mode == "DC":
        assert M == 1
    G = K // GROUP
    esz_a, esz_out = 2, 2
    esz_b = 0.5
    ntiles = N // 128

    x_bytes = M * K * esz_a
    out_bytes = M * N * esz_out
    scale_bytes = N * G * 2

    x_sram = 0
    out_sram = (x_bytes + 15) // 16 * 16
    scale_sram = out_sram + (out_bytes + 15) // 16 * 16

    row_stride_a = K * esz_a
    row_stride_b = int(K * esz_b)
    row_stride_c = N * esz_out

    L = []
    mn = "GEMM" if mode == "PF" else "GEMV"
    L.append(f"MODE {mode}")
    L.append(f"CONFIG AR{AR_X_SRAM} = 0x{_sram_addr(x_sram):X}")
    L.append(f"CONFIG AR{AR_X_HBM} = 0x{_hbm_addr(INPUT_HBM):X}")
    L.append(f"CONFIG AR{AR_OUT_SRAM} = 0x{_sram_addr(out_sram):X}")
    L.append(f"CONFIG AR{AR_OUT_HBM} = 0x{_hbm_addr(OUTPUT_HBM):X}")
    L.append(f"CONFIG AR{AR_WQ_HBM} = 0x{_hbm_addr(WQ_HBM):X}")
    L.append(f"CONFIG AR{AR_SCALE_SRAM} = 0x{_sram_addr(scale_sram):X}")
    L.append(f"CONFIG AR{AR_SCALE_HBM} = 0x{_hbm_addr(SCALES_HBM):X}")
    for t in range(ntiles):
        L.append(f"CONFIG AR{AR_TILE_BASE + t} = "
                 f"0x{_sram_addr(out_sram + t * 128 * esz_out):X}")
        L.append(f"CONFIG AR{AR_TILE_B_BASE + t} = "
                 f"0x{_hbm_addr(WQ_HBM + int(t * 128 * K * esz_b)):X}")
    L.append(f"CONFIG C{C_CA} = 0x{_stride_val(row_stride_a):X}")
    L.append(f"CONFIG C{C_CB} = 0x{_stride_val(row_stride_b):X}")
    L.append(f"CONFIG C{C_CC} = 0x{_stride_val(row_stride_c):X}")
    for t in range(ntiles):
        cd_t = (1 << 20) | (0 << 19) | _sram_addr(scale_sram + t * 128 * G * 2)
        L.append(f"CONFIG C{C_CD_BASE + t} = 0x{cd_t:X}")
    L.append(f"CONFIG C{C_DMA_X} = 0x{K * esz_a:X}")
    L.append(f"CONFIG C{C_DMA_OUT} = 0x{N * esz_out:X}")
    L.append(f"DMA.LOAD SrcAR={AR_X_HBM} DstAR={AR_X_SRAM} RowBytes={K * esz_a} "
             f"NumRows={M} StrideC={C_DMA_X} mode=1 srcA=BF16")
    L.append(f"DMA.LOAD SrcAR={AR_SCALE_HBM} DstAR={AR_SCALE_SRAM} "
             f"RowBytes={scale_bytes} NumRows=1 StrideC=0 mode=0 srcA=BF16")
    for t in range(ntiles):
        L.append(f"{mn} ARa={AR_X_SRAM} ARb={AR_TILE_B_BASE + t} "
                 f"ARc={AR_TILE_BASE + t} M={M} N=128 K={K} batch=1 "
                 f"CA={C_CA} CB={C_CB} CC={C_CC} CD={C_CD_BASE + t} "
                 f"acc_init=1 bsrc=1 dequant=1 transpose_A=0 transpose_B=1 "
                 f"srcA=BF16 srcB=INT4 acc=FP32")
    L.append("BARRIER")
    L.append(f"DMA.STORE SrcAR={AR_OUT_SRAM} DstAR={AR_OUT_HBM} "
             f"RowBytes={N * esz_out} NumRows={M} StrideC={C_DMA_OUT} "
             f"mode=1 srcA=BF16")

    prog = I.assemble_bytes("\n".join(L))
    plan = {
        "M": M, "N": N, "K": K, "G": G,
        "out_sram": out_sram, "out_bytes": out_bytes,
        "row_stride_c": row_stride_c,
    }
    return prog, plan


def _detile_single(dump: bytes, M: int, N: int) -> np.ndarray:
    # single tile, row-major [M, N] bf16 (row_stride_c == N*2 -> contiguous)
    return np.frombuffer(dump, dtype=BF16).astype(np.float32).reshape(M, N)


def _pack_int4(exe: Executor, W: np.ndarray) -> bytes:
    """Pack an INT4 [N,K] weight matrix via the executor's authoritative
    `_write_vector` (even -> low nibble, odd -> high nibble). W is logical
    [N,K] row-major; each row packs K/2 bytes."""
    packed = exe._write_vector("hbm", WQ_HBM, I.DT_INT4, W.ravel())
    # _write_vector returns None; re-read the exact bytes written for clarity
    return exe.read_bytes("hbm", WQ_HBM, W.size // 2)


def _setup(exe: Executor, M: int, N: int, K: int, W: np.ndarray,
           x: np.ndarray, scales: np.ndarray):
    """Place weights (INT4 packed), activations (BF16) and scales (BF16)."""
    exe.write_bytes("hbm", WQ_HBM, _pack_int4(exe, W))
    exe.write_bytes("hbm", INPUT_HBM, x.astype(BF16).tobytes())
    exe.write_bytes("hbm", SCALES_HBM, scales.astype(BF16).tobytes())
    return exe


def _run_and_ref(exe: Executor, prog: bytes, plan: dict):
    M, N = plan["M"], plan["N"]
    preload = dump_preload(exe)
    etr, etot = expected_trace(prog)
    regions = [(0, plan["out_sram"], plan["out_bytes"])]
    trace, total, dump = run_rtl(prog, preload, regions)
    rtl_out = _detile_single(dump, M, N)
    exe.run(prog)
    ref_raw = exe.read_bytes("sram", plan["out_sram"], plan["out_bytes"])
    ref_out = _detile_single(ref_raw, M, N)
    return trace, total, etr, etot, rtl_out, ref_out


def run_roundtrip():
    """executor pack -> RTL unpack == original matrix (bit-order lock).

    Identity activation + unit per-group scale makes C == W^T exactly (INT4
    integers are exact in bf16, so any nibble-order drift shows up directly).
    """
    rng = np.random.default_rng(0xC4)
    M = N = K = 128
    W = rng.integers(-8, 8, size=(N, K)).astype(np.int8)
    x = np.eye(M, K, dtype=np.float32)          # identity activation
    scales = np.ones((N, 1), dtype=np.float32)  # unit scale

    exe = Executor()
    _setup(exe, M, N, K, W, x, scales)
    prog, plan = build_w4a16_program("PF", M, N, K)
    trace, total, etr, etot, rtl_out, ref_out = _run_and_ref(exe, prog, plan)

    # C[m,n] = sum_k I[m,k] * B[k,n] * 1.0 = B[m,n] = W[n,m]  (transpose_B=1)
    expected = W.T.astype(np.float32)
    n_exact = int((rtl_out == expected).sum())
    ok = (n_exact == M * N) and (trace == etr) and (total == etot)
    return {
        "name": "roundtrip_pack_unpack",
        "pass": ok, "trace_ok": (trace == etr) and (total == etot),
        "n_exact": n_exact, "n_total": M * N,
        "total_cycles": total, "expected_total": etot,
    }


def run_w4a16_case(mode: str, K: int):
    """Random W4A16 GEMM (PF) / GEMV (DC) vs the merged qsim baseline."""
    rng = np.random.default_rng(0xC4 if mode == "PF" else 0xC5)
    N = 128
    M = 128 if mode == "PF" else 1
    G = K // GROUP

    W = rng.integers(-8, 8, size=(N, K)).astype(np.int8)
    x = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    scales = (0.5 + rng.random((N, G))).astype(np.float32)

    exe = Executor()
    _setup(exe, M, N, K, W, x, scales)
    prog, plan = build_w4a16_program(mode, M, N, K)
    trace, total, etr, etot, rtl_out, ref_out = _run_and_ref(exe, prog, plan)

    trace_ok = (trace == etr) and (total == etot)
    dist = bf16_ulp_dist(rtl_out, ref_out)
    max_ulp = float(np.ceil(dist.max())) if dist.size else 0.0
    normal = np.abs(ref_out) >= 1e-3
    max_ulp_normal = float(np.ceil(dist[normal].max())) if normal.any() else 0.0
    return {
        "name": f"w4a16_{mode.lower()}_k{K}",
        "mode": mode, "K": K, "G": G,
        "trace_ok": trace_ok, "total_cycles": total,
        "expected_total": etot,
        "max_ulp_vs_executor": max_ulp,
        "max_ulp_normal": max_ulp_normal,
        "pass": trace_ok and max_ulp_normal <= 1,
    }


def main():
    results = []
    results.append(run_roundtrip())
    for K in (128, 256):
        results.append(run_w4a16_case("PF", K))
    results.append(run_w4a16_case("DC", 128))

    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        extra = ""
        if "n_exact" in r:
            extra = f" exact={r['n_exact']}/{r['n_total']}"
        elif "max_ulp_vs_executor" in r:
            extra = (f" ulp={r['max_ulp_vs_executor']:.1f} "
                     f"ulp_normal={r['max_ulp_normal']:.1f} "
                     f"cycles={r['total_cycles']}")
        print(f"[{mark}] {r['name']:20s} trace={r['trace_ok']}{extra}")

    all_pass = all(r["pass"] for r in results)
    print(f"\nQ4c W4A16 co-sim result: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    with open("/tmp/cosim_int4_results.json", "w") as f:
        json.dump({"results": results, "all_pass": all_pass}, f, indent=2)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
