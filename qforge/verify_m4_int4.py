"""M4 INT4 (W4A16) verification (plans/int4-plan.md §3 Q4a).

Dual-track per-class report (6 classes x PF/DC):
  (a) implementation correctness: same-scale fp64 reference dequant < 1e-6
      (fp32 W4A16 dequant vs fp64 reference using the SAME BF16 CD scales);
  (b) quantization error vs fp32 golden (rel, informational — expected
      8-15%), with an INT8 same-metric comparison column (from m3-results.json).

Plus the pack -> executor unpack round-trip lock test (packing authority =
qsim/executor.py _read_vector nibble order: even index -> low nibble).

Produces an INT4 full-model qbin (tensors dtype=INT4) + per-class W4A16 qbins.
Writes only qforge/ and docs/p4/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = np.float16

from compiler.isa import isa as I
from qsim.executor import Executor, load_qbin_into_executor
from qforge import config as C, graph, quant, build
from qforge.safetensors import SafeTensors
from qforge.verify_m3 import (load_golden_input, load_golden_yref, load_weight,
                              ST_PATH)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3_RESULTS = os.path.join(REPO, "docs", "p4", "m3-results.json")
DEFAULT_QBIN = "/tmp/qwen3-0.6b-m4-int4.qbin"

GROUP = 128


# ---------------------------------------------------------------------------
# W4A16 dequant reference (fp32 impl vs fp64 ref, SAME BF16 CD scales)
# ---------------------------------------------------------------------------
def _w4a16_dequant(A: np.ndarray, wqi: np.ndarray, cd: np.ndarray,
                   dtype) -> np.ndarray:
    """Per-128-K-group W4A16 dequant: C[m,n] = sum_g scale[n,g] *
    sum_k A[m,k] * wqi[n,k] (fp-domain accumulation, 02 §6 / 04 §1.2)."""
    M, K = A.shape
    N = wqi.shape[0]
    G = K // GROUP
    C = np.zeros((M, N), dtype=dtype)
    for g in range(G):
        part = (A[:, g * GROUP:(g + 1) * GROUP].astype(dtype)
                @ wqi[:, g * GROUP:(g + 1) * GROUP].T.astype(dtype))
        C += cd[:, g].astype(dtype)[None, :] * part
    return C


def pack_to_executor_lock_test(wqi: np.ndarray, packed: np.ndarray) -> bool:
    """qforge pack -> executor _read_vector unpack == original INT4 matrix."""
    exe = Executor()
    addr = 0x0010_0000
    exe.write_bytes("hbm", addr, packed.tobytes())
    flat = exe._read_vector("hbm", addr, I.DT_INT4, int(wqi.size))
    return bool(np.array_equal(flat, wqi.reshape(-1).astype(np.int32)))


# ---------------------------------------------------------------------------
# per-class INT4 case
# ---------------------------------------------------------------------------
def run_int4_case(kind: str, mode: str, st: SafeTensors) -> dict:
    w = load_weight(kind, st)
    x = load_golden_input(kind, mode)
    y_ref = load_golden_yref(kind, mode)
    M, K = x.shape
    N = w.shape[0]
    assert tuple(w.shape) == (N, K), f"{kind} {mode}: w {w.shape} != {(N, K)}"

    wqi, sw = quant.quantize_weight_int4(w)
    cd = quant.combine_scale(sw, 1.0)             # BF16 per-group scale (sx=1)
    packed = quant.pack_int4(wqi)

    # (0) pack -> executor unpack lock test (authoritative nibble order)
    lock_ok = pack_to_executor_lock_test(wqi, packed)

    # activation dtype = BF16 (W4A16); same A in both impl (fp32) and ref (fp64)
    A_bf16 = x.astype(BF16).astype(np.float32)

    # (a) implementation correctness: fp32 impl vs fp64 ref (SAME scales).
    # "dequant rel" is measured against the output's PEAK magnitude (robust to
    # near-zero elements; matches the M3 report's "logits |y| 峰值 ≈ 30 →
    # rel ≈ 1.1e-7 ≈ fp32 eps" note). Gate: abs < 1e-6 OR rel < 1e-6.
    C_fp32 = _w4a16_dequant(A_bf16, wqi, cd, np.float32)
    C_fp64 = _w4a16_dequant(A_bf16, wqi, cd, np.float64)
    dequant_abs = float(np.abs(C_fp32.astype(np.float64) - C_fp64).max())
    dequant_rel = dequant_abs / max(float(np.abs(C_fp64).max()), 1e-30)
    dequant_pass = dequant_abs < 1e-6 or dequant_rel < 1e-6

    # (b) quantization error vs fp32 golden (informational, report as-is)
    out = C_fp32.astype(BF16).astype(np.float32)   # executor BF16 writeback
    qerr_abs = float(np.abs(out - y_ref).max())
    qerr_rel = float(np.abs(out - y_ref).max() / np.abs(y_ref).max())

    # build the single-projection W4A16 qbin (load check + artifact)
    qb, cd_r, wqi_r, sw_r, in_hbm, out_hbm = build.build_projection_qbin_int4(
        f"/tmp/qforge_{kind}_{mode}_int4.qbin", kind, w, x, mode)
    assert qb.tensors[0].dtype == "INT4", f"{kind} {mode}: dtype != INT4"
    assert len(qb.tensors[0].data) == N * (K // 2), f"{kind} {mode}: packed bytes"
    qbin_load_ok = np.array_equal(
        np.frombuffer(qb.tensors[0].data, dtype=np.uint8),
        np.asarray(packed, np.uint8).reshape(-1))
    assert np.array_equal(cd_r.astype(np.float32), cd.astype(np.float32)), \
        f"{kind} {mode}: cd mismatch"
    assert np.array_equal(wqi_r, wqi), f"{kind} {mode}: wqi mismatch"

    return {
        "kind": kind, "mode": mode, "N": N, "K": K, "M": M,
        "pack_unpack_lock_ok": bool(lock_ok),
        "dequant_max_abs_err": dequant_abs,
        "dequant_max_rel_err": dequant_rel,
        "dequant_pass": bool(dequant_pass),
        "quant_err_vs_golden_abs": qerr_abs,
        "quant_err_vs_golden_rel": qerr_rel,
        "qbin_dtype": qb.tensors[0].dtype,
        "qbin_load_ok": bool(qbin_load_ok),
        "criteria": "lock bit-exact & fp64-ref dequant<1e-6 (abs or rel)",
    }


# ---------------------------------------------------------------------------
# full-model INT4 qbin load check
# ---------------------------------------------------------------------------
def full_model_int4_checks(qbin_path: str) -> dict:
    from compiler.isa.qbin import read_qbin
    qb = read_qbin(qbin_path)
    pf_len = qb.header["pf_len"]
    dc_len = qb.header["dc_len"]
    exact_pf = qb.pf_program[:pf_len]
    exact_dc = qb.dc_program[:dc_len]
    res = {
        "magic": qb.magic.decode(), "version": qb.version, "flags": qb.flags,
        "header_size": qb.header_size, "model": qb.header["model"],
        "quant": qb.header["quant"],
        "n_tensors": len(qb.tensors),
        "pf_insts": len(exact_pf) // 16, "dc_insts": len(exact_dc) // 16,
        "pf_len_bytes": len(exact_pf), "dc_len_bytes": len(exact_dc),
    }
    assert qb.magic == b"NLPU", "bad magic"
    assert qb.version == 1, "bad version"
    assert qb.flags & 0x3 == 3, "dtype flag != INT4(3)"
    assert qb.flags & 0x4, "dual-mode flag not set"
    assert len(exact_pf) % 16 == 0 and len(exact_dc) % 16 == 0
    assert all(t.dtype == "INT4" for t in qb.tensors), "tensors dtype != INT4"

    # program decode (all instructions valid + engine tags valid)
    for name, prog in (("pf", exact_pf), ("dc", exact_dc)):
        for off in range(0, len(prog), 16):
            d = I.decode_inst(int.from_bytes(prog[off:off + 16], "little"))
            assert d["engine_tag_valid"], f"{name} inst {off // 16} engine mismatch"
    res["programs_decode_ok"] = True

    # weight/scale round-trip via executor HBM (loadable: write then read back)
    exe = Executor()
    load_qbin_into_executor(exe, qb)
    total_w = total_s = 0
    for t in qb.tensors:
        back = exe.read_bytes("hbm", t.hbm_off, len(t.data))
        assert back == t.data, f"round-trip mismatch: {t.name}"
        if t.scales is not None:
            back_s = exe.read_bytes("hbm", t.scales_hbm_off, len(t.scales))
            assert back_s == t.scales, f"scale round-trip mismatch: {t.name}"
        total_w += len(t.data)
        total_s += len(t.scales or b"")
    res["round_trip_ok"] = True
    res["weight_bytes"] = total_w
    res["scale_bytes"] = total_s

    # W4A16 GEMM dtype smoke: the PF/DC programs must emit srcB=INT4 GEMM/GEMV
    # with srcA=BF16 acc=FP32 and no QUANT before the linear layers.
    def _count_w4a16(prog: bytes) -> dict:
        w4 = 0
        quant = 0
        for off in range(0, len(prog), 16):
            d = I.decode_inst(int.from_bytes(prog[off:off + 16], "little"))
            if d["mnemonic"] in ("GEMM", "GEMV") and d["srcB"] == I.DT_INT4 \
                    and d["srcA"] == I.DT_BF16 and d["acc"] == I.ACC_FP32 \
                    and d["dequant"] == 1:
                w4 += 1
            if d["mnemonic"] == "QUANT":
                quant += 1
        return {"w4a16_gemm": w4, "quant_ops": quant}
    res["pf_w4a16"] = _count_w4a16(exact_pf)
    res["dc_w4a16"] = _count_w4a16(exact_dc)
    assert res["pf_w4a16"]["w4a16_gemm"] > 0, "PF: no W4A16 GEMM emitted"
    assert res["dc_w4a16"]["w4a16_gemm"] > 0, "DC: no W4A16 GEMV emitted"
    assert res["pf_w4a16"]["quant_ops"] == 0, "PF: unexpected QUANT op"
    assert res["dc_w4a16"]["quant_ops"] == 0, "DC: unexpected QUANT op"
    return res


def _load_int8_comparison() -> dict:
    if not os.path.exists(M3_RESULTS):
        return {}
    with open(M3_RESULTS) as f:
        m3 = json.load(f)
    return {(c["kind"], c["mode"]): c["quant_err_vs_golden_rel"]
            for c in m3.get("cases", [])}


def _logits_rmse(out: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((out.astype(np.float64)
                                  - y_ref.astype(np.float64)) ** 2)))


def run_awq_case(kind: str, mode: str, st: SafeTensors) -> dict:
    """AWQ fallback (plans/int4-plan.md §5): per-group scale search vs the
    symmetric baseline, measured on the logits RMSE (the AWQ objective /
    token-agreement proxy) and on the max-abs quant error (same metric as the
    symmetric report). Calibration is oracle (the eval-mode golden input)."""
    w = load_weight(kind, st)
    x = load_golden_input(kind, mode)
    y_ref = load_golden_yref(kind, mode)
    A = x.astype(BF16).astype(np.float32)

    wqi_s, sw_s = quant.quantize_weight_int4(w)
    out_s = _w4a16_dequant(A, wqi_s, quant.combine_scale(sw_s, 1.0),
                           np.float32).astype(BF16).astype(np.float32)
    wqi_a, sw_a = quant.quantize_weight_int4_awq(w, x)
    out_a = _w4a16_dequant(A, wqi_a, quant.combine_scale(sw_a, 1.0),
                           np.float32).astype(BF16).astype(np.float32)

    return {
        "kind": kind, "mode": mode,
        "sym_logits_rmse": _logits_rmse(out_s, y_ref),
        "awq_logits_rmse": _logits_rmse(out_a, y_ref),
        "sym_quant_err_rel": float(np.abs(out_s - y_ref).max()
                                   / np.abs(y_ref).max()),
        "awq_quant_err_rel": float(np.abs(out_a - y_ref).max()
                                   / np.abs(y_ref).max()),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qbin", default=DEFAULT_QBIN)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "p4",
                                                  "m4-int4-results.json"))
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--awq", action="store_true",
                    help="also run the AWQ per-group scale-search fallback")
    args = ap.parse_args()

    t0 = time.time()
    st = SafeTensors.open(ST_PATH)
    if not args.skip_build:
        build.build_model_qbin(args.qbin, st, C.QUANT_INT4,
                               C.ACTIVATION_SCALE_DEFAULT)
        build_sec = time.time() - t0
    else:
        build_sec = 0.0

    fm = full_model_int4_checks(args.qbin)
    fm["build_sec"] = round(build_sec, 3)

    int8_cmp = _load_int8_comparison()
    cases = []
    for kind in graph.CLASS_NAMES:
        for mode in ("PF", "DC"):
            r = run_int4_case(kind, mode, st)
            r["int8_quant_err_rel"] = int8_cmp.get((kind, mode))
            cases.append(r)
            mark = "PASS" if (r["pack_unpack_lock_ok"] and r["dequant_pass"]) \
                else "FAIL"
            print(f"[{mark}] {kind:7s} {mode:2s}  lock={r['pack_unpack_lock_ok']} "
                  f"dequant={r['dequant_max_abs_err']:.2e} "
                  f"quant_err_rel={r['quant_err_vs_golden_rel']:.4f} "
                  f"(int8={r['int8_quant_err_rel']})")

    all_pass = all(c["pack_unpack_lock_ok"] and c["dequant_pass"] for c in cases)
    awq_cases = []
    if args.awq:
        print("\n(c) AWQ fallback (per-group scale search, oracle calibration):")
        for kind in graph.CLASS_NAMES:
            for mode in ("PF", "DC"):
                a = run_awq_case(kind, mode, st)
                awq_cases.append(a)
                print(f"  {kind:7s} {mode:2s}  rmse {a['sym_logits_rmse']:.4f}->"
                      f"{a['awq_logits_rmse']:.4f} "
                      f"(ratio={a['awq_logits_rmse']/a['sym_logits_rmse']:.3f})  "
                      f"maxrel {a['sym_quant_err_rel']:.4f}->"
                      f"{a['awq_quant_err_rel']:.4f}")

    print("\n(a) full-model INT4:", json.dumps(fm, indent=2))
    print(f"\n(b) 6 classes x PF/DC INT4: "
          f"{'ALL PASS' if all_pass else 'FAILURES'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"full_model": fm, "cases": cases, "all_pass": all_pass,
                   "awq_cases": awq_cases}, f, indent=2)
    print(f"results -> {args.out}")
    return 0 if (all_pass and fm["round_trip_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
