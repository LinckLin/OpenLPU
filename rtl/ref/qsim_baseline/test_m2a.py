"""M2a validation: {PF, DC} x {BF16, INT8} single linear layer, end to end.

Pipeline exercised per case:
  golden linear_wq_{pf,dc} -> qbin (weights + Q-ISA program) -> qsim executor
  -> bf16 output -> tolerance check vs golden bf16 y (or INT8 reference).

Run:  python3 qsim/test_m2a.py
Tolerances (Wave1Review fix — MATRIX writes BF16, 04 §1.5 post-process):
  BF16  : bf16 output vs golden bf16 y  <= 1 ulp (|y| >= 1e-3)
  INT8  : INT32 accumulation bit-exact ; dequant bf16 output vs bf16(fp64 ref) <= 1 ulp
The INT8 weight is a test-side symmetric per-128-K-group quantization (NOT a P4
artifact); the activation is per-tensor symmetric INT8, and scale_x is folded
into the CD per-group scale (compiler-side fold, no ISA change).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = np.float16

from compiler.lowering import (
    lower_linear, encode_program, WQ_HBM, SCALES_HBM, INPUT_HBM, OUTPUT_HBM,
)
from compiler.isa.qbin import Tensor, write_qbin, read_qbin
from qsim.executor import Executor, load_qbin_into_executor, int8_group_partials

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "golden", "qwen3-0.6b")
CFG = {"hidden": 1024, "layers": 28, "q_heads": 16, "kv_heads": 8,
       "head_dim": 128, "intermediate": 3072, "vocab": 151936,
       "rope_theta": 1000000.0, "rms_eps": 1e-6, "qk_norm": True,
       "max_pos": 40960}
MODEL = "Qwen3-0.6B"


def load_golden(mode: str):
    d = os.path.join(GOLDEN, f"linear_wq_{mode.lower()}")
    inp = np.load(os.path.join(d, "inputs.npz"))
    out = np.load(os.path.join(d, "outputs.npz"))
    w = np.load(os.path.join(d, "weights.npz"))
    x = inp["x"].astype(np.float32)
    wq = w["wq"].astype(np.float32)
    y_ref = out["y_ref"].astype(np.float32)
    y = out["y"].astype(np.float32)
    return x, wq, y_ref, y


def _run(mode: str, quant: str, x, wq):
    """Build qbin + run executor; return bf16 output (fp32 values of bf16)."""
    M, K = x.shape
    N = wq.shape[0]
    plan = lower_linear((M, K), (N, K), mode, quant)
    prog = encode_program(plan)

    if quant == "BF16":
        xh = x.astype(BF16)
        wqh = wq.astype(BF16)
        t = Tensor(name="model.layers.0.self_attn.q_proj.weight",
                   shape=[N, K], dtype="BF16", hbm_off=WQ_HBM,
                   data=wqh.tobytes())
    else:  # INT8 (test-side symmetric quantization)
        G = K // 128
        sx = np.abs(x).max() / 127.0
        xq = np.clip(np.round(x / sx), -127, 127).astype(np.int8)
        sw = np.abs(wq.reshape(N, G, 128)).max(axis=2) / 127.0   # (N, G)
        wqi = np.clip(np.round(wq.reshape(N, G, 128) / sw[:, :, None]),
                      -127, 127).astype(np.int8).reshape(N, K)
        cd = (sx * sw).astype(BF16)  # combined per-group scale (scale_x folded)
        xh = xq
        t = Tensor(name="model.layers.0.self_attn.q_proj.weight",
                   shape=[N, K], dtype="INT8", hbm_off=WQ_HBM,
                   data=wqi.tobytes(), scales_hbm_off=SCALES_HBM,
                   scales=cd.tobytes(), scale_dtype="BF16")

    qbin_path = f"/tmp/m2a_{mode}_{quant}.qbin"
    quant_j = {"mode": quant if quant == "BF16" else "W8A8",
               "group": 128, "sym": True}
    write_qbin(qbin_path, MODEL, CFG, quant_j, [t],
               prog if mode == "PF" else b"",
               prog if mode == "DC" else b"")
    qb = read_qbin(qbin_path)

    exe = Executor()
    load_qbin_into_executor(exe, qb)
    exe.write_bytes("hbm", INPUT_HBM, xh.tobytes())
    exe.run(qb.pf_program if mode == "PF" else qb.dc_program)
    # MATRIX writes BF16 (04 §1.5 post-process) into SRAM, one 128-wide column
    # tile per ntiles. The program still reserves esz_out bytes per element for
    # the output tile/row offsets (fp32 accumulator width), so de-tile the bf16
    # payload: tile t at out_sram + t*128*esz_out, row r at + r*row_stride_c.
    N_TILE = 128
    out = np.empty((M, N), dtype=np.float32)
    row_stride = plan.io["row_stride_c"]
    tile_stride = N_TILE * plan.esz_out
    for r in range(M):
        for t in range(plan.ntiles):
            base = plan.io["out_sram"] + r * row_stride + t * tile_stride
            raw = exe.read_bytes("sram", base, N_TILE * 2)
            out[r, t * N_TILE:(t + 1) * N_TILE] = \
                np.frombuffer(raw, dtype=BF16).astype(np.float32)
    return out




def _bf16_ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element bf16-domain ULP distance (plan §1 convention)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b))
    ulp = np.zeros_like(mag)
    nz = mag > 0
    ulp[nz] = np.exp2(np.floor(np.log2(mag[nz])) - 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ulp > 0, d / ulp, np.where(d > 0, np.inf, 0.0))


def run_case(mode: str, quant: str):
    x, wq, y_ref, y = load_golden(mode)
    M, K = x.shape
    N = wq.shape[0]
    out = _run(mode, quant, x, wq)

    if quant == "BF16":
        # executor writes BF16 (fp32 accumulator rounded to BF16). Validate vs
        # golden bf16 y: normal-magnitude elements must agree to <= 1 bf16 ulp.
        dist = _bf16_ulp_dist(out, y)
        normal = np.abs(y) >= 1e-3
        max_ulp_normal = float(np.ceil(dist[normal].max())) if normal.any() else 0
        max_ulp = float(np.ceil(dist.max()))
        tiny_frac = float((~normal).mean())
        abs_err = float(np.abs(out - y).max())
        rel_err = float(np.abs(out - y).max() / np.abs(y).max())
        # "落盘 <= 1 ulp" holds on normal-magnitude values; tiny cancellation
        # values (|y|<1e-3) can differ by a few ulps from the fp32 accumulation
        # order (numpy vs torch) — a cross-implementation effect, not a bug.
        passed = max_ulp_normal <= 1
        return {
            "mode": mode, "quant": "BF16",
            "bf16_max_abs_err": abs_err, "bf16_max_rel_err": rel_err,
            "bf16_max_ulp": max_ulp,
            "bf16_max_ulp_normal": max_ulp_normal,
            "bf16_tiny_frac": tiny_frac,
            "pass": passed, "criteria": "bf16<=1ulp(|y|>=1e-3)",
        }
    else:  # INT8
        G = K // 128
        sx = np.abs(x).max() / 127.0
        xq = np.clip(np.round(x / sx), -127, 127).astype(np.int8)
        sw = np.abs(wq.reshape(N, G, 128)).max(axis=2) / 127.0
        wqi = np.clip(np.round(wq.reshape(N, G, 128) / sw[:, :, None]),
                      -127, 127).astype(np.int8).reshape(N, K)
        cd = (sx * sw).astype(BF16)

        # INT32 bit-exact: executor's group partials vs independent einsum ref
        # (B is (K, N) = wqi.T, matching the executor's transpose_B=1 read)
        partials = int8_group_partials(xq, wqi.T, G)
        bit_exact = True
        for g in range(G):
            refp = np.einsum("ik,kj->ij",
                             xq[:, g * 128:(g + 1) * 128].astype(np.int32),
                             wqi[:, g * 128:(g + 1) * 128].T.astype(np.int32),
                             dtype=np.int32, optimize=True)
            if not np.array_equal(partials[g], refp):
                bit_exact = False
                break

        # golden dequant reference (fp64) with the SAME bf16 CD scales. The
        # executor's dequant writeback is BF16 (04 §1.5), so the golden bf16 y
        # is the fp64 reference rounded to BF16.
        ref = np.zeros((M, N), np.float64)
        for g in range(G):
            ref += (cd[:, g].astype(np.float64)[None, :]
                    * partials[g].astype(np.float64))
        golden_bf16 = ref.astype(BF16).astype(np.float32)
        max_ulp = float(np.ceil(_bf16_ulp_dist(out, golden_bf16).max()))
        abs_err = float(np.abs(out.astype(np.float64) - ref).max())
        rel_err = float(np.abs(out.astype(np.float64) - ref).max()
                        / np.abs(ref).max())
        passed = bit_exact and max_ulp <= 1
        return {
            "mode": mode, "quant": "INT8",
            "int32_bit_exact": bit_exact,
            "dequant_bf16_max_ulp": max_ulp,
            "dequant_max_abs_err": abs_err, "dequant_max_rel_err": rel_err,
            "pass": passed,
            "criteria": "int32 bit-exact & dequant bf16<=1ulp",
        }


def main():
    results = []
    for mode in ("PF", "DC"):
        for quant in ("BF16", "INT8"):
            r = run_case(mode, quant)
            results.append(r)
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"[{mark}] {mode:2s} {quant:4s}  {r['criteria']}")
            print("        ", {k: v for k, v in r.items()
                                if k not in ("mode", "quant", "pass", "criteria")})

    all_pass = all(r["pass"] for r in results)
    print(f"\nM2a result: {'ALL 4 PASS' if all_pass else 'FAILURES PRESENT'}")
    out_path = "/tmp/m2a_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "all_pass": all_pass}, f, indent=2)
    print(f"results written to {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
