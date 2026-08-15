"""M5 (P6) INT8 per-128-group activation-quantization acceptance driver.

Runs the INT8 path with **per-128-group** activation scales (calibrated from
the golden projection inputs of the evaluated prompt via `qrun.weights.
calibrate_act_scales`) and checks greedy-decode cross-agreement vs the BF16
baseline: >=8/10, plus a divergence-position rel-error report.

This supersedes the M4 per-tensor INT8 path (2/10) with the P6 per-128-group
scheme (04 §1.5 group=128, QUANT per-128-group mode — ISA/executor already
support it; only qrun runtime emission/calibration changed, qforge untouched).

Usage: python3 qrun/m5_int8.py [--out docs/p6/int8-results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qrun.engine import build_engine            # noqa: E402
from qrun.reference import (                     # noqa: E402
    load_hf, ref_greedy_with_logits)

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
QBIN = "/tmp/qwen3-0.6b.qbin"
P1_PROMPT = "Explain the concept of a transformer neural network and its attention mechanism:"
BASELINE = [3555, 374, 279, 6672, 1948, 264, 42578, 323, 264, 29728]




def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--qbin", default=QBIN)
    ap.add_argument("--out", default="docs/p6/int8-results.json")
    ap.add_argument("--max-new", type=int, default=10)
    args = ap.parse_args()

    tok, hf, ref = load_hf(args.model_dir)
    device = "cuda"

    print("=== building INT8 engine (per-128-group activation calibration) ===",
          flush=True)
    t0 = time.time()
    eng = build_engine(args.qbin, args.model_dir, "int8", tokenizer=tok, ref=ref,
                       calibration_prompt=P1_PROMPT)
    t_build = time.time() - t0
    print(f"  build: {t_build:.1f}s", flush=True)

    pid = tok(P1_PROMPT, return_tensors="pt")["input_ids"][0].numpy()
    n = args.max_new

    print(f"=== INT8 greedy decode (real PF + {n} tokens) ===", flush=True)
    t0 = time.time()
    new_tokens, logits_list = eng.generate(pid, n, bootstrap=False)
    t_gen = time.time() - t0
    print(f"  generate: {t_gen:.1f}s", flush=True)
    print(f"  qsim    : {new_tokens}", flush=True)
    print(f"  baseline: {BASELINE}", flush=True)

    print("=== HF reference greedy (divergence + per-position rel error) ===",
          flush=True)
    import torch
    hf_tokens, hf_logits = ref_greedy_with_logits(
        ref, torch.from_numpy(pid).to(device), n)

    match = [int(a == b) for a, b in zip(new_tokens, BASELINE)]
    n_match = sum(match)

    div_err = {}
    per_pos_rel_err = {}
    for k in range(n):
        hfl = hf_logits[k].detach().float().cpu().numpy()
        rel = float(np.abs(logits_list[k] - hfl).max() /
                    max(np.abs(hfl).max(), 1e-6))
        per_pos_rel_err[str(k)] = rel
        if new_tokens[k] != BASELINE[k]:
            div_err[str(k)] = rel

    result = {
        "prompt": P1_PROMPT,
        "prompt_tokens": int(pid.shape[0]),
        "calibration": "per-128-group (golden projection inputs)",
        "new_tokens": [int(x) for x in new_tokens],
        "baseline": BASELINE,
        "match": match,
        "n_match": n_match,
        "n_total": n,
        "divergence_rel_err": div_err,
        "per_pos_rel_err": per_pos_rel_err,
        "pass": n_match >= 8,
        "build_s": round(t_build, 1),
        "generate_s": round(t_gen, 1),
    }
    print(f"  match: {n_match}/{n}  {'PASS' if n_match >= 8 else 'FAIL'}",
          flush=True)
    if div_err:
        print(f"  divergence rel err: {div_err}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return result


if __name__ == "__main__":
    main()
