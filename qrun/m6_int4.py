"""M6 INT4 (W4A16) full-model decode acceptance driver (int4-plan Q4b).

Runs the INT4 path (per-128-group weight quantized W4A16, weights pre-quantized,
activations stay BF16 — no runtime QUANT/DEQUANT) and checks greedy-decode
cross-agreement vs the BF16 baseline (P1 baseline tokens): >=8/10 (here 20
tokens, so >=16/20), plus a divergence-position logits relative-error report
and the INT4 decode ceiling table (8B 317 peak / 190.5 sustained; 0.6B
~2417 short ctx / ~938 @4K / ~582 @8K sustained).

Usage: python3 qrun/m6_int4.py [--out docs/p5/int4-results.json]
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
# BF16 baseline (P1 baseline tokens, greedy decode of the above prompt).
BASELINE = [3555, 374, 279, 6672, 1948, 264, 42578, 323, 264, 29728,
            3922, 304, 4586, 30, 3555, 374, 279, 3476, 315, 279]


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--qbin", default=QBIN)
    ap.add_argument("--out", default="docs/p5/int4-results.json")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--awq", action="store_true",
                    help="AWQ-style per-(n,g) scale search (fallback)")
    args = ap.parse_args()

    tok, hf, ref = load_hf(args.model_dir)
    device = "cuda"

    mode = "W4A16-AWQ" if args.awq else "W4A16"
    print(f"=== building INT4 engine ({mode}, pre-quantized weights) ===",
          flush=True)
    t0 = time.time()
    eng = build_engine(args.qbin, args.model_dir, "int4", tokenizer=tok, ref=ref,
                       calibration_prompt=P1_PROMPT, awq=args.awq)
    t_build = time.time() - t0
    print(f"  build: {t_build:.1f}s", flush=True)

    pid = tok(P1_PROMPT, return_tensors="pt")["input_ids"][0].numpy()
    n = args.max_new

    print(f"=== INT4 greedy decode (real PF + {n} tokens) ===", flush=True)
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
        rel = float(np.abs(logits_list[k] - hfl).max() / np.abs(hfl).max())
        per_pos_rel_err[str(k)] = rel
        if new_tokens[k] != BASELINE[k]:
            div_err[str(k)] = rel

    # INT4 decode ceiling (spec 04 §2.5, 8B; 0.6B roofline incl. KV re-read,
    # same correction note as INT8).  Sustained = HBM 720 GB/s bound.
    ceiling = {
        "8B_peak_tokens_s": 317,
        "8B_sustained_tokens_s": 190.5,
        "0.6B_short_ctx_sustained": 2417,
        "0.6B_4k_sustained": 938,
        "0.6B_8k_sustained": 582,
        "notes": (
            "W4A16 decode is HBM-bound on weight streaming (INT4 weight read "
            "3.78 GB/token for 8B; 0.6B ~0.298 GB/token dense + lm_head). "
            "KV re-read uses the same per-token window formula as the INT8 "
            "correction note (114,688 B/token x ctx); short-ctx excludes the "
            "full-window re-read. W4A16 array throughput is BF16-class "
            "(8.19 TMAC/s), so decode stays weight-bandwidth-bound, not "
            "compute-bound."
        ),
    }

    result = {
        "prompt_tokens": int(pid.shape[0]),
        "prompt": P1_PROMPT,
        "dtype": "int4_awq" if args.awq else "int4",
        "new_tokens": [int(t) for t in new_tokens],
        "baseline": BASELINE,
        "match": match,
        "n_match": n_match,
        "n_total": n,
        "cross_agreement_pass": n_match >= int(0.8 * n),
        "divergence_rel_err": div_err,
        "per_position_rel_err": per_pos_rel_err,
        "build_s": t_build,
        "generate_s": t_gen,
        "ceiling": ceiling,
    }
    print(f"  match: {n_match}/{n}  "
          f"{'PASS' if result['cross_agreement_pass'] else 'FAIL'}", flush=True)
    if div_err:
        print(f"  divergence rel err: {div_err}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return result


if __name__ == "__main__":
    main()
