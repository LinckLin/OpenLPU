"""Quality baseline driver — perf-research §4 step 1 (golden long-context PPL).

Establishes the *golden* quality baseline for Qwen3-0.6B (BF16, HF native
full-attention) that the windowed-KV optimisations (O1 StreamingLLM, O2 INT8 KV,
O3 H2O) will be gated against:

  1. Long-context PPL: PG19 spans of length 4K / 8K tokens (>= ``--n-samples``
     per length).  Each span is scored with HF native
     full causal attention over the WHOLE span — no sliding window, no stride —
     so every predicted token attends to the full 4K/8K context.  This is what
     makes the baseline discriminative: the window candidates W=1024/2048 are
     strictly shorter than the eval length, so windowed KV must drop context and
     any degradation shows up as ΔPPL.
  2. HellaSwag zero-shot sanity: a deterministic ``--hellaswag-n`` subset,
     scored as zero-shot multiple choice (argmax of the summed conditional
     log-likelihood of each completion given the context).  Recorded for
     completeness; not a pass/fail gate.

Outputs (single JSON): per-token NLL **and** the exact token ids for every span
(so a later windowed-KV run can re-score the identical spans and compute ΔPPL
token-for-token), plus the HellaSwag per-example predictions, plus the full
environment metadata for reproducibility.

Thresholds encoded here as documented constants (see the report):
  * O1 hard gate:  ΔPPL <= 2% relative (windowed vs full attention).
  * HellaSwag:     Δacc recorded as sanity only (no pass/fail).

Usage:
  python3 qrun/quality_baseline.py \
      --dataset pg19 --seq-lens 4096,8192 --n-samples 10 --hellaswag-n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"),
)

DEFAULT_OUT = "docs/perf-research/quality-baseline/quality_baseline.json"
DEFAULT_SEQ_LENS = "4096,8192"
DEFAULT_N_SAMPLES = 10
DEFAULT_HELLASWAG_N = 100
DEFAULT_SEED = 0

# Documented gates (perf-research §4 / roadmap.md O1).
O1_DELTA_PPL_GATE = 0.02  # <= 2% relative ΔPPL (windowed vs full) is the hard gate.


def load_model(model_dir: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # exact full attention — the golden path
    )
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def full_attention_nll(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token NLL under HF native full causal attention over the whole span.

    ``input_ids``: [1, L].  Returns [L-1] fp32 NLL values where element ``i``
    is ``-log P(token[i+1] | token[0..i])`` — i.e. every prediction attends to
    the full L-token context (no window, no stride).
    """
    logits = model(input_ids).logits  # [1, L, V] (bf16)
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # fp32 softmax (golden)
    targets = input_ids[:, 1:]  # [1, L-1]
    nll = -log_probs[:, :-1, :].gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return nll[0]  # [L-1] fp32, on device


def load_texts(dataset: str):
    """Return (dataset_display_name, list_of_(source_id, text))."""
    from datasets import load_dataset

    if dataset == "pg19":
        ds = load_dataset("emozilla/pg19")
        val = ds["validation"]
        texts = [(i, row["text"]) for i, row in enumerate(val)]
        return f"pg19 (emozilla/pg19, validation, {len(texts)} books)", texts
    raise ValueError(f"unknown dataset {dataset!r} (only 'pg19' supported)")


def collect_spans(tok, texts, seq_len: int, n_samples: int):
    """Deterministic interior spans of exactly ``seq_len`` tokens.

    Per-book: take the interior span ``tokens[seq_len : 2*seq_len]`` (skips the
    opening title/TOC, stays fully inside one coherent book).  Only books long
    enough (>= 2*seq_len tokens) qualify, so every span is a genuine
    long-range dependency chunk.
    """
    spans = []
    for src_id, text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) >= 2 * seq_len:
            spans.append((src_id, ids[seq_len : 2 * seq_len]))
            if len(spans) >= n_samples:
                break
    return spans


def evaluate_ppl(model, tok, texts, seq_lens, n_samples, device):
    samples = {}
    for L in seq_lens:
        spans = collect_spans(tok, texts, L, n_samples)
        if len(spans) < n_samples:
            raise RuntimeError(
                f"only {len(spans)} spans of length {L} available (need {n_samples}); "
                "raise --n-samples or pick a corpus with longer documents")
        out = []
        for k, (src_id, ids) in enumerate(spans):
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            nll = full_attention_nll(model, inp)  # [L-1]
            mean_nll = nll.mean().item()
            out.append({
                "sample_id": f"L{L}_s{k:02d}_book{src_id:04d}",
                "source_id": int(src_id),
                "offset_tokens": L,
                "seq_len": L,
                "n_target_tokens": int(nll.numel()),
                "token_ids": ids,
                "per_token_nll": [round(float(x), 6) for x in nll.tolist()],
                "mean_nll": float(mean_nll),
                "ppl": float(np.exp(mean_nll)),
            })
        samples[str(L)] = out
    return samples


@torch.no_grad()
def hellaswag_score(model, tok, ctx: str, endings, device: str):
    """Conditional log-likelihood of each completion given the context.

    Tokenises ``ctx + sep + ending`` together (BPE prefix stability guarantees
    the context prefix tokenises identically), then scores only the ending
    tokens conditioned on the full prefix.  Returns three parallel lists over
    the completions:

      * ``raw``    — summed log-likelihood (the classic ``acc`` score);
      * ``norm``   — length-normalised mean log-likelihood per ending token
                     (the standard HellaSwag ``acc_norm`` score; used for the
                     greedy argmax, which removes the shorter-ending bias of
                     the raw sum);
      * ``n_toks`` — number of scored ending tokens.
    """
    ctx_ids = tok.encode(ctx, add_special_tokens=False)
    sep = "" if ctx.endswith(" ") else " "
    full = [tok.encode(ctx + sep + e, add_special_tokens=False) for e in endings]
    max_len = max(len(x) for x in full)
    pad = tok.pad_token_id
    batch = torch.full((len(full), max_len), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(full), max_len), dtype=torch.long, device=device)
    for j, x in enumerate(full):
        batch[j, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
        mask[j, : len(x)] = 1

    logits = model(input_ids=batch, attention_mask=mask).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)

    raw, norm, n_toks = [], [], []
    for j, x in enumerate(full):
        L = len(x)
        n_ctx = len(ctx_ids)
        # Ending tokens sit at positions [n_ctx, L); token at pos p is predicted
        # by logits at p-1.  Guard the degenerate empty-ending case.
        if L <= n_ctx:
            raw.append(0.0)
            norm.append(0.0)
            n_toks.append(0)
            continue
        pos = torch.arange(n_ctx - 1, L - 1, device=device)  # logits indices
        tgt = batch[j, n_ctx:L]  # ending token ids
        s = log_probs[j, pos, tgt].sum().item()
        k = L - n_ctx
        raw.append(s)
        norm.append(s / k)
        n_toks.append(k)
    return raw, norm, n_toks


def evaluate_hellaswag(model, tok, n_examples, seed, device):
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag")
    val = ds["validation"]
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(val), size=n_examples, replace=False).tolist())

    per_example = []
    n_correct = 0
    n_correct_raw = 0
    for i in idxs:
        row = val[int(i)]
        ctx = row["ctx"]
        endings = list(row["endings"])
        label = int(row["label"])
        raw, norm, n_toks = hellaswag_score(model, tok, ctx, endings, device)
        pred = int(np.argmax(norm))          # length-normalised greedy argmax
        pred_raw = int(np.argmax(raw))        # raw-sum argmax (secondary)
        correct = pred == label
        correct_raw = pred_raw == label
        n_correct += int(correct)
        n_correct_raw += int(correct_raw)
        per_example.append({
            "idx": int(i),
            "label": label,
            "pred": pred,
            "pred_raw": pred_raw,
            "correct": bool(correct),
            "correct_raw": bool(correct_raw),
            "scores_raw": [round(float(s), 6) for s in raw],
            "scores_norm": [round(float(s), 6) for s in norm],
            "n_toks": n_toks,
        })

    return {
        "method": "zero-shot multiple choice: argmax of length-normalised "
                  "conditional log-likelihood (acc_norm, primary) and raw-sum "
                  "log-likelihood (acc, secondary); no chat template",
        "n_examples": n_examples,
        "seed": seed,
        "indices": idxs,
        "accuracy_norm": n_correct / n_examples,
        "accuracy_raw": n_correct_raw / n_examples,
        "n_correct_norm": n_correct,
        "n_correct_raw": n_correct_raw,
        "per_example": per_example,
    }


def main() -> dict:
    import transformers

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--dataset", default="pg19", choices=["pg19"])
    ap.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS,
                    help="comma-separated eval lengths in tokens (each > W candidates)")
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                    help="spans per seq length (>=10 required)")
    ap.add_argument("--hellaswag-n", type=int, default=DEFAULT_HELLASWAG_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-hellaswag", action="store_true")
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    assert args.n_samples >= 10, "--n-samples must be >= 10 (report contract)"
    for L in seq_lens:
        assert L >= 4096, f"seq len {L} < 4096 is not discriminative vs W=2048"

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA not available; pass --device cpu (slow)")

    print(f"=== load model ({args.model_dir}) ===", flush=True)
    t0 = time.time()
    tok, model = load_model(args.model_dir, args.device)
    print(f"  {type(model).__name__} {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"params, loaded in {time.time()-t0:.1f}s", flush=True)

    result = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "Qwen/Qwen3-0.6B",
            "model_dir": args.model_dir,
            "dtype": "bfloat16",
            "attention": "eager (HF native full attention)",
            "device": args.device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "seed": args.seed,
            "ppl_strategy": "contiguous interior span [L, 2L) scored with full "
                             "causal attention over all L tokens (fp32 softmax over bf16 logits)",
            "o1_delta_ppl_gate": O1_DELTA_PPL_GATE,
        },
    }

    if not args.skip_ppl:
        print(f"=== PPL ({args.dataset}, seq lens {seq_lens}, "
              f"{args.n_samples} spans each) ===", flush=True)
        name, texts = load_texts(args.dataset)
        result["meta"]["ppl_dataset"] = name
        t0 = time.time()
        result["ppl"] = evaluate_ppl(model, tok, texts, seq_lens, args.n_samples, args.device)
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        for L in seq_lens:
            means = [s["mean_nll"] for s in result["ppl"][str(L)]]
            ppls = [s["ppl"] for s in result["ppl"][str(L)]]
            print(f"  L={L}: mean NLL {np.mean(means):.4f} "
                  f"| PPL mean {np.mean(ppls):.3f} "
                  f"| per-sample NLL [{min(means):.4f}, {max(means):.4f}]", flush=True)

    if not args.skip_hellaswag:
        print(f"=== HellaSwag ({args.hellaswag_n} examples, seed {args.seed}) ===",
              flush=True)
        t0 = time.time()
        result["hellaswag"] = evaluate_hellaswag(
            model, tok, args.hellaswag_n, args.seed, args.device)
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        print(f"  acc_norm {result['hellaswag']['accuracy_norm']:.4f} "
              f"({result['hellaswag']['n_correct_norm']}/{args.hellaswag_n}) | "
              f"acc_raw {result['hellaswag']['accuracy_raw']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return result


if __name__ == "__main__":
    main()
