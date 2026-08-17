"""A-Gate: host-side n-gram / Lookup draft — greedy acceptance-rate (α) measurement.

Pre-RTL gate for the speculative-decoding direction (DECISION v2 §3.1 /
speculative.md §5).  Zero-training host drafts are the only D13-compatible path
to break the weight wall (B1), and the only open question is the greedy
acceptance rate α of a 0.6B target — no literature anchor exists below 7B.

Two host-side draft sources are measured, both lossless under greedy
verification (the verified output is byte-identical to plain greedy decoding,
so α is the *per-token* probability that a proposed draft token equals the
target's greedy argmax):

  1. ``ngram``  — a static 4-gram dictionary (context = last 3 tokens ->
                 most-frequent continuation), trained on a few MB of general
                 text (PG19 *train*, disjoint from every eval corpus).
  2. ``pld``    — Prompt Lookup Decoding: dynamic dictionary built from the
                 prompt + already-accepted tokens (the "Lookup 草案"); the
                 most recent prior occurrence of the current 3-gram gives the
                 next token.  ~100% coverage by construction.

Measurement is offline and exact: greedy tokens are produced once per corpus
(HF ``generate``, do_sample=False → argmax, BF16, eager attention), then the
draft sources are replayed over the identical greedy continuation and compared
token-for-token.  A γ=4 chain-verify simulation (candidate generation →
token-by-token greedy accept-until-first-mismatch) is also replayed to estimate
the simulated wall-clock speedup against single-token decode, using the frozen
qsim timing constants (T_step = 1,154,087 cyc/token; T_verify(n) =
1,154,087 + (n-1)*4,748 cyc/step — DECISION §3 / speculative.md §3).

Gate (DECISION §3.1, applied to the measured α):
  α >= 0.40            -> 立项 (project);
  α in [0.25, 0.40)    -> 降级 gen-2;
  α <  0.25            -> 关闭 (close the direction).

Outputs: a single JSON (``--out``) with per-corpus α tables, the chain-verify
wall-clock estimate, and the gate verdict.

Usage:
  python3 qrun/spec_alpha.py \
      --hellaswag-n 100 --general-spans 6 --train-mb 6 --gamma 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"),
)

DEFAULT_OUT = "docs/perf-research/decision/alpha-measure.json"
DEFAULT_HELLASWAG_N = 100
DEFAULT_GENERAL_SPANS = 6
DEFAULT_GENERAL_PROMPT = 256   # prompt tokens per general-text span
DEFAULT_GENERAL_GEN = 128      # greedy tokens generated per span
DEFAULT_HELLASWAG_GEN = 32     # greedy tokens generated per HellaSwag context
DEFAULT_TRAIN_MB = 6
DEFAULT_SEED = 0
DEFAULT_GAMMA = 4

# Frozen qsim timing constants (DECISION v2 §3 / speculative.md §3; qsim/timing_p6.py).
N_LAYERS = 28
PER_LAYER_CYC = 33_500        # HBM weight + KV window read per layer
LM_HEAD_CYC = 216_087         # lm_head 155.6 MB weight stream tail
T_STEP = N_LAYERS * PER_LAYER_CYC + LM_HEAD_CYC          # 1,154,087 cyc/token
LM_HEAD_MAC_STEP = 4_748      # incremental lm_head MAC cyc per extra candidate
TOK_PER_S = 866.0             # audit §3.1 decode ceiling (W=2048 windowed BF16)

# Gate (DECISION §3.1).
ALPHA_START = 0.40
ALPHA_GEN2 = 0.25


def load_model(model_dir: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",   # exact full attention — same path as quality-baseline
    )
    model.to(device).eval()
    return tok, model


def load_train_texts(max_bytes: int):
    """PG19 *train* split (disjoint from eval corpora), first ``max_bytes`` of text."""
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19")
    train = ds["train"]
    buf, nbytes = [], 0
    for row in train:
        text = row["text"]
        buf.append(text)
        nbytes += len(text.encode("utf-8", "ignore"))
        if nbytes >= max_bytes:
            break
    return buf


def build_ngram(tok, texts):
    """Static 4-gram dictionary: (t_{-3},t_{-2},t_{-1}) -> most-frequent next token.

    Returns (dict, n_4grams, n_contexts, n_tokens)."""
    counts: Counter = Counter()
    n_tokens = 0
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        n_tokens += len(ids)
        for i in range(3, len(ids)):
            counts[(ids[i - 3], ids[i - 2], ids[i - 1], ids[i])] += 1
    best: dict = {}
    for (a, b, c, d), cnt in counts.items():
        key = (a, b, c)
        if key not in best or cnt > best[key][1]:
            best[key] = (d, cnt)
    return {k: v[0] for k, v in best.items()}, len(counts), len(best), n_tokens


def load_hellaswag_prompts(n_examples: int, seed: int):
    """HellaSwag validation subset — identical index selection as quality-baseline."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag")
    val = ds["validation"]
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(val), size=n_examples, replace=False).tolist())
    return [(int(i), val[int(i)]["ctx"]) for i in idxs]


def load_general_prompts(tok, n_spans: int, prompt_len: int):
    """PG19 *validation* interior spans — same deterministic strategy as
    quality-baseline ``collect_spans`` (interior [L, 2L) tokens), but shorter."""
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19")
    val = ds["validation"]
    spans = []
    for src_id, row in enumerate(val):
        ids = tok.encode(row["text"], add_special_tokens=False)
        if len(ids) >= 2 * prompt_len:
            spans.append((src_id, ids[prompt_len:2 * prompt_len]))
            if len(spans) >= n_spans:
                break
    return spans


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new: int, device: str):
    """Greedy (argmax) continuation; returns the generated token ids (list[int])."""
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(inp, do_sample=False, max_new_tokens=max_new,
                         pad_token_id=model.config.eos_token_id,
                         attention_mask=torch.ones_like(inp))
    return out[0, len(prompt_ids):].tolist()


def measure_alpha(P, G, ngram):
    """Offline per-token acceptance over a greedy continuation.

    ``P`` = prompt token ids, ``G`` = greedy-generated token ids.  For every
    generated position p (target = G[p]), propose a draft from each source using
    the *true* prefix (P + G[:i]); count proposal coverage and accept matches.

    Returns (stats, n_positions).
    """
    full = list(P) + list(G)
    Lp = len(P)
    Lf = len(full)

    # PLD dynamic index over the prompt (positions where a 3-gram's next token is known).
    last_follow: dict = {}
    for q in range(3, Lp):
        key = (full[q - 3], full[q - 2], full[q - 1])
        last_follow[key] = q

    stats = {
        "ngram": {"proposed": 0, "accepted": 0, "miss": 0},
        "pld":   {"proposed": 0, "accepted": 0, "miss": 0},
    }
    for p in range(Lp, Lf):
        g = full[p]
        key = (full[p - 3], full[p - 2], full[p - 1]) if p >= 3 else None

        if key is not None and key in ngram:
            stats["ngram"]["proposed"] += 1
            stats["ngram"]["accepted"] += int(ngram[key] == g)
        else:
            stats["ngram"]["miss"] += 1

        if key is not None and key in last_follow and last_follow[key] < p:
            stats["pld"]["proposed"] += 1
            stats["pld"]["accepted"] += int(full[last_follow[key]] == g)
        else:
            stats["pld"]["miss"] += 1

        if key is not None:
            last_follow[key] = p

    return stats, Lf - Lp


def _verify_cost(n: int) -> int:
    """Frozen qsim cost of a chain-verify step with ``n`` candidates."""
    return T_STEP + (n - 1) * LM_HEAD_MAC_STEP


def chain_sim(P, G, gamma: int, ngram, source: str):
    """Chain-verify (γ) replay: candidate generation -> token-by-token greedy
    accept-until-first-mismatch.  Returns per-step accounting + wall-clock.

    Uses the frozen qsim timing model: baseline = T_STEP per produced token;
    speculative = Σ T_verify(n_step) over verification steps (n = 1 + draft
    chain length actually proposed).
    """
    prefix = list(P)
    nG = len(G)
    gi = 0
    steps = 0
    proposed = 0
    accepted = 0
    produced = 0
    cyc_spec = 0

    # PLD dynamic index over the accepted prefix (grows as tokens are committed).
    last_follow: dict = {}
    for q in range(3, len(prefix)):
        key = (prefix[q - 3], prefix[q - 2], prefix[q - 1])
        last_follow[key] = q

    while gi < nG:
        chain = []
        ctx = tuple(prefix[-3:]) if len(prefix) >= 3 else None
        for _ in range(gamma):
            if ctx is None:
                break
            if source == "ngram":
                d = ngram.get(ctx)
            else:  # pld — reference = accepted prefix only
                d = prefix[last_follow[ctx]] if ctx in last_follow else None
            if d is None:
                break
            chain.append(d)
            ctx = ctx[1:] + (d,)
        # clamp to the available greedy continuation
        chain = chain[: max(0, nG - gi - 1)]
        n = 1 + len(chain)

        acc = 0
        for j, d in enumerate(chain):
            if d == G[gi + j]:
                acc += 1
            else:
                break
        produced_step = 1 + acc
        for j in range(produced_step):
            tok_id = G[gi + j]
            prefix.append(tok_id)
            # register the 3-gram whose following token is the just-appended token
            if len(prefix) >= 4:
                key = (prefix[-4], prefix[-3], prefix[-2])
                last_follow[key] = len(prefix) - 1
        gi += produced_step
        steps += 1
        proposed += len(chain)
        accepted += acc
        produced += produced_step
        cyc_spec += _verify_cost(n)

    cyc_base = produced * T_STEP
    return {
        "steps": steps,
        "proposed": proposed,
        "accepted": accepted,
        "produced": produced,
        "alpha": (accepted / proposed) if proposed else None,
        "cyc_base": cyc_base,
        "cyc_spec": cyc_spec,
        "speedup": cyc_base / cyc_spec if cyc_spec else None,
    }


def summarize(stats, n_pos, source: str):
    """Aggregate per-source stats into (coverage, alpha_covered, alpha_effective)."""
    s = stats[source]
    proposed = s["proposed"]
    coverage = proposed / n_pos if n_pos else 0.0
    alpha_covered = s["accepted"] / proposed if proposed else None
    alpha_effective = s["accepted"] / n_pos if n_pos else None
    return {
        "n_positions": n_pos,
        "proposed": proposed,
        "accepted": s["accepted"],
        "miss": s["miss"],
        "coverage": coverage,
        "alpha_covered": alpha_covered,
        "alpha_effective": alpha_effective,
    }


def gate_verdict(alpha: float) -> str:
    if alpha >= ALPHA_START:
        return "立项 (alpha >= 0.40)"
    if alpha >= ALPHA_GEN2:
        return "降级 gen-2 (alpha in [0.25, 0.40))"
    return "关闭 (alpha < 0.25)"


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hellaswag-n", type=int, default=DEFAULT_HELLASWAG_N)
    ap.add_argument("--hellaswag-gen", type=int, default=DEFAULT_HELLASWAG_GEN)
    ap.add_argument("--general-spans", type=int, default=DEFAULT_GENERAL_SPANS)
    ap.add_argument("--general-prompt", type=int, default=DEFAULT_GENERAL_PROMPT)
    ap.add_argument("--general-gen", type=int, default=DEFAULT_GENERAL_GEN)
    ap.add_argument("--train-mb", type=float, default=DEFAULT_TRAIN_MB)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--gamma", type=int, default=DEFAULT_GAMMA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    import transformers

    t0 = time.time()
    print(f"[spec_alpha] loading {args.model_dir} (BF16, eager) ...", flush=True)
    tok, model = load_model(args.model_dir, args.device)
    print(f"  transformers={transformers.__version__} torch={torch.__version__}", flush=True)

    print(f"[spec_alpha] building static 4-gram dict from PG19 train (~{args.train_mb} MB) ...",
          flush=True)
    t1 = time.time()
    train_texts = load_train_texts(int(args.train_mb * 1e6))
    ngram, n_4grams, n_contexts, n_train_tok = build_ngram(tok, train_texts)
    print(f"  train: {len(train_texts)} books, {n_train_tok} tokens, "
          f"{n_4grams} 4-grams, {n_contexts} unique contexts  ({time.time()-t1:.1f}s)",
          flush=True)

    print("[spec_alpha] loading eval prompts ...", flush=True)
    hs_prompts = load_hellaswag_prompts(args.hellaswag_n, args.seed)
    gen_spans = load_general_prompts(tok, args.general_spans, args.general_prompt)
    hs_prompt_ids = [(i, tok.encode(ctx, add_special_tokens=False))
                     for i, ctx in hs_prompts]
    print(f"  HellaSwag: {len(hs_prompt_ids)} contexts; "
          f"general text: {len(gen_spans)} PG19-validation spans", flush=True)

    # ---- greedy generation (once per prompt) ----
    print("[spec_alpha] greedy generation (HF eager, BF16) ...", flush=True)
    t2 = time.time()
    examples = []   # (corpus, src_id, P, G)
    for i, pid in hs_prompt_ids:
        G = greedy_generate(model, pid, args.hellaswag_gen, args.device)
        examples.append(("hellaswag", i, pid, G))
    for src_id, pid in gen_spans:
        G = greedy_generate(model, pid, args.general_gen, args.device)
        examples.append(("general", src_id, pid, G))
    print(f"  {len(examples)} prompts, "
          f"{sum(len(G) for *_, G in examples)} generated tokens  ({time.time()-t2:.1f}s)",
          flush=True)

    # ---- offline per-position α + chain-verify replay, aggregated per corpus ----
    corpus_stats = {}
    corpus_chain = {}
    global_stats = {"ngram": {"proposed": 0, "accepted": 0, "miss": 0},
                    "pld": {"proposed": 0, "accepted": 0, "miss": 0}}
    global_pos = 0

    for corpus, src_id, P, G in examples:
        stats, n_pos = measure_alpha(P, G, ngram)
        key = corpus
        cs = corpus_stats.setdefault(key, {"ngram": {"proposed": 0, "accepted": 0, "miss": 0},
                                           "pld": {"proposed": 0, "accepted": 0, "miss": 0},
                                           "n_pos": 0})
        cs["n_pos"] += n_pos
        for src in ("ngram", "pld"):
            for f in ("proposed", "accepted", "miss"):
                cs[src][f] += stats[src][f]
                global_stats[src][f] += stats[src][f]
        global_pos += n_pos

        cch = corpus_chain.setdefault(key, {})
        for src in ("ngram", "pld"):
            r = chain_sim(P, G, args.gamma, ngram, src)
            prev = cch.get(src)
            if prev is None:
                cch[src] = {"steps": 0, "proposed": 0, "accepted": 0,
                            "produced": 0, "cyc_base": 0, "cyc_spec": 0}
            p = cch[src]
            for f in ("steps", "proposed", "accepted", "produced",
                      "cyc_base", "cyc_spec"):
                p[f] += r[f]

    # ---- per-corpus α tables ----
    per_corpus = {}
    for corpus in sorted(corpus_stats):
        cs = corpus_stats[corpus]
        per_corpus[corpus] = {
            src: summarize(cs, cs["n_pos"], src) for src in ("ngram", "pld")
        }
        per_corpus[corpus]["n_positions"] = cs["n_pos"]

    overall = {src: summarize(global_stats, global_pos, src) for src in ("ngram", "pld")}
    overall["n_positions"] = global_pos

    # ---- chain-verify wall-clock (frozen qsim) ----
    chain_report = {}
    for corpus in sorted(corpus_chain):
        cch = corpus_chain[corpus]
        chain_report[corpus] = {}
        for src in ("ngram", "pld"):
            r = cch[src]
            alpha_chain = (r["accepted"] / r["proposed"]) if r["proposed"] else None
            speedup = (r["cyc_base"] / r["cyc_spec"]) if r["cyc_spec"] else None
            chain_report[corpus][src] = {
                "steps": r["steps"],
                "proposed": r["proposed"],
                "accepted": r["accepted"],
                "produced": r["produced"],
                "alpha_chain": alpha_chain,
                "cyc_base": r["cyc_base"],
                "cyc_spec": r["cyc_spec"],
                "speedup": speedup,
                "tok_s": TOK_PER_S * (speedup if speedup else 1.0),
            }

    # DECISION-linear cross-reference speedup, γ candidates always verified.
    def decision_linear(alpha_effective, gamma):
        if alpha_effective is None:
            return None
        toks_per_step = 1.0 + alpha_effective * gamma
        return toks_per_step * T_STEP / _verify_cost(gamma + 1)

    linear_ref = {
        src: {
            "alpha_effective": overall[src]["alpha_effective"],
            "tokens_per_step": (1.0 + overall[src]["alpha_effective"] * args.gamma)
                                if overall[src]["alpha_effective"] is not None else None,
            "speedup": decision_linear(overall[src]["alpha_effective"], args.gamma),
        }
        for src in ("ngram", "pld")
    }

    # ---- gate verdict ----
    # Two readings of "α" are reported:
    #   * α_cov  — literal per-proposal acceptance rate (DECISION's wording).
    #   * α_eff  — per-decoded-token acceptance (α_cov × coverage); this is what
    #              the DECISION gain formula (1+αγ -> 2.6–3.2×) actually needs,
    #              since a missed lookup produces no draft and no speedup.
    # The chain-verify wall-clock (chain_report) is the authoritative speedup.
    def verdict_entry(src):
        o = overall[src]
        speedup = None
        # combined chain speedup across corpora (weighted by produced tokens)
        cb = sum(chain_report[c][src]["cyc_base"] for c in chain_report)
        cs = sum(chain_report[c][src]["cyc_spec"] for c in chain_report)
        if cs:
            speedup = cb / cs
        return {
            "coverage": o["coverage"],
            "alpha_covered": o["alpha_covered"],
            "alpha_effective": o["alpha_effective"],
            "verdict_alpha_covered": (gate_verdict(o["alpha_covered"])
                                      if o["alpha_covered"] is not None else "n/a"),
            "verdict_alpha_effective": (gate_verdict(o["alpha_effective"])
                                        if o["alpha_effective"] is not None else "n/a"),
            "chain_sim_speedup": speedup,
        }
    verdict = {src: verdict_entry(src) for src in ("ngram", "pld")}

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
            "seed": args.seed,
            "gamma": args.gamma,
            "gate": {"alpha_start": ALPHA_START, "alpha_gen2": ALPHA_GEN2},
            "timing_model": {
                "t_step_cyc": T_STEP, "tok_per_s": TOK_PER_S,
                "lm_head_mac_step_cyc": LM_HEAD_MAC_STEP,
                "note": "T_verify(n) = T_STEP + (n-1)*lm_head_mac_step (DECISION §3 / speculative.md §3)",
            },
        },
        "draft_sources": {
            "ngram": {
                "desc": "static 4-gram dict (context=3 tokens -> most-frequent continuation)",
                "train_corpus": f"pg19 train (~{args.train_mb} MB, {len(train_texts)} books)",
                "n_tokens": n_train_tok, "n_4grams": n_4grams, "n_contexts": n_contexts,
            },
            "pld": {
                "desc": "Prompt Lookup Decoding: dynamic dict over prompt + accepted tokens "
                        "(most recent prior occurrence of current 3-gram)",
                "train_corpus": "none (reference = prompt + generated history)",
            },
        },
        "eval": {
            "hellaswag": {"n_examples": args.hellaswag_n, "gen_tokens_per": args.hellaswag_gen,
                          "seed": args.seed, "indices": [i for i, _ in hs_prompts]},
            "general": {"n_spans": args.general_spans, "prompt_tokens": args.general_prompt,
                        "gen_tokens_per": args.general_gen,
                        "span_ids": [s for s, _ in gen_spans]},
            "total_generated_tokens": sum(len(G) for *_, G in examples),
        },
        "alpha_per_corpus": per_corpus,
        "alpha_overall": overall,
        "chain_verify_wallclock": chain_report,
        "decision_linear_reference": linear_ref,
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }

    # ---- human-readable table ----
    def fmt(a):
        return "  n/a" if a is None else f"{a:5.3f}"

    print("\n==== per-corpus acceptance rate α ====", flush=True)
    print(f"{'corpus':<10} {'src':<7} {'n_pos':>6} {'proposed':>9} {'cov':>6} "
          f"{'α_cov':>8} {'α_eff':>8}", flush=True)
    for corpus in sorted(per_corpus):
        for src in ("ngram", "pld"):
            d = per_corpus[corpus][src]
            print(f"{corpus:<10} {src:<7} {d['n_positions']:>6} {d['proposed']:>9} "
                  f"{d['coverage']:>6.3f} {fmt(d['alpha_covered'])} {fmt(d['alpha_effective'])}",
                  flush=True)
    for src in ("ngram", "pld"):
        d = overall[src]
        print(f"{'OVERALL':<10} {src:<7} {d['n_positions']:>6} {d['proposed']:>9} "
              f"{d['coverage']:>6.3f} {fmt(d['alpha_covered'])} {fmt(d['alpha_effective'])}",
              flush=True)

    print("\n==== chain-verify (γ=%d) simulated wall-clock ====" % args.gamma, flush=True)
    print(f"{'corpus':<10} {'src':<7} {'produced':>8} {'α_chain':>8} {'speedup':>8} "
          f"{'tok/s':>8}", flush=True)
    for corpus in sorted(chain_report):
        for src in ("ngram", "pld"):
            d = chain_report[corpus][src]
            sp = "  n/a" if d["speedup"] is None else f"{d['speedup']:6.2f}x"
            a = "  n/a" if d["alpha_chain"] is None else f"{d['alpha_chain']:6.3f}"
            print(f"{corpus:<10} {src:<7} {d['produced']:>8} {a:>8} {sp:>8} "
                  f"{d['tok_s']:>8.0f}", flush=True)

    print("\n==== gate verdict (DECISION §3.1) ====", flush=True)
    for src in ("ngram", "pld"):
        v = verdict[src]
        sp = "n/a" if v["chain_sim_speedup"] is None else f"{v['chain_sim_speedup']:.2f}x"
        print(f"  {src:<6} cov={v['coverage']:.3f} α_cov={fmt(v['alpha_covered'])} "
              f"α_eff={fmt(v['alpha_effective'])} sim={sp}", flush=True)
        print(f"          literal α_cov -> {v['verdict_alpha_covered']}", flush=True)
        print(f"          effective α  -> {v['verdict_alpha_effective']}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[spec_alpha] wrote {args.out}  (total {result['wall_s']:.1f}s)", flush=True)
    return result


if __name__ == "__main__":
    main()
