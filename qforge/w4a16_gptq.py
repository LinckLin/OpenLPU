"""W4A16 GPTQ g64/g128 + SpinQuant-nohad rotation — 0.6B full-model verification.

Decision C-track (DECISION.md §3.3): GPTQ (2nd-order error compensation, Hessian
from the golden pg19 calibration set) + SpinQuant-nohad compile-time rotation
absorption (W-side Hadamard folded into weights; ISA/qbin format unchanged),
full 0.6B model.  Gate: 20-token greedy cross-agreement >= 8/10 (16/20) OR
Delta-PPL <= 2% vs the BF16 full baseline -> upgrade; else archive W4A16.

Outputs JSON to --out (default /tmp/w4a16-gptq-results.json); the decision
report lives in docs/perf-research/decision/w4a16-gptq.md.

Configs measured:
  rtn128        plain symmetric RTN g128 (unrotated) — control (== existing 3/20)
  gptq128       GPTQ g128, no rotation (ablation)
  gptq64        GPTQ g64,  no rotation (ablation)
  gptq128+rot   GPTQ g128 + SpinQuant-nohad R1+R2 (the decision recipe)
  gptq64+rot    GPTQ g64  + SpinQuant-nohad R1+R2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qforge.gptq import gptq_quantize, dequant_to_bf16            # noqa: E402
from qforge.rot_quant import apply_spinquant                       # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
BASELINE_JSON = os.path.join(ROOT, "docs/perf-research/quality-baseline/quality_baseline.json")
P1_PROMPT = "Explain the concept of a transformer neural network and its attention mechanism:"
BASELINE = [3555, 374, 279, 6672, 1948, 264, 42578, 323, 264, 29728,
            3922, 304, 4586, 30, 3555, 374, 279, 3476, 315, 279]

HIDDEN = 1024
N_LAYERS = 28
N_KV_HEADS = 8
HEAD_DIM = 128

# linear projection state-dict keys (embed is NOT quantized; q_norm/k_norm untouched)
PROJ_KEYS = []
for _l in range(N_LAYERS):
    p = f"model.layers.{_l}"
    PROJ_KEYS += [f"{p}.self_attn.{k}_proj.weight" for k in ("q", "k", "v", "o")]
    PROJ_KEYS += [f"{p}.mlp.{k}_proj.weight" for k in ("gate", "up", "down")]
PROJ_KEYS.append("lm_head.weight")


def load_model(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    return tok, model


def _untie(model: nn.Module, sd: dict) -> None:
    """Break the embed<->lm_head tie after a load_state_dict (the rotation makes
    them differ; the tie would otherwise let one overwrite the other)."""
    dev = next(model.parameters()).device
    model.model.embed_tokens.weight = nn.Parameter(
        sd["model.embed_tokens.weight"].detach().clone().to(dev))
    model.lm_head.weight = nn.Parameter(sd["lm_head.weight"].detach().clone().to(dev))


def build_model_from_sd(sd: dict, device="cuda"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    model.load_state_dict(sd, assign=False)
    _untie(model, sd)
    return model


def build_rotated(model: nn.Module, seed: int):
    sd_rot = apply_spinquant(model.state_dict(), hidden=HIDDEN, n_layers=N_LAYERS,
                             n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
                             seed=seed, device=next(model.parameters()).device)
    return build_model_from_sd(sd_rot), sd_rot


# ---------------------------------------------------------------------------
# calibration activation collection (forward pre-hooks)
# ---------------------------------------------------------------------------
def collect_calib(model: nn.Module, tok, n_seqs: int, seq_len: int) -> dict:
    """Capture each linear layer's input activations over pg19 spans.

    Returns {sd_key: X [n, K] bf16 on CPU}.  Deterministic (first n_seqs pg19
    validation books, interior span [seq_len, 2*seq_len) like the golden PPL).
    """
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19")["validation"]
    spans = []
    for i, row in enumerate(ds):
        ids = tok.encode(row["text"], add_special_tokens=False)
        if len(ids) >= 2 * seq_len:
            spans.append(ids[seq_len:2 * seq_len])
            if len(spans) >= n_seqs:
                break

    # module name -> sd key
    mod2key = {}
    for l in range(N_LAYERS):
        p = f"model.layers.{l}"
        for k in ("q", "k", "v", "o"):
            mod2key[f"{p}.self_attn.{k}_proj"] = f"{p}.self_attn.{k}_proj.weight"
        for k in ("gate", "up", "down"):
            mod2key[f"{p}.mlp.{k}_proj"] = f"{p}.mlp.{k}_proj.weight"
    mod2key["lm_head"] = "lm_head.weight"

    buf = {k: [] for k in PROJ_KEYS}
    hooks = []
    device = next(model.parameters()).device

    def make_hook(key):
        def hook(mod, args):
            x = args[0].detach()                    # [batch, seq, K] or [seq, K]
            buf[key].append(x.reshape(-1, x.shape[-1]).cpu())
        return hook

    for name, mod in model.named_modules():
        if name in mod2key:
            hooks.append(mod.register_forward_pre_hook(make_hook(mod2key[name])))

    with torch.no_grad():
        for span in spans:
            inp = torch.tensor([span], device=device)
            model(inp)

    for h in hooks:
        h.remove()

    out = {k: torch.cat(v, dim=0) for k, v in buf.items()}   # [n, K] bf16 cpu
    return out


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------
def _rtn_deq(W: torch.Tensor, group: int) -> torch.Tensor:
    """Symmetric RTN INT4 -> bf16 dequantized weight [N, K]."""
    N, K = W.shape
    G = K // group
    Wf = W.to(torch.float32)
    wr = Wf.reshape(N, G, group)
    sw = wr.abs().max(dim=2).values / 7.0                 # [N, G]
    wqi = (wr / sw[:, :, None]).round().clamp(-7, 7)
    sw_bf = sw.to(torch.bfloat16).to(torch.float32)
    return (wqi * sw_bf[:, :, None]).reshape(N, K).to(torch.bfloat16)


def quantize_sd(sd: dict, calib: dict, group: int, method: str, device="cuda") -> dict:
    """Return a NEW state_dict with each linear weight replaced by the bf16
    dequantized W4A16 weight.  ``calib`` maps sd_key -> X [n, K] (bf16 cpu).
    """
    out = {k: v.detach().clone() for k, v in sd.items()}
    for key in PROJ_KEYS:
        W = sd[key].to(torch.float32).to(device)
        if method == "rtn":
            deq = _rtn_deq(sd[key].to(device), group)
        else:  # gptq
            X = calib[key].to(torch.float32).to(device)
            Q, scales = gptq_quantize(W, X, group)
            deq = dequant_to_bf16(Q, scales, group)
        out[key] = deq.detach().cpu()
    return out


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(model: nn.Module, tok, prompt: str, n: int) -> list[int]:
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(next(model.parameters()).device)
    full = list(ids[0].cpu().numpy())
    out = []
    for _ in range(n):
        inp = torch.tensor([full], device=next(model.parameters()).device)
        logits = model(inp).logits[0, -1]
        t = int(logits.argmax().item())
        out.append(t)
        full.append(t)
    return out


@torch.no_grad()
def full_attention_nll(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits                       # [1, L, V] bf16
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = input_ids[:, 1:]
    nll = -lp[:, :-1, :].gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    return nll[0]                                          # [L-1] fp32


def measure_ppl(model: nn.Module, baseline_json: str, device="cuda") -> dict:
    base = json.load(open(baseline_json))
    res = {}
    for L, spans in base["ppl"].items():
        d_nlls = []
        q_nlls = []
        b_nlls = []
        for s in spans:
            ids = torch.tensor([s["token_ids"]], device=device)
            q_nll = full_attention_nll(model, ids)
            b_nll = torch.tensor(s["per_token_nll"], device=device, dtype=torch.float32)
            d_nlls.append((q_nll - b_nll).cpu())
            q_nlls.append(q_nll.cpu())
            b_nlls.append(b_nll.cpu())
        d = torch.cat(d_nlls)
        q = torch.cat(q_nlls)
        b = torch.cat(b_nlls)
        # pooled PPL: exp(mean NLL)
        q_ppl = float(torch.exp(q.mean()).item())
        b_ppl = float(torch.exp(b.mean()).item())
        # per-token Delta-PPL (O1 gate definition): exp(mean DeltaNLL) - 1
        delta_ppl = float(torch.exp(d.mean()).item() - 1)
        res[L] = {
            "ppl_quantized_pooled": q_ppl,
            "ppl_baseline_pooled": b_ppl,
            "delta_ppl_per_token": delta_ppl,
            "delta_ppl_relative_pooled": (q_ppl - b_ppl) / b_ppl,
            "mean_delta_nll": float(d.mean().item()),
            "n_tokens": int(d.numel()),
        }
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--baseline", default=BASELINE_JSON)
    ap.add_argument("--out", default="/tmp/w4a16-gptq-results.json")
    ap.add_argument("--n-calib-seqs", type=int, default=32)
    ap.add_argument("--calib-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--configs", default="rtn128,gptq128,gptq64,gptq128+rot,gptq64+rot")
    args = ap.parse_args()

    device = "cuda"
    t0 = time.time()
    tok, model = load_model(device)
    print(f"loaded model in {time.time()-t0:.1f}s", flush=True)

    # sanity: original model reproduces the BF16 baseline greedy decode
    orig_tokens = greedy_decode(model, tok, P1_PROMPT, args.max_new)
    base_match = sum(a == b for a, b in zip(orig_tokens, BASELINE))
    print(f"BF16 original vs baseline: {base_match}/{args.max_new}", flush=True)

    # rotated model (SpinQuant-nohad) + its calibration activations
    model_rot, sd_rot = build_rotated(model, args.seed)
    rot_tokens = greedy_decode(model_rot, tok, P1_PROMPT, args.max_new)
    rot_match = sum(a == b for a, b in zip(rot_tokens, BASELINE))
    print(f"rotated-unquantized vs baseline: {rot_match}/{args.max_new} "
          f"(bf16 reparameterization noise floor)", flush=True)

    calib_rot = collect_calib(model_rot, tok, args.n_calib_seqs, args.calib_seq_len)
    calib_orig = collect_calib(model, tok, args.n_calib_seqs, args.calib_seq_len)
    print(f"calibration collected ({args.n_calib_seqs}x{args.calib_seq_len} tokens)", flush=True)

    sd_orig = {k: v.detach().clone() for k, v in model.state_dict().items()}

    results = {
        "meta": {
            "model": "Qwen/Qwen3-0.6B", "dtype": "bfloat16",
            "attention": "eager (HF native full attention)", "device": device,
            "seed": args.seed, "prompt": P1_PROMPT, "baseline_tokens": BASELINE,
            "n_calib_seqs": args.n_calib_seqs, "calib_seq_len": args.calib_seq_len,
            "rotation": "SpinQuant-nohad R1(residual, global, RMSNorm-folded) + R2(V-O head pair), random Hadamard",
            "gate_cross_agreement": ">= 8/10 (16/20)", "gate_delta_ppl": "<= 2%",
            "bf16_original_vs_baseline_match": f"{base_match}/{args.max_new}",
            "rotated_unquantized_vs_baseline_match": f"{rot_match}/{args.max_new}",
        },
        "configs": {},
    }

    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if not cfg:
            continue
        rotated = cfg.endswith("+rot")
        base = cfg[:-4] if rotated else cfg
        method = "".join(ch for ch in base if not ch.isdigit())
        g = int("".join(ch for ch in base if ch.isdigit()) or 128)
        if method == "rtn":
            g = 128
        print(f"\n=== config {cfg} (method={method}, group={g}, rotated={rotated}) ===", flush=True)

        sd_base = sd_rot if rotated else sd_orig
        calib = calib_rot if rotated else calib_orig
        tq = time.time()
        sd_q = quantize_sd(sd_base, calib, g, "rtn" if method == "rtn" else "gptq", device)
        model_q = build_model_from_sd(sd_q, device)
        print(f"  quantized in {time.time()-tq:.1f}s", flush=True)

        tokens = greedy_decode(model_q, tok, P1_PROMPT, args.max_new)
        match = sum(a == b for a, b in zip(tokens, BASELINE))
        print(f"  cross-agreement: {match}/{args.max_new} "
              f"{'PASS' if match >= 0.8*args.max_new else 'FAIL'}", flush=True)

        entry = {
            "method": method, "group": g, "rotated": rotated,
            "new_tokens": tokens, "baseline": BASELINE,
            "n_match": match, "n_total": args.max_new,
            "cross_agreement_pass": match >= 0.8 * args.max_new,
        }
        if not args.skip_ppl:
            tq = time.time()
            entry["ppl"] = measure_ppl(model_q, args.baseline, device)
            print(f"  PPL measured in {time.time()-tq:.1f}s", flush=True)
            for L, r in entry["ppl"].items():
                print(f"    L={L}: PPL {r['ppl_quantized_pooled']:.3f} "
                      f"(base {r['ppl_baseline_pooled']:.3f}) | "
                      f"delta_ppl {r['delta_ppl_per_token']*100:.3f}%", flush=True)

        results["configs"][cfg] = entry
        del model_q, sd_q
        torch.cuda.empty_cache()

    results["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return results


if __name__ == "__main__":
    main()
