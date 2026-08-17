"""O2 INT8 KV quantization — numerical feasibility + qsim ceiling + RTL-change basis.

Scheme: symmetric INT8 KV with per-group scales.  The group granularity is a
parameter (`--k-group` / `--v-group`, over the head_dim=128):

  * group=128  -> per-head (one scale per (head, token)) — the roadmap's
    "per-128-group" reading (head_dim == 128, so 1 group = 1 head).
  * group=1    -> per-channel (one scale per (head, token, dim)) — the finest
    granularity, the KVQuant-style fix for RoPE/QK-norm key channel outliers.
  * group=0    -> no quantization (BF16), for the windowing-only baseline.

At KV.APPEND the golden BF16 K (post-RoPE) / V are quantized
  scale = bf16(max|x_group| / 127),  q = round(x / scale) clipped to [-127, 127],
stored as int8 (1 B/element) + BF16 scale metadata; at KV.LOAD dequantized
  x ~= q * scale  back to BF16 for attention.  The scale is derived from the
golden activation max-abs statistics (== qrun.weights._group_scales / qforge.quant).

The driver validates the scheme on the torch reference model (no RTL change)
against (a) the golden full-attention PPL baseline (ΔPPL gate <= 2%), (b) the
BF16 full-attention 20-token greedy baseline (cross-consistency >= 16/20 +
divergence-position logits rel-error), and recomputes the qsim decode ceiling
with KV read bandwidth halved (q=0.5) — roadmap 866 -> 1009 (W=2048, R = S + W).
The §6 ceiling table also computes the two alternatives: V-only (3088 B/token
-> 930.8) and dynamic per-channel-K (4096 B/token -> 866.0, zero net gain);
`--ceiling-only` emits the whole table without loading the model.
`--static-kchan` reproduces the static per-channel-K failure (report §2 last
row, 66.9% ΔPPL @ 4K/4 spans) and writes `docs/perf-research/o2-kv-int8-results.json`.

Usage:
  python3 qrun/o2_kv_int8.py \
      --baseline docs/perf-research/quality-baseline/quality_baseline.json \
      --sinks 4 --window 2048 --seq-lens 4096,8192 \
      --k-group 128 --v-group 128 --out /tmp/o2-kv-int8.json
  python3 qrun/o2_kv_int8.py --ceiling-only --out /tmp/o2-ceiling.json
  python3 qrun/o2_kv_int8.py --static-kchan --skip-ppl --skip-cross
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_REF_DIR = os.path.join(ROOT, "ref")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)
from model import (Qwen3Ref, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, HIDDEN,  # noqa: E402
                   VOCAB, ATTN_SCALE, N_REP, rmsnorm, apply_rope, rope_embeddings,
                   repeat_kv, causal_mask)

from qrun.reference import load_hf  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
BASELINE = "docs/perf-research/quality-baseline/quality_baseline.json"

S_DEFAULT = 4
W_DEFAULT = 2048

O2_DELTA_PPL_GATE = 0.02   # <= 2% relative ΔPPL (windowed+INT8-KV vs full) — same gate as O1.
CROSS_GATE_10 = 8          # p6-style >= 8/10; 20 tokens -> >= 16/20.

# INT8 KV per-token-per-layer byte budget (Qwen3-0.6B, 8 KV heads):
KV_DATA_INT8_PER_TOKEN = 2048    # 8 heads x 2 (K,V) x 128 x 1 B
KV_SCALE_PER_TOKEN = 32          # 8 heads x 2 (K,V) x 2 B BF16 scale metadata (per-head)

# §6 ceiling alternatives (per-token-per-layer KV read budgets, W=2048):
VONLY_KV_BYTES = 3088            # K BF16 2048 + V INT8 1024 + V per-head scale 16
VONLY_KV_DATA = 3072             # K BF16 2048 + V INT8 1024 (SRAM staging write)
VONLY_KV_SCALE = 16              # V per-head BF16 scale (8 heads x 2 B)
KCHAN_DYN_KV_BYTES = 4096        # K+V INT8 data 2048 + K per-channel scale 2048 (V 16 B negligible)
KCHAN_DYN_DATA = 2048            # K+V INT8 data (SRAM staging write)
KCHAN_DYN_SCALE = 2048           # K per-channel BF16 scale (8 heads x 128 x 2 B)

# Static per-channel K repro (report §2 last row): the per-channel K scale is
# calibrated from each span's first `STATIC_CALIB_TOKENS` post-RoPE K positions
# and reused for the whole span (dynamic per-head V).  Fails: RoPE makes
# per-channel K magnitude token-dependent, so a static scale clips/underflows.
STATIC_CALIB_TOKENS = 128
STATIC_SPANS = 4                 # baseline["ppl"]["4096"][:4]
STATIC_RESULTS = "docs/perf-research/o2-kv-int8-results.json"


# =============================================================================
# StreamingLLM windowed mask (matches ref/model.causal_mask [1, seq, seq] shape)
# =============================================================================

def streaming_mask(L: int, S: int, W: int, device) -> torch.Tensor:
    """[1, L, L] additive bf16 mask: query i attends to key j iff j <= i and
    (j < S sink or j >= i - W + 1 rolling)."""
    idx = torch.arange(L, device=device)
    causal = idx[:, None] >= idx[None, :]
    sink = idx[None, :] < S
    window = idx[None, :] >= (idx[:, None] - W + 1)
    allowed = causal & (sink | window)
    minv = torch.finfo(torch.bfloat16).min
    mask = torch.where(allowed,
                       torch.zeros((), device=device, dtype=torch.bfloat16),
                       torch.full((), minv, device=device, dtype=torch.bfloat16))
    return mask[None]


def windowed_row_mask(L: int, S: int, W: int, device) -> torch.Tensor:
    """[1, 1, L] additive bf16 row mask for the single decode query at pos = L-1."""
    idx = torch.arange(L, device=device)
    pos = L - 1
    allowed = (idx < S) | (idx >= pos - W + 1)
    minv = torch.finfo(torch.bfloat16).min
    mask = torch.where(allowed,
                       torch.zeros((), device=device, dtype=torch.bfloat16),
                       torch.full((), minv, device=device, dtype=torch.bfloat16))
    return mask[None, None]


# =============================================================================
# INT8 KV quantize + dequantize (symmetric, per-group over head_dim, golden stats)
# =============================================================================

def quantize_dequant_int8(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Symmetric per-group INT8 roundtrip of x: [heads, seq, dim] bf16.

    scale = bf16(max|x_group| / 127) per (head, token, group); q = round(x/scale)
    clipped to [-127, 127]; returns the dequantized x ~= q * scale in x.dtype.
    group=128 -> per-head; group=1 -> per-channel (per-dim)."""
    xf = x.float()
    heads, seq, dim = x.shape
    G = dim // group
    xg = xf.reshape(heads, seq, G, group)
    amax = xg.abs().amax(dim=-1, keepdim=True)                 # [heads, seq, G, 1]
    scale = (amax / 127.0).clamp(min=1e-6)
    scale_bf16 = scale.to(torch.bfloat16).to(torch.float32)    # BF16 metadata round
    q = (xg / scale_bf16).round().clamp(-127.0, 127.0)
    dq = q * scale_bf16
    return dq.reshape(heads, seq, dim).to(x.dtype)


# =============================================================================
# Reference forward: full-span (prefill) + single decode step.  k_group/v_group
# = 0 -> BF16 (no quant); else symmetric INT8 with that per-group granularity.
# =============================================================================

@torch.no_grad()
def forward_span(ref, token_ids, S=None, W=None, k_group=0, v_group=0, k_static_calib=0):
    """Full-span (prefill) forward over token_ids [L].  Returns (logits, cache).

    S=None -> full causal attention; else StreamingLLM windowed (sinks + window).
    cache: list of {"k": [8, L, 128], "v": [8, L, 128]} (dequantized if quantized).
    k_static_calib > 0: K is quantized per-channel with a static scale calibrated
    from the first `k_static_calib` positions of this span (overrides k_group)."""
    seq = token_ids.shape[0]
    device = ref.device
    positions = torch.arange(0, seq, device=device)
    cos, sin = rope_embeddings(positions, ref.inv_freq.to(device), torch.bfloat16)
    hidden = ref.embed[token_ids.to(device)]
    if S is None:
        mask = causal_mask(seq, seq, 0, torch.bfloat16, device)
    else:
        mask = streaming_mask(seq, S, W, device)
    new_cache = []
    for lw in ref.layers:
        norm_in = rmsnorm(hidden, lw.in_norm)
        q_raw = F.linear(norm_in, lw.q_proj)
        k_raw = F.linear(norm_in, lw.k_proj)
        v_raw = F.linear(norm_in, lw.v_proj)
        q = rmsnorm(q_raw.view(seq, N_HEADS, HEAD_DIM), lw.q_norm).transpose(0, 1)
        k = rmsnorm(k_raw.view(seq, N_KV_HEADS, HEAD_DIM), lw.k_norm).transpose(0, 1)
        v = v_raw.view(seq, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)
        if k_static_calib:
            amax = k_rot[:, :k_static_calib, :].float().abs().amax(dim=1, keepdim=True)
            cal = (amax.clamp(min=1e-6) / 127.0).to(torch.bfloat16).to(torch.float32)
            qk = (k_rot.float() / cal).round().clamp(-127.0, 127.0)
            k_full = (qk * cal).to(k_rot.dtype)
        elif k_group:
            k_full = quantize_dequant_int8(k_rot, k_group)
        else:
            k_full = k_rot
        v_full = quantize_dequant_int8(v, v_group) if v_group else v
        k_rep = repeat_kv(k_full, N_REP)
        v_rep = repeat_kv(v_full, N_REP)
        scores = torch.matmul(q_rot, k_rep.transpose(-1, -2)) * ATTN_SCALE
        masked = scores + mask
        probs = F.softmax(masked, dim=-1, dtype=torch.float32).to(q_rot.dtype)
        ctx = torch.matmul(probs, v_rep)
        attn_out = ctx.transpose(0, 1).reshape(seq, N_HEADS * HEAD_DIM).contiguous()
        o = F.linear(attn_out, lw.o_proj)
        hidden1 = hidden + o
        norm_post = rmsnorm(hidden1, lw.post_norm)
        gate = F.linear(norm_post, lw.gate)
        up = F.linear(norm_post, lw.up)
        act = F.silu(gate) * up
        down = F.linear(act, lw.down)
        hidden = hidden1 + down
        new_cache.append({"k": k_full, "v": v_full})
    final = rmsnorm(hidden, ref.final_norm)
    logits = F.linear(final, ref.lm_head)
    return logits, new_cache


@torch.no_grad()
def decode_step(ref, token_id, pos, cache, S=None, W=None, k_group=0, v_group=0):
    """One decode step at absolute position `pos` (cache length == pos).

    Returns (logits [1, vocab], new_cache)."""
    device = ref.device
    token = torch.tensor([token_id], device=device)
    positions = torch.tensor([pos], device=device)
    cos, sin = rope_embeddings(positions, ref.inv_freq.to(device), torch.bfloat16)
    hidden = ref.embed[token]
    new_cache = []
    for i, lw in enumerate(ref.layers):
        norm_in = rmsnorm(hidden, lw.in_norm)
        q_raw = F.linear(norm_in, lw.q_proj)
        k_raw = F.linear(norm_in, lw.k_proj)
        v_raw = F.linear(norm_in, lw.v_proj)
        q = rmsnorm(q_raw.view(1, N_HEADS, HEAD_DIM), lw.q_norm).transpose(0, 1)
        k = rmsnorm(k_raw.view(1, N_KV_HEADS, HEAD_DIM), lw.k_norm).transpose(0, 1)
        v = v_raw.view(1, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)
        k_new = quantize_dequant_int8(k_rot, k_group) if k_group else k_rot
        v_new = quantize_dequant_int8(v, v_group) if v_group else v
        k_full = torch.cat([cache[i]["k"], k_new], dim=1)   # [8, pos+1, 128]
        v_full = torch.cat([cache[i]["v"], v_new], dim=1)
        k_rep = repeat_kv(k_full, N_REP)
        v_rep = repeat_kv(v_full, N_REP)
        scores = torch.matmul(q_rot, k_rep.transpose(-1, -2)) * ATTN_SCALE
        L = pos + 1
        if S is None:
            mask = causal_mask(1, L, pos, torch.bfloat16, device)
        else:
            mask = windowed_row_mask(L, S, W, device)
        masked = scores + mask
        probs = F.softmax(masked, dim=-1, dtype=torch.float32).to(q_rot.dtype)
        ctx = torch.matmul(probs, v_rep)
        attn_out = ctx.transpose(0, 1).reshape(1, N_HEADS * HEAD_DIM).contiguous()
        o = F.linear(attn_out, lw.o_proj)
        hidden1 = hidden + o
        norm_post = rmsnorm(hidden1, lw.post_norm)
        gate = F.linear(norm_post, lw.gate)
        up = F.linear(norm_post, lw.up)
        act = F.silu(gate) * up
        down = F.linear(act, lw.down)
        hidden = hidden1 + down
        new_cache.append({"k": k_full, "v": v_full})
    final = rmsnorm(hidden, ref.final_norm)
    logits = F.linear(final, ref.lm_head)
    return logits, new_cache


@torch.no_grad()
def ref_greedy(ref, prompt_ids, n, S=None, W=None, k_group=0, v_group=0):
    """Greedy decode; returns (new_tokens list[int], logits_list list[torch])."""
    logits, cache = forward_span(ref, prompt_ids, S, W, k_group, v_group)
    new_tokens = []
    logits_list = [logits[-1]]
    for _ in range(n):
        tok = int(logits[-1].argmax(dim=-1).item())
        new_tokens.append(tok)
        logits, cache = decode_step(ref, tok, prompt_ids.shape[0] + len(new_tokens) - 1,
                                    cache, S, W, k_group, v_group)
        logits_list.append(logits[-1])
    return new_tokens, logits_list


@torch.no_grad()
def span_nll_ref(ref, token_ids, S, W, k_group, v_group, k_static_calib=0) -> torch.Tensor:
    """Per-token NLL [L-1] fp32 under the configured attention/quantization."""
    logits, _ = forward_span(ref, token_ids, S, W, k_group, v_group, k_static_calib)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    targets = token_ids[1:]
    nll = -log_probs[:-1, :].gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return nll


# =============================================================================
# ΔPPL gate (windowed BF16 vs golden full, and windowed+INT8-KV vs golden full)
# =============================================================================

def evaluate_ppl_o2(ref, baseline_ppl, seq_lens, S, W, device, k_group, v_group):
    """Score the baseline JSON's exact spans with (windowed BF16) and
    (windowed + quantized KV) attention; ΔPPL vs golden full per_token_nll."""
    out = {}
    for L in seq_lens:
        key = str(L)
        spans = baseline_ppl[key]
        L = int(L)
        for mode in ("windowed_bf16", "windowed_quant"):
            kg, vg = (0, 0) if mode == "windowed_bf16" else (k_group, v_group)
            rows = []
            all_dnll = []
            all_w = []
            all_f = []
            for s in spans:
                ids = torch.tensor(s["token_ids"], dtype=torch.long, device=device)
                wnll = span_nll_ref(ref, ids, S, W, kg, vg)
                fnll = np.asarray(s["per_token_nll"], dtype=np.float64)
                dnll = wnll.double().cpu().numpy() - fnll
                mean_w = float(wnll.mean().item())
                mean_f = float(s["mean_nll"])
                dppl = float(np.exp(dnll.mean()) - 1.0)
                rows.append({
                    "sample_id": s["sample_id"],
                    "mean_nll_full": mean_f,
                    "mean_nll_windowed": mean_w,
                    "ppl_full": float(np.exp(mean_f)),
                    "ppl_windowed": float(np.exp(mean_w)),
                    "delta_ppl": dppl,
                })
                all_dnll.append(dnll)
                all_w.append(mean_w)
                all_f.append(mean_f)
            pooled_dnll = np.concatenate(all_dnll)
            pooled_dppl = float(np.exp(pooled_dnll.mean()) - 1.0)
            out.setdefault(key, {})[mode] = {
                "per_sample": rows,
                "pooled_mean_dnll": float(pooled_dnll.mean()),
                "pooled_delta_ppl": pooled_dppl,
                "pooled_ppl_full": float(np.exp(np.mean(all_f))),
                "pooled_ppl_windowed": float(np.exp(np.mean(all_w))),
                "gate": O2_DELTA_PPL_GATE,
                "pass": bool(pooled_dppl <= O2_DELTA_PPL_GATE),
            }
    return out


def evaluate_static_kchan(ref, baseline_ppl, S, W, device, v_group=128,
                          calib_tokens=STATIC_CALIB_TOKENS, n_spans=STATIC_SPANS):
    """Static per-channel K (scale calibrated from each span's first
    ``calib_tokens`` post-RoPE K positions) + dynamic per-head V; ΔPPL vs golden
    full.  Report §2 last row: 66.9% @ 4K (4 spans) — static scale fails."""
    spans = baseline_ppl["4096"][:n_spans]
    rows = []
    all_dnll = []
    all_w = []
    all_f = []
    for s in spans:
        ids = torch.tensor(s["token_ids"], dtype=torch.long, device=device)
        wnll = span_nll_ref(ref, ids, S, W, 0, v_group, k_static_calib=calib_tokens)
        fnll = np.asarray(s["per_token_nll"], dtype=np.float64)
        dnll = wnll.double().cpu().numpy() - fnll
        mean_w = float(wnll.mean().item())
        mean_f = float(s["mean_nll"])
        dppl = float(np.exp(dnll.mean()) - 1.0)
        rows.append({
            "sample_id": s["sample_id"],
            "mean_nll_full": mean_f,
            "mean_nll_static": mean_w,
            "ppl_full": float(np.exp(mean_f)),
            "ppl_static": float(np.exp(mean_w)),
            "delta_ppl": dppl,
        })
        all_dnll.append(dnll)
        all_w.append(mean_w)
        all_f.append(mean_f)
    pooled_dnll = np.concatenate(all_dnll)
    pooled_dppl = float(np.exp(pooled_dnll.mean()) - 1.0)
    return {
        "seq_len": 4096,
        "n_spans": len(spans),
        "calib_tokens": calib_tokens,
        "v_group": v_group,
        "per_sample": rows,
        "pooled_mean_dnll": float(pooled_dnll.mean()),
        "pooled_delta_ppl": pooled_dppl,
        "pooled_ppl_full": float(np.exp(np.mean(all_f))),
        "pooled_ppl_static": float(np.exp(np.mean(all_w))),
        "gate": O2_DELTA_PPL_GATE,
        "pass": bool(pooled_dppl <= O2_DELTA_PPL_GATE),
    }


# =============================================================================
# 20-token cross-consistency
# =============================================================================

FIXED_TEXT = ("The transformer architecture processes sequences in parallel. "
              "Attention weighs the relevance of every token against the query. ")


def build_seq(tok, target_len: int) -> np.ndarray:
    base = tok(FIXED_TEXT, return_tensors="pt")["input_ids"][0]
    reps = (target_len + base.shape[0]) // base.shape[0]
    return base.repeat(reps)[:target_len].numpy()


def cross_consistency(ref, tok, prompt_len, n_tokens, S, W, device, k_group, v_group):
    pid = build_seq(tok, prompt_len)
    pid_t = torch.from_numpy(pid)

    print(f"  full BF16 attention baseline ({n_tokens} tokens) ...", flush=True)
    full_tokens, full_logits = ref_greedy(ref, pid_t, n_tokens)
    print(f"  windowed BF16 (W={W}) ...", flush=True)
    win_tokens, win_logits = ref_greedy(ref, pid_t, n_tokens, S=S, W=W)
    print(f"  windowed + INT8-KV (W={W}, k={k_group}, v={v_group}) ...", flush=True)
    q_tokens, q_logits = ref_greedy(ref, pid_t, n_tokens, S=S, W=W,
                                    k_group=k_group, v_group=v_group)

    def _match(a, b):
        return [int(x == y) for x, y in zip(a, b)]

    def _rel_err(logits_a, logits_b, ta, tb):
        per = {}
        div = {}
        for k in range(n_tokens):
            la = logits_a[k].detach().float().cpu().numpy()
            lb = logits_b[k].detach().float().cpu().numpy()
            rel = float(np.abs(la - lb).max() / max(np.abs(lb).max(), 1e-6))
            per[str(k)] = rel
            if ta[k] != tb[k]:
                div[str(k)] = rel
        return per, div

    per_q, div_q = _rel_err(q_logits, full_logits, q_tokens, full_tokens)
    per_w, div_w = _rel_err(win_logits, full_logits, win_tokens, full_tokens)

    return {
        "prompt_len": prompt_len,
        "n_tokens": n_tokens,
        "full_tokens": [int(x) for x in full_tokens],
        "windowed_bf16_tokens": [int(x) for x in win_tokens],
        "windowed_quant_tokens": [int(x) for x in q_tokens],
        "windowed_bf16_vs_full": {
            "match": _match(win_tokens, full_tokens),
            "n_match": sum(_match(win_tokens, full_tokens)),
            "divergence_positions": [i for i in range(n_tokens)
                                     if win_tokens[i] != full_tokens[i]],
            "per_pos_rel_err": per_w,
            "divergence_rel_err": div_w,
        },
        "windowed_quant_vs_full": {
            "match": _match(q_tokens, full_tokens),
            "n_match": sum(_match(q_tokens, full_tokens)),
            "divergence_positions": [i for i in range(n_tokens)
                                     if q_tokens[i] != full_tokens[i]],
            "per_pos_rel_err": per_q,
            "divergence_rel_err": div_q,
            "pass": sum(_match(q_tokens, full_tokens)) >= (CROSS_GATE_10 * n_tokens // 10),
        },
        "cross_gate_10": CROSS_GATE_10,
    }


# =============================================================================
# qsim decode ceiling with KV read bandwidth halved (q=0.5)
# =============================================================================

class _KVInt8Demands:
    """Decode per-layer demands for INT8 KV (data 1 B/element + BF16 scale)."""

    def __init__(self, R, kv_data_bytes, kv_scale_bytes):
        from qsim.timing import hbm_read_cycles, sram_write_cycles
        from qsim.timing_p6 import WEIGHT_LAYER
        self.ctx = R
        self.weight_read = hbm_read_cycles(WEIGHT_LAYER)
        self.kv_read = hbm_read_cycles(kv_data_bytes + kv_scale_bytes)
        self.kv_sram_write = sram_write_cycles(kv_data_bytes)


def ceiling_table(S, W, seq_lens=(4096, 8192)):
    """Recompute the decode ceiling: full (ctx), windowed BF16 (R=S+W), and the
    INT8-KV alternatives.  ``q05`` ignores the 32 B/token scale metadata
    (roadmap convention); ``scale32`` includes it; ``vonly`` halves V only
    (K BF16 2048 + V INT8 1024 + scale 16 = 3088 B/token); ``kchan_dynamic``
    quantizes K+V with per-channel K scale (data 2048 + scale 2048 = 4096
    B/token, zero net gain — report §6)."""
    from qsim.timing import hbm_read_cycles
    from qsim.timing_p6 import (WEIGHT_LAYER, decode_demands, schedule_overlap,
                                schedule_double_buffer, full_token_cycles)

    def tok_per_s(cycles):
        return 1e9 / cycles

    fill = hbm_read_cycles(WEIGHT_LAYER)
    R = S + W

    def kv_row(data_per_tok, scale_per_tok, label_bytes, extra=None):
        d = _KVInt8Demands(R, data_per_tok * R, scale_per_tok * R)
        row = {
            "R_tokens": R,
            "kv_bytes_per_token": label_bytes,
            "overlap_tok_s": round(tok_per_s(full_token_cycles(d, schedule_overlap(d))), 1),
            "double_buffer_tok_s": round(
                tok_per_s(full_token_cycles(d, schedule_double_buffer(d), fill)), 1),
        }
        if extra:
            row.update(extra)
        return row

    out = {"S": S, "W": W, "R_tokens": R, "baseline": {}, "windowed_bf16": {},
           "windowed_int8kv_q05": {}, "windowed_int8kv_scale32": {},
           "windowed_vonly": {}, "windowed_kchan_dynamic": {}}

    for ctx in seq_lens:
        d = decode_demands(ctx)
        out["baseline"][str(ctx)] = {
            "R_tokens": ctx,
            "overlap_tok_s": round(tok_per_s(full_token_cycles(d, schedule_overlap(d))), 1),
            "double_buffer_tok_s": round(
                tok_per_s(full_token_cycles(d, schedule_double_buffer(d), fill)), 1),
        }

    d = decode_demands(R)
    out["windowed_bf16"] = {
        "R_tokens": R,
        "overlap_tok_s": round(tok_per_s(full_token_cycles(d, schedule_overlap(d))), 1),
        "double_buffer_tok_s": round(
            tok_per_s(full_token_cycles(d, schedule_double_buffer(d), fill)), 1),
    }

    out["windowed_int8kv_q05"] = kv_row(
        KV_DATA_INT8_PER_TOKEN, 0, KV_DATA_INT8_PER_TOKEN,
        {"kv_data_bytes_per_layer": KV_DATA_INT8_PER_TOKEN * R})
    out["windowed_int8kv_scale32"] = kv_row(
        KV_DATA_INT8_PER_TOKEN, KV_SCALE_PER_TOKEN,
        KV_DATA_INT8_PER_TOKEN + KV_SCALE_PER_TOKEN,
        {"kv_data_bytes_per_layer": KV_DATA_INT8_PER_TOKEN * R,
         "kv_scale_bytes_per_layer": KV_SCALE_PER_TOKEN * R})
    out["windowed_vonly"] = kv_row(
        VONLY_KV_DATA, VONLY_KV_SCALE, VONLY_KV_BYTES,
        {"kv_data_bytes_per_layer": VONLY_KV_DATA * R,
         "kv_scale_bytes_per_layer": VONLY_KV_SCALE * R})
    out["windowed_kchan_dynamic"] = kv_row(
        KCHAN_DYN_DATA, KCHAN_DYN_SCALE, KCHAN_DYN_KV_BYTES,
        {"kv_data_bytes_per_layer": KCHAN_DYN_DATA * R,
         "kv_scale_bytes_per_layer": KCHAN_DYN_SCALE * R})
    return out


def _print_ceiling(ceiling: dict) -> None:
    """Print the §6 ceiling table (six row-groups)."""
    for ctx, r in ceiling["baseline"].items():
        print(f"  baseline @{ctx}: overlap {r['overlap_tok_s']} | "
              f"double-buffer {r['double_buffer_tok_s']}", flush=True)
    r = ceiling["windowed_bf16"]
    print(f"  windowed BF16 (R={r['R_tokens']}): overlap {r['overlap_tok_s']} | "
          f"db {r['double_buffer_tok_s']}", flush=True)
    r = ceiling["windowed_int8kv_q05"]
    print(f"  windowed INT8-KV q=0.5 (R={r['R_tokens']}): overlap {r['overlap_tok_s']} | "
          f"db {r['double_buffer_tok_s']}", flush=True)
    r = ceiling["windowed_int8kv_scale32"]
    print(f"  windowed INT8-KV +scale (R={r['R_tokens']}): overlap {r['overlap_tok_s']} | "
          f"db {r['double_buffer_tok_s']}", flush=True)
    r = ceiling["windowed_vonly"]
    print(f"  V-only ({r['kv_bytes_per_token']} B/tok, R={r['R_tokens']}): "
          f"overlap {r['overlap_tok_s']} | db {r['double_buffer_tok_s']}", flush=True)
    r = ceiling["windowed_kchan_dynamic"]
    print(f"  per-channel-K dynamic ({r['kv_bytes_per_token']} B/tok, R={r['R_tokens']}): "
          f"overlap {r['overlap_tok_s']} | db {r['double_buffer_tok_s']}", flush=True)


# =============================================================================
# Driver
# =============================================================================

def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--sinks", type=int, default=S_DEFAULT)
    ap.add_argument("--window", type=int, default=W_DEFAULT)
    ap.add_argument("--seq-lens", default="4096,8192")
    ap.add_argument("--k-group", type=int, default=128,
                    help="K quantization group over head_dim (0=BF16, 1=per-dim, 128=per-head)")
    ap.add_argument("--v-group", type=int, default=128,
                    help="V quantization group over head_dim (0=BF16, 1=per-dim, 128=per-head)")
    ap.add_argument("--cross-tokens", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="output JSON; default /tmp/o2-kv-int8.json, or "
                         + STATIC_RESULTS + " with --static-kchan")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--ceiling-only", action="store_true",
                    help="skip model/PPL/cross; emit only the §6 qsim ceiling table")
    ap.add_argument("--static-kchan", action="store_true",
                    help="repro the static per-channel-K 66.9%% experiment (report §2 last row)")
    ap.add_argument("--static-calib-tokens", type=int, default=STATIC_CALIB_TOKENS,
                    help="static K scale calibrated from the first N positions of each span")
    ap.add_argument("--static-spans", type=int, default=STATIC_SPANS,
                    help="number of 4K spans to score for --static-kchan")
    args = ap.parse_args()

    S = args.sinks
    W = args.window
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    device = args.device
    k_group = args.k_group
    v_group = args.v_group
    out_path = args.out or (STATIC_RESULTS if args.static_kchan else "/tmp/o2-kv-int8.json")

    if args.ceiling_only:
        result = {
            "meta": {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": "Qwen/Qwen3-0.6B",
                "dtype": "bfloat16",
                "scheme": "KV INT8 symmetric per-group + BF16 scale metadata",
                "sinks": S, "window": W, "seq_lens": seq_lens,
                "ceiling_only": True,
            },
            "ceiling": ceiling_table(S, W, seq_lens),
        }
        print("=== qsim decode ceiling (q=0.5, W=%d) ===" % W, flush=True)
        _print_ceiling(result["ceiling"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_path}", flush=True)
        return result

    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available; pass --device cpu")

    print(f"=== load model + reference ({args.model_dir}) ===", flush=True)
    t0 = time.time()
    tok, hf, ref = load_hf(args.model_dir, device)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    with open(args.baseline) as f:
        baseline = json.load(f)

    result = {
        "meta": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "Qwen/Qwen3-0.6B",
            "dtype": "bfloat16",
            "scheme": "KV INT8 symmetric per-group + BF16 scale metadata",
            "sinks": S, "window": W,
            "k_group": k_group, "v_group": v_group,
            "static_kchan": bool(args.static_kchan),
            "seq_lens": seq_lens,
            "baseline": args.baseline,
            "delta_ppl_gate": O2_DELTA_PPL_GATE,
            "cross_gate": f">={CROSS_GATE_10}/10",
            "device": device,
        },
    }

    # ---- sanity: full-attention forward reproduces the reference greedy ----
    print(f"=== sanity: full-attention forward vs Qwen3Ref (5 tokens) ===", flush=True)
    pid = build_seq(tok, 64)
    from qrun.reference import ref_greedy_with_logits
    r_tokens, _ = ref_greedy_with_logits(ref, torch.from_numpy(pid), 5)
    m_tokens, _ = ref_greedy(ref, torch.from_numpy(pid), 5)
    result["sanity"] = {
        "ref_greedy_tokens": [int(x) for x in r_tokens],
        "my_forward_tokens": [int(x) for x in m_tokens],
        "match": [int(a == b) for a, b in zip(r_tokens, m_tokens)],
    }
    print(f"  ref vs mine: {result['sanity']['match']}", flush=True)

    # ---- ΔPPL gate ----
    if not args.skip_ppl:
        print(f"=== ΔPPL (windowed BF16 & windowed+INT8-KV vs full, S={S}, W={W}, "
              f"k={k_group}, v={v_group}) ===", flush=True)
        result["ppl"] = evaluate_ppl_o2(ref, baseline["ppl"], seq_lens, S, W,
                                        device, k_group, v_group)
        for key, r in result["ppl"].items():
            for mode in ("windowed_bf16", "windowed_quant"):
                m = r[mode]
                print(f"  L={key} {mode}: pooled ΔPPL={m['pooled_delta_ppl']*100:.3f}% "
                      f"(PPL {m['pooled_ppl_full']:.3f} -> {m['pooled_ppl_windowed']:.3f}) "
                      f"{'PASS' if m['pass'] else 'FAIL'}", flush=True)

    # ---- 20-token cross-consistency ----
    if not args.skip_cross:
        print(f"=== 20-token cross-consistency (S={S}, W={W}, k={k_group}, v={v_group}) ===",
              flush=True)
        result["cross"] = cross_consistency(ref, tok, 4096, args.cross_tokens, S, W,
                                            device, k_group, v_group)
        c = result["cross"]["windowed_quant_vs_full"]
        print(f"  INT8-KV vs full: {c['n_match']}/{args.cross_tokens} "
              f"{'PASS' if c['pass'] else 'FAIL'}; div at {c['divergence_positions']}",
              flush=True)
        w_ = result["cross"]["windowed_bf16_vs_full"]
        print(f"  windowed BF16 vs full: {w_['n_match']}/{args.cross_tokens} "
              f"div at {w_['divergence_positions']}", flush=True)

    # ---- static per-channel K (report §2 last row) ----
    if args.static_kchan:
        print(f"=== static per-channel-K (calib {args.static_calib_tokens} tok, "
              f"{args.static_spans} spans @4K, V per-head) ===", flush=True)
        result["static_kchan"] = evaluate_static_kchan(
            ref, baseline["ppl"], S, W, device, v_group=128,
            calib_tokens=args.static_calib_tokens, n_spans=args.static_spans)
        r = result["static_kchan"]
        print(f"  pooled ΔPPL={r['pooled_delta_ppl']*100:.3f}% "
              f"(PPL {r['pooled_ppl_full']:.3f} -> {r['pooled_ppl_static']:.3f}) "
              f"{'PASS' if r['pass'] else 'FAIL'}", flush=True)

    # ---- ceiling ----
    print(f"=== qsim decode ceiling (q=0.5) ===", flush=True)
    result["ceiling"] = ceiling_table(S, W, seq_lens)
    _print_ceiling(result["ceiling"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
