"""B-Fold: QK-norm fold + INT4 KV numerical verification (signed scale).

Implements DECISION §3.2 / kv-outlier.md §3.3 on the torch ref (no RTL):

  Qwen3 K = k_norm ⊙ rmsnorm(k_raw).  Fold the STATIC per-channel k_norm into
  the dequant scale so the quantized object becomes the UNIT-RMS pre-weight
  vector  k_unit = rmsnorm(k_raw)  (|k_unit| bounded by sqrt(128) ~ 11.3, no
  channel outliers).  K is quantized to INT4 with a per-head-per-token
  symmetric scale s_q = max|k_unit|/7; the dequant scale is the SIGNED
  scale_c = k_norm[c] * s_q  (k_norm has ~3.4% negative channels, min -3.86).
  RoPE is applied ON-READ (post dequant) — the pre-RoPE storage requirement.
  V is quantized per-token per-head symmetric INT4 (no QK-norm; ~2.4x clean).

Gates (DECISION §3.2):
  - ΔPPL <= 2% (relative, windowed + INT4-fold-KV vs golden full attention).
  - 20-token cross-consistency >= 8/10 (ISOLATED: quant vs windowed-BF16,
    per O2 §9.1 — windowing itself contributes divergence vs full).
The 1.27x ceiling is conditional on a dedicated on-read RoPE rotator; this run
is numerical-only and reports that condition explicitly (no hardware).

--k-bits {4,8} lets us also score the INT8 fold (validation anchor: should
reproduce O2's per-channel INT8 K ≈ 0.7%/1.8%).

Usage:
  python3 qrun/fold_verify.py --ceiling-only            # fast, no model
  python3 qrun/fold_verify.py --baseline ... --out ...   # full ΔPPL + cross
  python3 qrun/fold_verify.py --k-bits 8 ...             # INT8-fold anchor
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

from qrun.o2_kv_int8 import (streaming_mask, windowed_row_mask,  # noqa: E402
                             build_seq, FIXED_TEXT, O2_DELTA_PPL_GATE)
from qrun.reference import load_hf, ref_greedy_with_logits  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
BASELINE = "docs/perf-research/quality-baseline/quality_baseline.json"

S_DEFAULT = 4
W_DEFAULT = 2048

INT4_LEVEL = 7.0            # symmetric INT4: [-7, 7] (zero representable, -8 reserved)
CROSS_GATE_10 = 8           # >= 8/10; 20 tokens -> >= 16/20

# INT4 KV per-token-per-layer byte budgets (Qwen3-0.6B, 8 KV heads):
KV_DATA_INT4_PER_TOKEN = 1024      # 8 heads x 2 (K,V) x 128 x 0.5 B
KV_SCALE_INT4_PER_TOKEN = 32       # K per-head s_q (16) + V per-head (16), BF16
KV_SCALE_INT4_FOLDED = 16          # V per-head only; K scale folded into static k_norm
K_NORM_STATIC_BYTES = 8 * 128 * 2  # 2048 B/layer SRAM-resident (shared across sequence)

# INT8-fold KV per-token-per-layer byte budgets (Qwen3-0.6B, 8 KV heads):
KV_DATA_INT8K_INT4V_PER_TOKEN = 1536   # K INT8 (8x128x1) + V INT4 (8x128x0.5)
KV_DATA_INT8KV_PER_TOKEN = 2048        # K,V INT8 (8 heads x 2 x 128 x 1 B)


# =============================================================================
# Symmetric per-group INT{b} quantize/dequantize (over head_dim)
# =============================================================================

def quantize_dequant_int(x: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Symmetric per-group INT{bits} roundtrip of x: [heads, seq, dim] bf16.

    level = 2^(bits-1) - 1; scale = bf16(max|x_group| / level) per
    (head, token, group); q = round(x/scale) clipped to [-level, level];
    returns dequantized x ~= q * scale in x.dtype.  group=128 -> per-head;
    group=1 -> per-channel."""
    level = float(2 ** (bits - 1) - 1)
    xf = x.float()
    heads, seq, dim = x.shape
    G = dim // group
    xg = xf.reshape(heads, seq, G, group)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / level).clamp(min=1e-6)
    scale_bf16 = scale.to(torch.bfloat16).to(torch.float32)
    q = (xg / scale_bf16).round().clamp(-level, level)
    dq = q * scale_bf16
    return dq.reshape(heads, seq, dim).to(x.dtype)


def quantize_dequant_int4(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Symmetric per-group INT4 roundtrip (level 7)."""
    return quantize_dequant_int(x, 4, group)


# =============================================================================
# QK-norm fold: quantize the unit-norm pre-weight K, fold k_norm into a SIGNED
# per-channel dequant scale.  Returns (k_hat pre-RoPE, s_q, scale_c).
# =============================================================================

def fold_quantize_dequant(k_unit: torch.Tensor, k_norm: torch.Tensor, bits: int = 4):
    """k_unit: [heads, seq, dim] bf16 (unit RMS over dim).  k_norm: [dim] bf16.

    k_hat = round(k_unit / s_q) * (k_norm[c] * s_q), s_q = per-head-per-token
    max|k_unit|/(2^(bits-1)-1).  scale_c = k_norm[c] * s_q is SIGNED (k_norm can
    be negative).  All metadata rounded to BF16 (hardware-faithful).  k_hat is
    pre-RoPE (RoPE applied on-read)."""
    level = float(2 ** (bits - 1) - 1)
    kf = k_unit.float()
    amax = kf.abs().amax(dim=-1, keepdim=True)                 # [heads, seq, 1]
    s_q = (amax / level).clamp(min=1e-6)
    s_q_bf16 = s_q.to(torch.bfloat16).to(torch.float32)        # stored per-head scale
    q = (kf / s_q_bf16).round().clamp(-level, level)           # INT{bits}
    kn = k_norm.float().view(1, 1, -1)                         # [1, 1, dim]
    scale_c = (s_q_bf16 * kn).to(torch.bfloat16).to(torch.float32)  # SIGNED BF16
    k_hat = (q * scale_c).to(k_unit.dtype)
    return k_hat, s_q_bf16, scale_c


# =============================================================================
# Fold forward: full-span (prefill) + single decode step with on-read RoPE.
# fold_k=False -> BF16 K reference (QK-norm applied, no quant, RoPE on-read).
# =============================================================================

@torch.no_grad()
def forward_span_fold(ref, token_ids, S=None, W=None, fold_k=True, v_group=128, k_bits=4):
    seq = token_ids.shape[0]
    device = ref.device
    positions = torch.arange(0, seq, device=device)
    cos, sin = rope_embeddings(positions, ref.inv_freq.to(device), torch.bfloat16)
    hidden = ref.embed[token_ids.to(device)]
    if S is None:
        mask = causal_mask(seq, seq, 0, torch.bfloat16, device)
    else:
        mask = streaming_mask(seq, S, W, device)
    ones = torch.ones(HEAD_DIM, device=device, dtype=torch.bfloat16)
    new_cache = []
    for lw in ref.layers:
        norm_in = rmsnorm(hidden, lw.in_norm)
        q_raw = F.linear(norm_in, lw.q_proj)
        k_raw = F.linear(norm_in, lw.k_proj)
        v_raw = F.linear(norm_in, lw.v_proj)
        q = rmsnorm(q_raw.view(seq, N_HEADS, HEAD_DIM), lw.q_norm).transpose(0, 1)
        k_unit = rmsnorm(k_raw.view(seq, N_KV_HEADS, HEAD_DIM), ones).transpose(0, 1)
        v = v_raw.view(seq, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        q_rot = apply_rope(q, cos, sin)
        if fold_k:
            k_hat, _, _ = fold_quantize_dequant(k_unit, lw.k_norm, k_bits)
        else:
            k_hat = rmsnorm(k_raw.view(seq, N_KV_HEADS, HEAD_DIM), lw.k_norm).transpose(0, 1)
        k_rot = apply_rope(k_hat, cos, sin)                     # on-read RoPE
        v_hat = quantize_dequant_int4(v, v_group) if v_group else v
        k_rep = repeat_kv(k_rot, N_REP)
        v_rep = repeat_kv(v_hat, N_REP)
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
        new_cache.append({"k": k_hat, "v": v_hat})              # K cached PRE-RoPE
    final = rmsnorm(hidden, ref.final_norm)
    logits = F.linear(final, ref.lm_head)
    return logits, new_cache


@torch.no_grad()
def decode_step_fold(ref, token_id, pos, cache, S=None, W=None, fold_k=True, v_group=128,
                     k_bits=4):
    device = ref.device
    token = torch.tensor([token_id], device=device)
    positions_all = torch.arange(0, pos + 1, device=device)
    cos_all, sin_all = rope_embeddings(positions_all, ref.inv_freq.to(device), torch.bfloat16)
    cos_q, sin_q = cos_all[pos:pos + 1], sin_all[pos:pos + 1]
    hidden = ref.embed[token]
    ones = torch.ones(HEAD_DIM, device=device, dtype=torch.bfloat16)
    new_cache = []
    for i, lw in enumerate(ref.layers):
        norm_in = rmsnorm(hidden, lw.in_norm)
        q_raw = F.linear(norm_in, lw.q_proj)
        k_raw = F.linear(norm_in, lw.k_proj)
        v_raw = F.linear(norm_in, lw.v_proj)
        q = rmsnorm(q_raw.view(1, N_HEADS, HEAD_DIM), lw.q_norm).transpose(0, 1)
        k_unit = rmsnorm(k_raw.view(1, N_KV_HEADS, HEAD_DIM), ones).transpose(0, 1)
        v = v_raw.view(1, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        q_rot = apply_rope(q, cos_q, sin_q)
        if fold_k:
            k_new, _, _ = fold_quantize_dequant(k_unit, lw.k_norm, k_bits)
        else:
            k_new = rmsnorm(k_raw.view(1, N_KV_HEADS, HEAD_DIM), lw.k_norm).transpose(0, 1)
        v_new = quantize_dequant_int4(v, v_group) if v_group else v
        k_full = torch.cat([cache[i]["k"], k_new], dim=1)       # pre-RoPE [8, pos+1, 128]
        v_full = torch.cat([cache[i]["v"], v_new], dim=1)
        k_rot = apply_rope(k_full, cos_all, sin_all)            # on-read RoPE, ALL positions
        k_rep = repeat_kv(k_rot, N_REP)
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
def ref_greedy_fold(ref, prompt_ids, n, S=None, W=None, fold_k=True, v_group=128, k_bits=4):
    logits, cache = forward_span_fold(ref, prompt_ids, S, W, fold_k, v_group, k_bits)
    new_tokens = []
    logits_list = [logits[-1]]
    for _ in range(n):
        tok = int(logits[-1].argmax(dim=-1).item())
        new_tokens.append(tok)
        logits, cache = decode_step_fold(ref, tok, prompt_ids.shape[0] + len(new_tokens) - 1,
                                         cache, S, W, fold_k, v_group, k_bits)
        logits_list.append(logits[-1])
    return new_tokens, logits_list


@torch.no_grad()
def span_nll_fold(ref, token_ids, S, W, fold_k, v_group, k_bits=4) -> torch.Tensor:
    logits, _ = forward_span_fold(ref, token_ids, S, W, fold_k, v_group, k_bits)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    targets = token_ids[1:]
    nll = -log_probs[:-1, :].gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return nll


# =============================================================================
# Signed-scale stats: static k_norm + folded scale_c over sampled activations
# =============================================================================

def k_norm_static_stats(ref) -> dict:
    kn = torch.stack([lw.k_norm.float() for lw in ref.layers])   # [28, 128]
    neg = int((kn < 0).sum().item())
    return {
        "n_channels": int(kn.numel()),
        "neg_channels": neg,
        "neg_frac": neg / int(kn.numel()),
        "min": float(kn.min().item()),
        "max": float(kn.max().item()),
        "abs_min": float(kn.abs().min().item()),
        "abs_max": float(kn.abs().max().item()),
    }


def folded_scale_stats(ref, token_ids, S, W, device, k_bits=4) -> dict:
    """Fold-scale stats over the first (prefill) forward of a sample span:
    s_q range and SIGNED scale_c = k_norm[c] * s_q (neg fraction / min)."""
    seq = token_ids.shape[0]
    positions = torch.arange(0, seq, device=device)
    cos, sin = rope_embeddings(positions, ref.inv_freq.to(device), torch.bfloat16)
    hidden = ref.embed[token_ids.to(device)]
    ones = torch.ones(HEAD_DIM, device=device, dtype=torch.bfloat16)
    all_sq = []
    all_scale = []
    for lw in ref.layers:
        norm_in = rmsnorm(hidden, lw.in_norm)
        k_raw = F.linear(norm_in, lw.k_proj)
        k_unit = rmsnorm(k_raw.view(seq, N_KV_HEADS, HEAD_DIM), ones).transpose(0, 1)
        _, s_q, scale_c = fold_quantize_dequant(k_unit, lw.k_norm, k_bits)
        all_sq.append(s_q.reshape(-1))
        all_scale.append(scale_c.reshape(-1))
        # advance hidden through the layer (cheap path, quantized KV used)
        q_raw = F.linear(norm_in, lw.q_proj)
        v_raw = F.linear(norm_in, lw.v_proj)
        q = rmsnorm(q_raw.view(seq, N_HEADS, HEAD_DIM), lw.q_norm).transpose(0, 1)
        v = v_raw.view(seq, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        q_rot = apply_rope(q, cos, sin)
        k_hat, _, _ = fold_quantize_dequant(k_unit, lw.k_norm, k_bits)
        k_rot = apply_rope(k_hat, cos, sin)
        v_hat = quantize_dequant_int4(v, 128)
        k_rep = repeat_kv(k_rot, N_REP)
        v_rep = repeat_kv(v_hat, N_REP)
        scores = torch.matmul(q_rot, k_rep.transpose(-1, -2)) * ATTN_SCALE
        mask = streaming_mask(seq, S, W, device)
        probs = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q_rot.dtype)
        ctx = torch.matmul(probs, v_rep)
        attn_out = ctx.transpose(0, 1).reshape(seq, N_HEADS * HEAD_DIM).contiguous()
        o = F.linear(attn_out, lw.o_proj)
        hidden = hidden + o
        norm_post = rmsnorm(hidden, lw.post_norm)
        gate = F.linear(norm_post, lw.gate)
        up = F.linear(norm_post, lw.up)
        act = F.silu(gate) * up
        down = F.linear(act, lw.down)
        hidden = hidden + down
    sq = torch.cat(all_sq)
    sc = torch.cat(all_scale)
    return {
        "s_q_min": float(sq.min().item()),
        "s_q_max": float(sq.max().item()),
        "s_q_mean": float(sq.mean().item()),
        "scale_c_neg_frac": float((sc < 0).float().mean().item()),
        "scale_c_min": float(sc.min().item()),
        "scale_c_max": float(sc.max().item()),
    }


# =============================================================================
# ΔPPL gate (windowed BF16 & windowed+fold-KV vs golden full)
# =============================================================================

def evaluate_ppl_fold(ref, baseline_ppl, seq_lens, S, W, device, v_group=128, k_bits=4):
    out = {}
    for L in seq_lens:
        key = str(L)
        spans = baseline_ppl[key]
        L = int(L)
        for mode in ("windowed_bf16", "windowed_fold"):
            fold_k, vg = (False, 0) if mode == "windowed_bf16" else (True, v_group)
            rows = []
            all_dnll = []
            all_w = []
            all_f = []
            for s in spans:
                ids = torch.tensor(s["token_ids"], dtype=torch.long, device=device)
                wnll = span_nll_fold(ref, ids, S, W, fold_k, vg, k_bits)
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


# =============================================================================
# 20-token cross-consistency (full / windowed-BF16 / windowed-fold)
# =============================================================================

def cross_consistency_fold(ref, tok, prompt_len, n_tokens, S, W, device, v_group=128,
                           k_bits=4):
    pid = build_seq(tok, prompt_len)
    pid_t = torch.from_numpy(pid)

    print(f"  full BF16 attention baseline ({n_tokens} tokens) ...", flush=True)
    full_tokens, full_logits = ref_greedy_with_logits(ref, pid_t, n_tokens)
    print(f"  windowed BF16 (W={W}) ...", flush=True)
    win_tokens, win_logits = ref_greedy_fold(ref, pid_t, n_tokens, S=S, W=W,
                                             fold_k=False, v_group=0)
    print(f"  windowed + fold-KV (W={W}, K fold {k_bits}b, V per-head) ...", flush=True)
    q_tokens, q_logits = ref_greedy_fold(ref, pid_t, n_tokens, S=S, W=W,
                                         fold_k=True, v_group=v_group, k_bits=k_bits)

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
    per_qw, div_qw = _rel_err(q_logits, win_logits, q_tokens, win_tokens)

    iso_match = _match(q_tokens, win_tokens)
    return {
        "prompt_len": prompt_len,
        "n_tokens": n_tokens,
        "full_tokens": [int(x) for x in full_tokens],
        "windowed_bf16_tokens": [int(x) for x in win_tokens],
        "windowed_fold_tokens": [int(x) for x in q_tokens],
        "windowed_bf16_vs_full": {
            "match": _match(win_tokens, full_tokens),
            "n_match": sum(_match(win_tokens, full_tokens)),
            "divergence_positions": [i for i in range(n_tokens)
                                     if win_tokens[i] != full_tokens[i]],
            "per_pos_rel_err": per_w,
            "divergence_rel_err": div_w,
        },
        "windowed_fold_vs_full": {
            "match": _match(q_tokens, full_tokens),
            "n_match": sum(_match(q_tokens, full_tokens)),
            "divergence_positions": [i for i in range(n_tokens)
                                     if q_tokens[i] != full_tokens[i]],
            "per_pos_rel_err": per_q,
            "divergence_rel_err": div_q,
        },
        # ISOLATED gate: quant vs windowed-BF16 (windowing error removed)
        "fold_vs_windowed_bf16": {
            "match": iso_match,
            "n_match": sum(iso_match),
            "divergence_positions": [i for i in range(n_tokens) if iso_match[i] == 0],
            "per_pos_rel_err": per_qw,
            "divergence_rel_err": div_qw,
            "pass": sum(iso_match) >= (CROSS_GATE_10 * n_tokens // 10),
        },
        "cross_gate_10": CROSS_GATE_10,
    }


# =============================================================================
# qsim decode ceiling for INT4 KV (data 1024 B/token + scale metadata)
# =============================================================================

class _KVInt4Demands:
    def __init__(self, R, kv_data_bytes, kv_scale_bytes):
        from math import ceil
        from qsim.timing import hbm_read_cycles
        from qsim.timing_p6 import WEIGHT_LAYER
        self.ctx = R
        self.weight_read = hbm_read_cycles(WEIGHT_LAYER)
        self.kv_read = hbm_read_cycles(kv_data_bytes + kv_scale_bytes)
        # Retired: staged KV.LOAD SRAM write.  Replaced by the streaming B-feed
        # array occupancy (rotator-impl plan v3): K rotate R*q_out/256 +
        # V dequant R*kv_out/256 + dense MAC 480, overlapped with the HBM read
        # (26,317 cyc/层 for R=2052) -> 25,104 <= 26,317, HBM remains the bound.
        self.kv_sram_write = ceil(R * 2048 / 256) + ceil(R * 1024 / 256) + 480
        self.b_feed = self.kv_sram_write


def ceiling_int4(S, W, seq_lens=(4096, 8192)):
    from qsim.timing_p6 import (WEIGHT_LAYER, decode_demands, schedule_overlap,
                                schedule_double_buffer, full_token_cycles)
    from qsim.timing import hbm_read_cycles

    tok_per_s = lambda c: 1e9 / c
    R = S + W
    fill = hbm_read_cycles(WEIGHT_LAYER)

    def kv_row(data_per_tok, scale_per_tok, label_bytes):
        d = _KVInt4Demands(R, data_per_tok * R, scale_per_tok * R)
        return {
            "R_tokens": R,
            "kv_bytes_per_token": label_bytes,
            "overlap_tok_s": round(tok_per_s(full_token_cycles(d, schedule_overlap(d))), 1),
            "double_buffer_tok_s": round(
                tok_per_s(full_token_cycles(d, schedule_double_buffer(d), fill)), 1),
        }

    out = {"S": S, "W": W, "R_tokens": R, "baseline": {}, "windowed_bf16": {},
           "int4_scale0": {}, "int4_scale16_folded": {}, "int4_scale32": {},
           "int8_k_fold_int4_v": {}, "int8_kv_fold": {}}
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
    out["int4_scale0"] = kv_row(KV_DATA_INT4_PER_TOKEN, 0, KV_DATA_INT4_PER_TOKEN)
    out["int4_scale16_folded"] = kv_row(KV_DATA_INT4_PER_TOKEN, KV_SCALE_INT4_FOLDED,
                                        KV_DATA_INT4_PER_TOKEN + KV_SCALE_INT4_FOLDED)
    out["int4_scale32"] = kv_row(KV_DATA_INT4_PER_TOKEN, KV_SCALE_INT4_PER_TOKEN,
                                 KV_DATA_INT4_PER_TOKEN + KV_SCALE_INT4_PER_TOKEN)
    out["int8_k_fold_int4_v"] = kv_row(KV_DATA_INT8K_INT4V_PER_TOKEN,
                                       KV_SCALE_INT4_PER_TOKEN,
                                       KV_DATA_INT8K_INT4V_PER_TOKEN + KV_SCALE_INT4_PER_TOKEN)
    out["int8_kv_fold"] = kv_row(KV_DATA_INT8KV_PER_TOKEN, KV_SCALE_INT4_PER_TOKEN,
                                 KV_DATA_INT8KV_PER_TOKEN + KV_SCALE_INT4_PER_TOKEN)
    out["k_norm_static_bytes_per_layer"] = K_NORM_STATIC_BYTES
    return out


def _print_ceiling_int4(ceiling: dict) -> None:
    for ctx, r in ceiling["baseline"].items():
        print(f"  baseline @{ctx}: overlap {r['overlap_tok_s']} | db {r['double_buffer_tok_s']}",
              flush=True)
    r = ceiling["windowed_bf16"]
    print(f"  windowed BF16 (R={r['R_tokens']}): overlap {r['overlap_tok_s']} | "
          f"db {r['double_buffer_tok_s']}", flush=True)
    for k in ("int4_scale0", "int4_scale16_folded", "int4_scale32",
              "int8_k_fold_int4_v", "int8_kv_fold"):
        r = ceiling[k]
        print(f"  {k} ({r['kv_bytes_per_token']} B/tok, R={r['R_tokens']}): "
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
    ap.add_argument("--v-group", type=int, default=128)
    ap.add_argument("--k-bits", type=int, default=4, choices=(4, 8),
                    help="K fold quantization bits (4 = INT4 primary, 8 = INT8 anchor)")
    ap.add_argument("--cross-tokens", type=int, default=20)
    ap.add_argument("--out", default="docs/perf-research/fold-verify-results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--ceiling-only", action="store_true")
    args = ap.parse_args()

    S = args.sinks
    W = args.window
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    device = args.device
    out_path = args.out

    if args.ceiling_only:
        result = {
            "meta": {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": "Qwen/Qwen3-0.6B",
                "dtype": "bfloat16",
                "scheme": "INT4 KV + QK-norm fold (signed scale), on-read RoPE",
                "sinks": S, "window": W, "seq_lens": seq_lens, "ceiling_only": True,
            },
            "ceiling": ceiling_int4(S, W, seq_lens),
        }
        print("=== qsim decode ceiling (INT4 KV, W=%d) ===" % W, flush=True)
        _print_ceiling_int4(result["ceiling"])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
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
            "scheme": "INT4 KV + QK-norm fold (signed scale_c = k_norm[c]*s_q), "
                      "on-read RoPE; V per-token per-head INT4",
            "sinks": S, "window": W, "v_group": args.v_group, "k_bits": args.k_bits,
            "seq_lens": seq_lens,
            "baseline": args.baseline,
            "delta_ppl_gate": O2_DELTA_PPL_GATE,
            "cross_gate": f">={CROSS_GATE_10}/10 (isolated fold vs windowed-BF16)",
            "device": device,
            "ropex_condition": (
                "1.27x ceiling conditional on dedicated on-read RoPE rotator; "
                "without rotator ~4x rotation cost (numerical-only run, no hardware)"),
        },
    }

    # ---- sanity: full-attention forward reproduces reference greedy ----
    print(f"=== sanity: full-attention forward vs Qwen3Ref (5 tokens) ===", flush=True)
    pid = build_seq(tok, 64)
    r_tokens, _ = ref_greedy_with_logits(ref, torch.from_numpy(pid), 5)
    m_tokens, _ = ref_greedy_fold(ref, torch.from_numpy(pid), 5, fold_k=False, v_group=0)
    result["sanity"] = {
        "ref_greedy_tokens": [int(x) for x in r_tokens],
        "my_forward_tokens": [int(x) for x in m_tokens],
        "match": [int(a == b) for a, b in zip(r_tokens, m_tokens)],
    }
    print(f"  ref vs mine: {result['sanity']['match']}", flush=True)

    # ---- signed-scale stats ----
    print("=== signed-scale stats (k_norm static + folded scale_c) ===", flush=True)
    result["k_norm_static"] = k_norm_static_stats(ref)
    print(f"  k_norm static: neg {result['k_norm_static']['neg_frac']*100:.3f}% "
          f"({result['k_norm_static']['neg_channels']}/{result['k_norm_static']['n_channels']}) "
          f"min {result['k_norm_static']['min']:.4f}", flush=True)
    sid0 = torch.tensor(baseline["ppl"]["4096"][0]["token_ids"], dtype=torch.long, device=device)
    result["folded_scale"] = folded_scale_stats(ref, sid0, S, W, device, args.k_bits)
    print(f"  folded scale_c: neg {result['folded_scale']['scale_c_neg_frac']*100:.3f}% "
          f"min {result['folded_scale']['scale_c_min']:.4f} "
          f"s_q[{result['folded_scale']['s_q_min']:.4f},{result['folded_scale']['s_q_max']:.4f}]",
          flush=True)

    # ---- ΔPPL gate ----
    if not args.skip_ppl:
        print(f"=== ΔPPL (windowed BF16 & windowed+fold-KV vs full, S={S}, W={W}, "
              f"K={args.k_bits}b) ===", flush=True)
        result["ppl"] = evaluate_ppl_fold(ref, baseline["ppl"], seq_lens, S, W,
                                          device, args.v_group, args.k_bits)
        for key, r in result["ppl"].items():
            for mode in ("windowed_bf16", "windowed_fold"):
                m = r[mode]
                print(f"  L={key} {mode}: pooled ΔPPL={m['pooled_delta_ppl']*100:.3f}% "
                      f"(PPL {m['pooled_ppl_full']:.3f} -> {m['pooled_ppl_windowed']:.3f}) "
                      f"{'PASS' if m['pass'] else 'FAIL'}", flush=True)

    # ---- 20-token cross-consistency ----
    if not args.skip_cross:
        print(f"=== 20-token cross-consistency (S={S}, W={W}, fold {args.k_bits}b) ===",
              flush=True)
        result["cross"] = cross_consistency_fold(ref, tok, 4096, args.cross_tokens, S, W,
                                                 device, args.v_group, args.k_bits)
        c = result["cross"]["fold_vs_windowed_bf16"]
        print(f"  fold-KV vs windowed-BF16 (ISOLATED): {c['n_match']}/{args.cross_tokens} "
              f"{'PASS' if c['pass'] else 'FAIL'}; div at {c['divergence_positions']}", flush=True)
        f_ = result["cross"]["windowed_fold_vs_full"]
        print(f"  fold-KV vs full: {f_['n_match']}/{args.cross_tokens} "
              f"div at {f_['divergence_positions']}", flush=True)
        w_ = result["cross"]["windowed_bf16_vs_full"]
        print(f"  windowed BF16 vs full: {w_['n_match']}/{args.cross_tokens} "
              f"div at {w_['divergence_positions']}", flush=True)

    # ---- ceiling ----
    print(f"=== qsim decode ceiling (INT4 KV) ===", flush=True)
    result["ceiling"] = ceiling_int4(S, W, seq_lens)
    _print_ceiling_int4(result["ceiling"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
