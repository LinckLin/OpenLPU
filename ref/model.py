"""Qwen3-0.6B hand-written reference forward path (op-decomposed).

Mirrors transformers 4.51.0 `modeling_qwen3.py` dtype-for-dtype so that each op
matches HF eager attention bit-compatibly (same torch primitives on the same
bfloat16 tensors). Every op in the J1 enum is emitted as a named (input, output)
pair for golden-trace capture and op-by-op comparison.

Layout conventions (batch=1 dropped everywhere):
- hidden / x      : [seq, hidden=1024]
- q_raw           : [seq, 2048]   (q_proj out = 16 heads x 128)
- k_raw / v_raw   : [seq, 1024]   (8 kv heads x 128)
- q               : [16, seq, 128]  (after per-head QK-norm, heads-first)
- k / v           : [8, seq, 128]
- scores / probs  : [16, seq_q, seq_k]
- ctx             : [16, seq_q, 128]
- attn_o / hidden : [seq, 1024]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---- model card (verified against Qwen/Qwen3-0.6B config.json + safetensors) ----
HIDDEN = 1024
N_LAYERS = 28
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
N_REP = N_HEADS // N_KV_HEADS          # GQA 2:1
INTERMEDIATE = 3072
VOCAB = 151936
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6
ATTN_SCALE = HEAD_DIM ** -0.5          # 128^-0.5


# --------------------------------------------------------------------------- #
# elementwise / small ops
# --------------------------------------------------------------------------- #
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """Qwen3RMSNorm: fp32 normalize, multiply by (bf16) weight in input dtype."""
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return weight * xf.to(input_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
def rope_inv_freq(dim: int = HEAD_DIM, theta: float = ROPE_THETA, device=None) -> torch.Tensor:
    """default rope: inv_freq = 1/(theta^(arange(0,dim,2)/dim)), fp32."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq


def rope_embeddings(positions: torch.Tensor, inv_freq: torch.Tensor, dtype: torch.dtype) -> tuple:
    """positions: [seq] int64. Returns (cos, sin) each [seq, head_dim] in `dtype`."""
    freqs = positions[:, None].float() * inv_freq[None, :].float()     # [seq, 64]
    emb = torch.cat((freqs, freqs), dim=-1)                                           # [seq, 128]
    cos, sin = emb.cos(), emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [heads, seq, dim]; cos/sin: [seq, dim] (broadcast over heads)."""
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int = N_REP) -> torch.Tensor:
    """[kv_heads, seq, dim] -> [kv_heads*n_rep, seq, dim] (repeat_interleave)."""
    if n_rep == 1:
        return x
    kv, s, d = x.shape
    return x[:, None, :, :].expand(kv, n_rep, s, d).reshape(kv * n_rep, s, d)


# --------------------------------------------------------------------------- #
# causal mask (min_dtype semantics, matching HF eager path)
# --------------------------------------------------------------------------- #
def causal_mask(seq_q: int, seq_k: int, cache_pos: int, dtype: torch.dtype, device) -> torch.Tensor:
    """[1, seq_q, seq_k]: 0 allowed, finfo(dtype).min masked (future positions)."""
    min_dtype = torch.finfo(dtype).min
    qpos = torch.arange(cache_pos, cache_pos + seq_q, device=device)
    kpos = torch.arange(seq_k, device=device)
    attend = (kpos[None, :] <= qpos[:, None]).to(dtype)     # [seq_q, seq_k]: 1 allowed / 0 masked
    mask = torch.where(attend.bool(), torch.zeros_like(attend), torch.full_like(attend, min_dtype))
    return mask[None, :, :]                                 # [1, seq_q, seq_k]


# --------------------------------------------------------------------------- #
# layer forward (op-decomposed, trace-capable)
# --------------------------------------------------------------------------- #
class LayerWeights:
    """A single decoder layer's weights, views into the HF state dict."""

    __slots__ = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
                 "gate", "up", "down", "in_norm", "post_norm")

    def __init__(self, sd: dict, layer: int):
        p = f"model.layers.{layer}."
        self.q_proj = sd[p + "self_attn.q_proj.weight"]
        self.k_proj = sd[p + "self_attn.k_proj.weight"]
        self.v_proj = sd[p + "self_attn.v_proj.weight"]
        self.o_proj = sd[p + "self_attn.o_proj.weight"]
        self.q_norm = sd[p + "self_attn.q_norm.weight"]
        self.k_norm = sd[p + "self_attn.k_norm.weight"]
        self.gate = sd[p + "mlp.gate_proj.weight"]
        self.up = sd[p + "mlp.up_proj.weight"]
        self.down = sd[p + "mlp.down_proj.weight"]
        self.in_norm = sd[p + "input_layernorm.weight"]
        self.post_norm = sd[p + "post_attention_layernorm.weight"]


def layer_forward(
    w: LayerWeights,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: dict | None,
    trace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one decoder layer; return (hidden, k_full, v_full).

    hidden: [seq, HIDDEN]. cos/sin: [seq, HEAD_DIM]. mask: [1, seq, seq_k] (or None).
    kv_cache: dict {"k": [8, L, 128], "v": [8, L, 128]} for decode (prefill passes None).
    k_full/v_full: [8, L+seq, 128] KV after append (used for cache and trace).
    """
    seq = hidden.shape[0]

    def rec(op, inputs, outputs):
        if trace is not None:
            trace[op] = {"inputs": inputs, "outputs": outputs}

    # --- input RMSNorm ---
    norm_in = rmsnorm(hidden, w.in_norm)
    rec("rmsnorm_in", {"x": hidden}, {"y": norm_in})

    # --- QKV projection ---
    q_raw = F.linear(norm_in, w.q_proj)          # [seq, 2048]
    k_raw = F.linear(norm_in, w.k_proj)          # [seq, 1024]
    v_raw = F.linear(norm_in, w.v_proj)          # [seq, 1024]
    rec("attn_qkv", {"x": norm_in}, {"q": q_raw, "k": k_raw, "v": v_raw})

    # --- per-head QK-norm (q/k only; v not normalized) ---
    q = rmsnorm(q_raw.view(seq, N_HEADS, HEAD_DIM), w.q_norm).transpose(0, 1)     # [16, seq, 128]
    k = rmsnorm(k_raw.view(seq, N_KV_HEADS, HEAD_DIM), w.k_norm).transpose(0, 1)  # [8, seq, 128]
    v = v_raw.view(seq, N_KV_HEADS, HEAD_DIM).transpose(0, 1)                     # [8, seq, 128]
    rec("attn_qknorm", {"q": q_raw, "k": k_raw}, {"q": q, "k": k})

    # --- RoPE ---
    q_rot = apply_rope(q, cos, sin)              # [16, seq, 128]
    k_rot = apply_rope(k, cos, sin)              # [8, seq, 128]
    rec("attn_rope", {"q": q, "k": k, "cos": cos, "sin": sin},
        {"q": q_rot, "k": k_rot})

    # --- KV cache append (decode) ---
    if kv_cache is not None and kv_cache["k"].numel() > 0:
        k_full = torch.cat([kv_cache["k"], k_rot], dim=1)   # [8, L+seq, 128]
        v_full = torch.cat([kv_cache["v"], v], dim=1)
    else:
        k_full, v_full = k_rot, v                           # [8, seq, 128]

    k_rep = repeat_kv(k_full, N_REP)             # [16, seq_k, 128]
    v_rep = repeat_kv(v_full, N_REP)

    # --- QK^T * scale ---
    scores = torch.matmul(q_rot, k_rep.transpose(-1, -2)) * ATTN_SCALE  # [16, seq, seq_k]
    rec("attn_score", {"q": q_rot, "k": k_full}, {"scores": scores})

    # --- causal mask + softmax (fp32) ---
    masked = scores + mask if mask is not None else scores
    probs = F.softmax(masked, dim=-1, dtype=torch.float32).to(q_rot.dtype)  # [16, seq, seq_k]
    rec("attn_softmax",
        {"scores": scores, "mask": mask.squeeze(0) if mask is not None else None},
        {"probs": probs})

    # --- AV ---
    ctx = torch.matmul(probs, v_rep)             # [16, seq, 128]
    rec("attn_ctx", {"probs": probs, "v": v_full}, {"ctx": ctx})

    # --- O projection ---
    attn_out = ctx.transpose(0, 1).reshape(seq, N_HEADS * HEAD_DIM).contiguous()
    o = F.linear(attn_out, w.o_proj)             # [seq, 1024]
    rec("attn_o", {"ctx": ctx}, {"o": o})

    # --- residual (attn) ---
    hidden1 = hidden + o
    rec("residual_attn", {"x": hidden, "attn_o": o}, {"y": hidden1})

    # --- pre-MLP RMSNorm ---
    norm_post = rmsnorm(hidden1, w.post_norm)
    rec("rmsnorm_mlp", {"x": hidden1}, {"y": norm_post})

    # --- MLP ---
    gate = F.linear(norm_post, w.gate)           # [seq, 3072]
    rec("mlp_gate", {"x": norm_post}, {"gate": gate})
    up = F.linear(norm_post, w.up)
    rec("mlp_up", {"x": norm_post}, {"up": up})
    act = silu(gate) * up                        # [seq, 3072]
    rec("mlp_silu", {"gate": gate, "up": up}, {"y": act})
    down = F.linear(act, w.down)                 # [seq, 1024]
    rec("mlp_down", {"x": act}, {"down": down})

    # --- residual (mlp) ---
    hidden2 = hidden1 + down
    rec("residual_mlp", {"x": hidden1, "mlp_down": down}, {"y": hidden2})

    return hidden2, k_full, v_full


# --------------------------------------------------------------------------- #
# full model forward
# --------------------------------------------------------------------------- #
class Qwen3Ref:
    """Reference model holding HF state_dict; forward with optional trace capture."""

    def __init__(self, state_dict: dict, device="cuda"):
        self.sd = state_dict
        self.device = device
        self.embed = state_dict["model.embed_tokens.weight"]
        self.lm_head = state_dict["lm_head.weight"]          # tied to embed (verified)
        self.final_norm = state_dict["model.norm.weight"]
        self.layers = [LayerWeights(state_dict, i) for i in range(N_LAYERS)]
        self.inv_freq = rope_inv_freq(HEAD_DIM, ROPE_THETA)

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor, cache_pos: int = 0,
                kv_cache: list | None = None, trace: list | None = None):
        """token_ids: [seq] int64. Returns (logits, new_kv_cache).

        kv_cache: list of per-layer dicts {"k","v"} (or None = prefill from scratch).
        trace: if a list is given, (name, trace_dict) entries are appended.
        """
        seq = token_ids.shape[0]
        positions = torch.arange(cache_pos, cache_pos + seq, device=self.device)
        cos, sin = rope_embeddings(positions, self.inv_freq.to(self.device), torch.bfloat16)

        hidden = self.embed[token_ids.to(self.device)]      # [seq, 1024] bf16
        if trace is not None:
            trace.append(("embed", {"inputs": {"x": token_ids}, "outputs": {"y": hidden}}))

        mask = causal_mask(seq, cache_pos + seq, cache_pos, torch.bfloat16, self.device)

        new_cache = []
        for i, lw in enumerate(self.layers):
            ltr = {} if trace is not None else None
            layer_cache = kv_cache[i] if kv_cache is not None else None
            hidden, k_full, v_full = layer_forward(lw, hidden, cos, sin, mask, layer_cache, ltr)
            if trace is not None:
                trace.append((i, ltr))
            new_cache.append({"k": k_full, "v": v_full})

        final = rmsnorm(hidden, self.final_norm)
        if trace is not None:
            trace.append(("final_norm", {"inputs": {"x": hidden}, "outputs": {"y": final}}))

        logits = F.linear(final, self.lm_head)             # [seq, 151936]
        if trace is not None:
            trace.append(("lm_head", {"inputs": {"x": final}, "outputs": {"logits": logits}}))

        return logits, new_cache
