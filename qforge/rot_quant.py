"""SpinQuant-nohad compile-time rotation absorption (W4A16, training-free).

Implements the two mergeable rotations of SpinQuant (2405.16406), which leave
the full-precision network numerically invariant and are fully absorbed into
the weight matrices (no online transform, activations stay BF16, qbin/ISA
format unchanged):

  R1 (residual stream, GLOBAL):  rotate the residual stream by one orthogonal
      R (H x H).  To stay exact across RMSNorm, the RMSNorm scale parameters
      alpha are folded into the weight matrix right after the norm (SpinQuant
      footnote 2 -> Ashkboos 2023), making the norm a parameter-free unit-norm
      op, which commutes with rotations:  norm(R x) = R norm(x).
      Absorptions (row convention, "rotate row v -> v @ R"):
        embed   <- embed   @ R
        q_proj  <- q_proj  @ diag(w_in)   @ R
        k_proj  <- k_proj  @ diag(w_in)   @ R
        v_proj  <- v_proj  @ diag(w_in)   @ R
        o_proj  <- R^T      @ o_proj
        gate    <- gate    @ diag(w_post) @ R
        up      <- up      @ diag(w_post) @ R
        down    <- R^T      @ down
        lm_head <- lm_head @ diag(w_final) @ R
        input/post/final RMSNorm weights -> 1.0 (bare norm)
      The q/k INPUT rotation is lossless and does NOT touch q_norm/k_norm/RoPE
      (those act on the 128-dim head output of q/k, orthogonal to this input
      dim).  A *partial* "gate/up/down only" residual rotation is mathematically
      infeasible (the residual add shares the same hidden basis across the
      attention and MLP branches).

  R2 (V-O head pair, per layer per KV head):  head-wise R_h (D_head x D_head)
      rotation of the value vectors and the matching O-projection input columns:
        v_proj[head h rows]      <- R_h @ v_proj[head h rows]
        o_proj[:, head blocks]   <- o_proj[:, block] @ R_h^T
      Lossless (A (V R_h^T) = (A V) R_h^T, and c R_h^T (R_h O) = c O).

Rotation matrices are random Hadamard (Sylvester + random +/-1 sign flips),
deterministic under a seed.  Cayley-learned rotations (SpinQuant's SGD) are NOT
run in this training-free verification (noted in the report).

NOTE (verified on Qwen3-0.6B): the R1 folding multiplies each weight column by
the RMSNorm weight.  Qwen3-0.6B's RMSNorm weights contain large values (final
norm max ~15.3, some layer norms max ~2.1), so the folding *creates* weight
outliers (lm_head max 0.32 -> 2.0 after folding).  For 7B Llama the norm
weights are ~1 and this is negligible, but for 0.6B Qwen3 it makes symmetric
INT4 worse — reported honestly in w4a16-gptq.md.
"""
from __future__ import annotations

import numpy as np
import torch


def random_hadamard(n: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """n x n orthogonal random Hadamard (Sylvester + random +/-1 sign flips).

    n must be a power of two.  Normalized so R R^T = I.
    """
    assert (n & (n - 1)) == 0 and n > 0, f"n={n} is not a power of two"
    rng = np.random.default_rng(seed)
    H = np.ones((1, 1), dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    d1 = rng.integers(0, 2, n).astype(np.float32) * 2 - 1
    d2 = rng.integers(0, 2, n).astype(np.float32) * 2 - 1
    R = d1[:, None] * H * d2[None, :]
    return torch.tensor(R / np.sqrt(n), device=device, dtype=dtype)


def _rot_in(w: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """w @ R: rotate the input (residual) columns (one bf16 cast)."""
    return (w.to(torch.float32) @ R).to(w.dtype)


def _fold_rot_in(w: torch.Tensor, s: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """(w @ diag(s)) @ R: fold norm scale + rotate input columns (one bf16 cast)."""
    return ((w.to(torch.float32) * s.to(torch.float32)[None, :]) @ R).to(w.dtype)


def _rot_out(w: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """R^T @ w: rotate the output (residual) rows."""
    return (R.t() @ w.to(torch.float32)).to(w.dtype)


def apply_spinquant(sd: dict, *, hidden: int, n_layers: int, n_kv_heads: int,
                    head_dim: int, seed: int, device) -> dict:
    """Apply SpinQuant-nohad R1 (residual) + R2 (V-O head) to a state_dict.

    ``sd`` is the HF Qwen3 state_dict (bf16 tensors).  Returns a NEW dict with
    rotated weights and bare (unit) input/post/final norm weights; q_norm and
    k_norm are left untouched.  The input dict is not modified.
    """
    sd = {k: v.detach().clone() for k, v in sd.items()}
    R = random_hadamard(hidden, seed, device)

    # ---- R1: residual-stream rotation (global R) with RMSNorm folding ----
    sd["model.embed_tokens.weight"] = _rot_in(sd["model.embed_tokens.weight"], R)

    w_final = sd["model.norm.weight"]
    sd["lm_head.weight"] = _fold_rot_in(sd["lm_head.weight"], w_final, R)
    sd["model.norm.weight"] = torch.ones_like(w_final)

    for l in range(n_layers):
        p = f"model.layers.{l}"
        w_in = sd[f"{p}.input_layernorm.weight"]
        w_post = sd[f"{p}.post_attention_layernorm.weight"]

        for key in ("q_proj", "k_proj", "v_proj"):
            k = f"{p}.self_attn.{key}.weight"
            sd[k] = _fold_rot_in(sd[k], w_in, R)
        for k in (f"{p}.self_attn.o_proj.weight", f"{p}.mlp.down_proj.weight"):
            sd[k] = _rot_out(sd[k], R)
        for key in ("gate_proj", "up_proj"):
            k = f"{p}.mlp.{key}.weight"
            sd[k] = _fold_rot_in(sd[k], w_post, R)

        sd[f"{p}.input_layernorm.weight"] = torch.ones_like(w_in)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones_like(w_post)

    # ---- R2: V-O head-pair rotation (per layer, per KV head) ----
    for l in range(n_layers):
        p = f"model.layers.{l}"
        v = sd[f"{p}.self_attn.v_proj.weight"]
        o = sd[f"{p}.self_attn.o_proj.weight"]
        vf = v.to(torch.float32)
        of = o.to(torch.float32)
        for h in range(n_kv_heads):
            Rh = random_hadamard(head_dim, seed + 1000 * l + h, device)
            vf[h * head_dim:(h + 1) * head_dim, :] = \
                Rh @ vf[h * head_dim:(h + 1) * head_dim, :]
            for qh in (2 * h, 2 * h + 1):
                blk = slice(qh * head_dim, (qh + 1) * head_dim)
                of[:, blk] = of[:, blk] @ Rh.t()
        sd[f"{p}.self_attn.v_proj.weight"] = vf.to(v.dtype)
        sd[f"{p}.self_attn.o_proj.weight"] = of.to(o.dtype)

    return sd
