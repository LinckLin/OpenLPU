"""qrun reference access: HF model + Qwen3Ref (torch) for KV bootstrap and the
HF live-reference greedy decode used by the M4 comparisons.

The reference forward path (`ref/model.py`) is the op-decomposed BF16 eager
model verified token-identical to HF in P1 (`docs/p1/greedy_match.json`).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_REF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ref")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)
from model import Qwen3Ref  # noqa: E402


def load_hf(model_dir: str, device: str = "cuda"):
    """Return (tokenizer, hf_model, Qwen3Ref)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    ref = Qwen3Ref({k: v for k, v in hf.state_dict().items()}, device=device)
    return tok, hf, ref


@torch.no_grad()
def ref_greedy_with_logits(ref: Qwen3Ref, prompt_ids: torch.Tensor, n: int):
    """Greedy decode; returns (new_tokens list[int], logits_list list[torch]).

    logits_list[k] is the [vocab] logits (fp32, on device) used to pick the
    k-th generated token (the last prompt token for k=0).
    """
    ids = prompt_ids.to(ref.device)
    logits, cache = ref.forward(ids)
    new_tokens: list[int] = []
    logits_list = [logits[-1]]
    for _ in range(n):
        tok_id = logits[-1].argmax(dim=-1)
        new_tokens.append(int(tok_id.item()))
        logits, cache = ref.forward(tok_id[None],
                                    cache_pos=ids.shape[0] + len(new_tokens) - 1,
                                    kv_cache=cache)
        logits_list.append(logits[-1])
    return new_tokens, logits_list


def log_softmax_nll(logits: torch.Tensor, token: int) -> float:
    """-log softmax(logits)[token] (fp32), for NLL spans."""
    z = logits.detach().float()
    z = z - z.max()
    lse = torch.logsumexp(z, dim=-1)
    return float((lse - z[token]).item())


def np_log_softmax_nll(logits: np.ndarray, token: int) -> float:
    """-log softmax(logits)[token] (fp32 numpy), for qsim NLL spans."""
    z = logits.astype(np.float64)
    z = z - z.max()
    lse = np.log(np.exp(z).sum())
    return float(lse - z[token])
