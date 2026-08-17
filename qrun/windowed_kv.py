"""O1 StreamingLLM windowed-KV — runtime + quality gate + performance.

Deliverable for perf-research O1 (roadmap §2 first tier, zero hardware change):

  * **Runtime** — windowed KV *read* for the qrun DC decode path.  The DC
    attention is re-emitted with **two KV.LOAD segments** per KV head instead of
    the full [0, ctx) window:

      1. attention **sinks** ``[0, S)`` (``S=4``, StreamingLLM attention sinks);
      2. a **rolling window** ``[pos-W+1, pos]`` (``W`` in {1024, 2048}).

    ``KV.APPEND`` still writes the full token stream (cache completeness is
    preserved — only the *read* window shrinks).  The per-token DC patch moves
    the rolling KV.LOAD ``pos_start`` instead of clamping a full-window count.

  * **Quality gate** — reuses ``quality_baseline.py``'s exact spans / evaluation
    functions to score windowed attention (``S`` sinks + rolling ``W``) with the
    identical ``token_ids`` from ``quality_baseline.json`` and computes the
    per-sample + pooled relative ΔPPL against the golden full-attention
    ``per_token_nll``.  HellaSwag ``acc_norm`` (same 100 seed-0 indices) and a
    20-token cross-consistency vs the BF16 full-attention baseline are recorded
    as secondary observations.

  * **Performance** — recomputes the qsim timing-model decode ceiling for the
    windowed KV re-read bucket ``R = S + W`` (``qsim/timing_p6`` frozen
    constants) and measures the qrun windowed single-token wall clock vs full.

The HF windowed attention uses a 4D additive mask in the native eager dtype
(bf16, ``finfo(bf16).min`` fill) so windowed and golden full attention differ
*only* in the mask pattern, not the arithmetic dtype.

Scope note: the windowed DC program assumes ``pos >= W + S - 1`` (window and
sinks disjoint); every evaluation here bootstraps 4K/8K contexts, so this always
holds.  The early-decode band is listed as a review item.

Usage::

  python3 qrun/windowed_kv.py \
      --model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B \
      --qbin /tmp/qwen3-0.6b-bf16.qbin \
      --baseline docs/perf-research/quality-baseline/quality_baseline.json \
      --sinks 4 --windows 1024,2048 --seq-lens 4096,8192 \
      --out /tmp/o1-streamingllm.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from math import ceil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from qrun import bf16 as B  # noqa: E402
from qrun import program as P  # noqa: E402
from compiler.isa import isa as I  # noqa: E402
from qrun.runtime import RunEngine, _CKVPOS_BITS, _IMM32_BITS  # noqa: E402
from qrun.engine import build_engine  # noqa: E402
from qrun.reference import load_hf  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"),
)
QBIN = "/tmp/qwen3-0.6b-bf16.qbin"
BASELINE = "docs/perf-research/quality-baseline/quality_baseline.json"

S_DEFAULT = 4
W_DEFAULT = [1024, 2048]

O1_DELTA_PPL_GATE = 0.02  # <= 2% relative ΔPPL is the hard gate (roadmap O1).

# KV.LOAD field bit layout (compiler/isa/isa.py): pos_start @ bit 68, 13-bit.
_KVLOAD_POS_START_BITS = 68
_KVLOAD_POS_START_WIDTH = 13


# =============================================================================
# Part A — HF windowed attention (the quality-gate scoring path)
# =============================================================================

def streaming_mask(L: int, S: int, W: int, device) -> torch.Tensor:
    """[1, 1, L, L] additive mask for StreamingLLM sinks + rolling window.

    Query i attends to key j iff ``j <= i`` (causal) and (``j < S`` sink or
    ``j >= i - W + 1`` rolling).  Filled with 0 / ``finfo(bf16).min`` in bf16 —
    identical dtype+fill to the native eager causal mask, so windowed and golden
    full attention differ only in the mask pattern.
    """
    idx = torch.arange(L, device=device)
    causal = idx[:, None] >= idx[None, :]                      # j <= i
    sink = idx[None, :] < S                                    # j in [0, S)
    window = idx[None, :] >= (idx[:, None] - W + 1)            # j >= i - W + 1
    allowed = causal & (sink | window)
    minv = torch.finfo(torch.bfloat16).min
    mask = torch.where(allowed,
                       torch.zeros((), device=device, dtype=torch.bfloat16),
                       torch.full((), minv, device=device, dtype=torch.bfloat16))
    return mask[None, None]  # [1, 1, L, L]


@torch.no_grad()
def windowed_attention_nll(model, input_ids: torch.Tensor, S: int, W: int,
                           mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-token NLL under StreamingLLM windowed attention (sinks + window).

    ``input_ids``: [1, L].  Returns [L-1] fp32 NLL, element ``i`` = the NLL of
    token ``i+1`` predicted from windowed attention over the L-token span.
    """
    L = input_ids.shape[1]
    if mask is None:
        mask = streaming_mask(L, S, W, input_ids.device)
    logits = model(input_ids=input_ids, attention_mask=mask).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)      # fp32 softmax (golden)
    targets = input_ids[:, 1:]
    nll = -log_probs[:, :-1, :].gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return nll[0]  # [L-1] fp32


@torch.no_grad()
def windowed_hellaswag_score(model, tok, ctx: str, endings, device: str,
                             S: int, W: int):
    """Windowed mirror of ``quality_baseline.hellaswag_score`` (per-ending, 4D
    windowed mask).  Returns ``(raw, norm, n_toks)`` parallel to the endings."""
    ctx_ids = tok.encode(ctx, add_special_tokens=False)
    sep = "" if ctx.endswith(" ") else " "
    raw, norm, n_toks = [], [], []
    for e in endings:
        full_ids = tok.encode(ctx + sep + e, add_special_tokens=False)
        L = len(full_ids)
        n_ctx = len(ctx_ids)
        if L <= n_ctx:
            raw.append(0.0)
            norm.append(0.0)
            n_toks.append(0)
            continue
        inp = torch.tensor([full_ids], dtype=torch.long, device=device)
        mask = streaming_mask(L, S, W, device)
        logits = model(input_ids=inp, attention_mask=mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        pos = torch.arange(n_ctx - 1, L - 1, device=device)
        tgt = inp[0, n_ctx:L]
        s = log_probs[0, pos, tgt].sum().item()
        k = L - n_ctx
        raw.append(s)
        norm.append(s / k)
        n_toks.append(k)
    return raw, norm, n_toks


# =============================================================================
# Part B — windowed DC program + per-token patch (qrun runtime)
# =============================================================================

def _emit_attention_dc_windowed(em, layer: int, b: dict, mask_base: int,
                                S: int, W: int):
    """Decode attention with sinks [0,S) + rolling window [pos-W+1, pos].

    One sink KV.LOAD (count=S) + one rolling KV.LOAD (count=W) per KV head;
    the current token is included in the rolling window (reached through the
    13-bit pos_start), so the full-window VMOV current-token subtile is dropped.
    """
    q_stride = P.HD * 2
    ctx_stride = P.HD * 2
    score_stride = P.N_TILE * 2
    n_window_subtiles = W // P.N_TILE
    em.cfg_c(P.C_CA, P._stride(q_stride, q_stride))
    em.cfg_c(P.C_CB, P._stride(P.HD * 2, 0))
    em.cfg_c(P.C_CC, P._stride(score_stride, score_stride))

    for g in range(P.KVH):
        em.cfg_ar(P.AR_QG, b["q"] + 2 * g * q_stride)
        first = True
        sub_idx = 0

        # attention sinks [0, S)
        em.cfg_ar(P.AR_KSTAGE, P.KSTAGE_SRAM)
        em.cfg_ar(P.AR_VSTAGE, P.VSTAGE_SRAM)
        em.i("KV.LOAD", dstK=P.AR_KSTAGE, dstV=P.AR_VSTAGE, layer=layer,
             head=g, sel=2, pos_start=0, count=S)
        em.wait(8)
        em.cfg_ar(P.AR_KSTAGE, P.KSTAGE_SRAM)
        em.cfg_ar(P.AR_VSTAGE, P.VSTAGE_SRAM)
        P._emit_subtile(em, b, first, mask_base + sub_idx * (2 * P.N_TILE) * 2)
        first = False
        sub_idx += 1

        # rolling window [pos-W+1, pos] (pos_start patched per token)
        em.cfg_ar(P.AR_KSTAGE, P.KSTAGE_SRAM)
        em.cfg_ar(P.AR_VSTAGE, P.VSTAGE_SRAM)
        em.i("KV.LOAD", dstK=P.AR_KSTAGE, dstV=P.AR_VSTAGE, layer=layer,
             head=g, sel=2, pos_start=0, count=W)
        em.wait(8)
        for _st in range(n_window_subtiles):
            em.cfg_ar(P.AR_KSTAGE, P.KSTAGE_SRAM + _st * P.N_TILE * P.HD * 2)
            em.cfg_ar(P.AR_VSTAGE, P.VSTAGE_SRAM + _st * P.N_TILE * P.HD * 2)
            P._emit_subtile(em, b, first, mask_base + sub_idx * (2 * P.N_TILE) * 2)
            first = False
            sub_idx += 1

        # online-softmax finalize + ctx writeback (mirror _emit_attention_dc_qr)
        for hh in range(P.GQA):
            ch = b["ctx_acc"] + hh * P.HD * 2
            lh = b["lrun"] + hh * 16
            rh = b["rinv"] + hh * 16
            em.cfg_ar(P.AR_LRUN, lh)
            em.cfg_ar(P.AR_RINV, rh)
            em.cfg_ar(P.AR_SC0, ch)
            em.cfg_ar(P.AR_SC1, b["ctx"] + (2 * g + hh) * ctx_stride)
            em.i("VRECIP", srcA=I.DT_BF16, acc=I.ACC_FP32,
                 ara=P.AR_LRUN, arb=P.AR_ONES, ard=P.AR_RINV, len=1, cv=0, imm=0)
            em.i("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                 ara=P.AR_SC0, arb=P.AR_RINV, ard=P.AR_SC0, len=P.HD,
                 cv=P.C_BROADCAST, imm=0)
            em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
                 ara=P.AR_SC0, arb=P.AR_ONES, ard=P.AR_SC1, len=P.HD, cv=0, imm=0)
        em.barrier()


def build_windowed_dc_program(layouts, dtype: str, hbm, S: int, W: int,
                              slab_shift: int = 22, n_layers: int = 28) -> bytes:
    """Rebuild the DC program with the windowed attention emission.

    Monkeypatches ``P._emit_attention_dc_qr`` for the duration of the
    ``lower_transformer`` call (the generator resolves the module global at call
    time), then restores it — ``qrun/program.py`` is left untouched.
    """
    orig = P._emit_attention_dc_qr
    P._emit_attention_dc_qr = lambda em, L, b, mask_base: \
        _emit_attention_dc_windowed(em, L, b, mask_base, S, W)
    try:
        return P.lower_transformer("DC", layouts, dtype, hbm,
                                   slab_shift=slab_shift, n_layers=n_layers)
    finally:
        P._emit_attention_dc_qr = orig


def build_windowed_tail_mask(S: int, W: int) -> bytes:
    """Static per-subtile BF16 tail mask: sink subtile valid on [0,S), the
    ``W // N_TILE`` rolling subtiles fully valid.  Shape [1 + W/128, 2, 128]."""
    n_window = W // P.N_TILE
    n_total = 1 + n_window
    mask = np.full((n_total, 2, P.N_TILE), -np.inf, dtype=B.BF16_NP)
    mask[0, :, :S] = 0.0
    mask[1:, :, :] = 0.0
    return mask.tobytes()


@dataclass
class WindowedDcPatch:
    """Pre-decoded patch locations in the windowed DC program."""
    words: list[int]
    ckv_pos_idx: int | None = None
    rope_idx: list[int] = field(default_factory=list)
    rolling_kvload: list[int] = field(default_factory=list)  # KV.LOAD with count==W
    W: int = 1024


def build_windowed_dc_patch(program: bytes, W: int) -> WindowedDcPatch:
    words = [int.from_bytes(program[o:o + 16], "little")
             for o in range(0, len(program), 16)]
    p = WindowedDcPatch(words=words, W=W)
    for i, w in enumerate(words):
        d = I.decode_inst(w)
        if d["mnemonic"] == "CONFIG" and d["reg_class"] == 0 and d["REG"] == 30:
            if p.ckv_pos_idx is None:
                p.ckv_pos_idx = i
        elif d["mnemonic"] == "ROPE":
            p.rope_idx.append(i)
        elif d["mnemonic"] == "KV.LOAD" and d["count"] == W:
            p.rolling_kvload.append(i)
    assert p.ckv_pos_idx is not None, "windowed DC program missing C_KV_POS CONFIG"
    assert len(p.rolling_kvload) == 28 * P.KVH, \
        f"expected 28*{P.KVH} rolling KV.LOAD, found {len(p.rolling_kvload)}"
    return p


def patch_windowed_dc(p: WindowedDcPatch, pos: int) -> bytes:
    """Patch the windowed DC program for decode position ``pos``:
    C_KV_POS = pos, ROPE = pos, rolling KV.LOAD pos_start = pos - W + 1."""
    words = list(p.words)
    words[p.ckv_pos_idx] = ((words[p.ckv_pos_idx]
                             & ~(((1 << 64) - 1) << _CKVPOS_BITS))
                            | (pos << _CKVPOS_BITS))
    for i in p.rope_idx:
        words[i] = ((words[i] & ~(0xFFFF << _IMM32_BITS))
                    | ((pos & 0xFFFF) << _IMM32_BITS))
    ps = pos - p.W + 1
    assert 0 <= ps < (1 << _KVLOAD_POS_START_WIDTH)
    psmask = ((1 << _KVLOAD_POS_START_WIDTH) - 1) << _KVLOAD_POS_START_BITS
    for i in p.rolling_kvload:
        words[i] = ((words[i] & ~psmask) | (ps << _KVLOAD_POS_START_BITS))
    return b"".join(w.to_bytes(16, "little") for w in words)


class WindowedEngine:
    """Windowed-KV decode wrapper around an existing ``RunEngine``.

    Reuses the engine's QMetal / HBM plan / layouts / embedding / bootstrap and
    only swaps the DC program, patch and tail mask for the windowed pattern.
    """

    def __init__(self, eng: RunEngine, S: int, W: int):
        self.eng = eng
        self.S = S
        self.W = W
        self.dc_program = build_windowed_dc_program(eng.layouts, eng.dtype,
                                                    eng.plan, S, W)
        self.dc_patch = build_windowed_dc_patch(self.dc_program, W)
        self.mask = build_windowed_tail_mask(S, W)

    def bootstrap_kv(self, token_ids):
        return self.eng.bootstrap_kv(token_ids)

    def decode_step(self, pos: int, token_id: int) -> np.ndarray:
        assert pos >= self.W + self.S - 1, \
            "windowed decode requires pos >= W+S-1 (window/sinks disjoint)"
        hidden = self.eng.embed_ids(np.array([token_id], dtype=np.int64))
        self.eng.write_input_hbm(hidden, 1)
        self.eng.qmetal.write_sram(P.MASK_BASE, self.mask)
        self.eng.qmetal.run_dc(patch_windowed_dc(self.dc_patch, pos))
        return self.eng.read_logits(1, 0)

    def generate(self, prompt_ids: np.ndarray, max_new: int, *, bootstrap=True):
        """Windowed greedy decode (bootstrap path; prompt >= W)."""
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        new_tokens: list[int] = []
        logits_list: list[np.ndarray] = []
        if bootstrap:
            self.bootstrap_kv(torch.from_numpy(prompt_ids))
            pos = prompt_ids.shape[0] - 1
            logits = self.decode_step(pos, int(prompt_ids[-1]))
            pos += 1
        else:
            logits = self.eng.prefill(prompt_ids)
            pos = prompt_ids.shape[0]
        for _ in range(max_new):
            tok = self.eng.argmax(logits)
            new_tokens.append(tok)
            logits_list.append(logits)
            logits = self.decode_step(pos, tok)
            pos += 1
        return new_tokens, logits_list


# =============================================================================
# Part C — qsim timing-model ceiling (windowed KV re-read bucket R = S + W)
# =============================================================================

def ceiling_table(S: int, windows, seq_lens=(4096, 8192)) -> dict:
    """Recompute the decode ceiling with the qsim timing model (timing_p6).

    The windowed KV re-read bucket is ``R = S + W`` tokens/layer (sinks + rolling
    window), independent of context length; the full-window baseline uses
    ``R = ctx``.  ``schedule_overlap`` reproduces the roadmap/flashattn 866/1009
    numbers exactly; ``schedule_double_buffer`` is the P6 scheduled baseline.
    """
    from qsim.timing import CFG, hbm_read_cycles
    from qsim.timing_p6 import (WEIGHT_LAYER, decode_demands, schedule_overlap,
                                schedule_double_buffer, full_token_cycles)

    def tok_per_s(cycles: int) -> float:
        return 1e9 / cycles

    # P6 convention: double-buffer schedule adds a one-time layer-0 weight
    # prefetch (pipeline fill), matching p6-opt-report's 481.1 / 255.2 tok/s.
    fill = hbm_read_cycles(WEIGHT_LAYER)

    out = {"sinks": S, "windows": {}, "baseline": {}}
    for ctx in seq_lens:
        d = decode_demands(ctx)
        ov = full_token_cycles(d, schedule_overlap(d))
        db = full_token_cycles(d, schedule_double_buffer(d), fill)
        out["baseline"][str(ctx)] = {
            "R_tokens": ctx,
            "overlap_cycles": ov, "overlap_tok_s": round(tok_per_s(ov), 1),
            "double_buffer_cycles": db, "double_buffer_tok_s": round(tok_per_s(db), 1),
        }
    for W in windows:
        R = S + W
        d = decode_demands(R)          # decode_demands treats ctx = KV re-read count
        ov = full_token_cycles(d, schedule_overlap(d))
        db = full_token_cycles(d, schedule_double_buffer(d), fill)
        out["windows"][str(W)] = {
            "S": S, "W": W, "R_tokens": R,
            "kv_bytes_per_layer": CFG.kv_bytes_per_token_per_layer * R,
            "weight_read_cycles": d.weight_read,
            "kv_read_cycles": d.kv_read,
            "kv_sram_write_cycles": d.kv_sram_write,
            "per_layer_wall_cycles": schedule_overlap(d),
            "overlap_cycles": ov, "overlap_tok_s": round(tok_per_s(ov), 1),
            "double_buffer_cycles": db, "double_buffer_tok_s": round(tok_per_s(db), 1),
        }
    return out


# =============================================================================
# Part D — driver
# =============================================================================

FIXED_TEXT = ("The transformer architecture processes sequences in parallel. "
              "Attention weighs the relevance of every token against the query. ")


def build_seq(tok, target_len: int) -> np.ndarray:
    base = tok(FIXED_TEXT, return_tensors="pt")["input_ids"][0]
    reps = (target_len + base.shape[0]) // base.shape[0]
    return base.repeat(reps)[:target_len].numpy()


def evaluate_windowed_ppl(hf, baseline_ppl: dict, seq_lens, S, W,
                          device: str) -> dict:
    """Score the baseline JSON's exact spans with windowed attention; compute
    per-sample + pooled ΔPPL vs the golden full-attention per_token_nll."""
    masks = {}
    out = {}
    for L in seq_lens:
        key = str(L)
        spans = baseline_ppl[key]
        L = int(L)
        if (L, W) not in masks:
            masks[(L, W)] = streaming_mask(L, S, W, device)
        mask = masks[(L, W)]
        rows = []
        all_dnll = []
        all_w = []
        all_f = []
        for s in spans:
            ids = torch.tensor([s["token_ids"]], dtype=torch.long, device=device)
            wnll = windowed_attention_nll(hf, ids, S, W, mask=mask)
            fnll = np.asarray(s["per_token_nll"], dtype=np.float64)
            dnll = wnll.double().cpu().numpy() - fnll           # per-token ΔNLL
            mean_w = float(wnll.mean().item())
            mean_f = float(s["mean_nll"])
            dppl = float(np.exp(dnll.mean()) - 1.0)              # per-sample ΔPPL
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
        out[key] = {
            "per_sample": rows,
            "pooled_mean_dnll": float(pooled_dnll.mean()),
            "pooled_delta_ppl": pooled_dppl,
            "pooled_ppl_full": float(np.exp(np.mean(all_f))),
            "pooled_ppl_windowed": float(np.exp(np.mean(all_w))),
            "gate": O1_DELTA_PPL_GATE,
            "pass": bool(pooled_dppl <= O1_DELTA_PPL_GATE),
        }
    return out


def evaluate_windowed_hellaswag(hf, tok, baseline_hs: dict, S, W,
                                device: str) -> dict:
    """Re-score the baseline JSON's exact 100 seed-0 indices with windowed
    attention; record acc_norm / acc_raw and Δacc vs the golden full score."""
    from datasets import load_dataset

    idxs = baseline_hs["indices"]
    val = load_dataset("Rowan/hellaswag")["validation"]
    n_correct = 0
    n_correct_raw = 0
    max_len = 0
    for i in idxs:
        row = val[int(i)]
        ctx = row["ctx"]
        endings = list(row["endings"])
        label = int(row["label"])
        max_len = max(max_len,
                      max(len(tok.encode(ctx + ("" if ctx.endswith(" ") else " ")
                                          + e, add_special_tokens=False))
                          for e in endings))
        raw, norm, n_toks = windowed_hellaswag_score(hf, tok, ctx, endings,
                                                     device, S, W)
        pred = int(np.argmax(norm))
        pred_raw = int(np.argmax(raw))
        n_correct += int(pred == label)
        n_correct_raw += int(pred_raw == label)
    acc_full = baseline_hs["accuracy_norm"]
    acc_w = n_correct / len(idxs)
    return {
        "n_examples": len(idxs),
        "sinks": S, "W": W,
        "max_ctx_plus_ending_tokens": max_len,
        "acc_norm_full": acc_full,
        "acc_norm_windowed": acc_w,
        "acc_raw_windowed": n_correct_raw / len(idxs),
        "delta_acc_norm": acc_w - acc_full,
    }


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--qbin", default=QBIN)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--sinks", type=int, default=S_DEFAULT)
    ap.add_argument("--windows", default="1024,2048")
    ap.add_argument("--seq-lens", default="4096,8192")
    ap.add_argument("--out", default="/tmp/o1-streamingllm.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-hellaswag", action="store_true")
    ap.add_argument("--skip-runtime", action="store_true")
    ap.add_argument("--cross-tokens", type=int, default=20)
    ap.add_argument("--wallclock-reps", type=int, default=5)
    args = ap.parse_args()

    S = args.sinks
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    device = args.device

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
            "sinks": S,
            "windows": windows,
            "seq_lens": seq_lens,
            "baseline": args.baseline,
            "delta_ppl_gate": O1_DELTA_PPL_GATE,
            "device": device,
        },
    }

    # ---- quality gate: ΔPPL (windowed vs full) -------------------------
    if not args.skip_ppl:
        print(f"=== ΔPPL (windowed vs full, S={S}) ===", flush=True)
        result["ppl"] = {}
        for W in windows:
            print(f"  W={W}:", flush=True)
            result["ppl"][str(W)] = evaluate_windowed_ppl(
                hf, baseline["ppl"], seq_lens, S, W, device)
            for key, r in result["ppl"][str(W)].items():
                print(f"    L={key}: pooled ΔPPL={r['pooled_delta_ppl']*100:.3f}% "
                      f"(PPL {r['pooled_ppl_full']:.3f} -> "
                      f"{r['pooled_ppl_windowed']:.3f}) "
                      f"{'PASS' if r['pass'] else 'FAIL'}", flush=True)

    # ---- HellaSwag sanity -----------------------------------------------
    if not args.skip_hellaswag:
        print(f"=== HellaSwag sanity (S={S}) ===", flush=True)
        result["hellaswag"] = {}
        for W in windows:
            result["hellaswag"][str(W)] = evaluate_windowed_hellaswag(
                hf, tok, baseline["hellaswag"], S, W, device)
            r = result["hellaswag"][str(W)]
            print(f"  W={W}: acc_norm {r['acc_norm_windowed']:.4f} "
                  f"(full {r['acc_norm_full']:.4f}, Δ {r['delta_acc_norm']:+.4f}) "
                  f"| max ctx+ending {r['max_ctx_plus_ending_tokens']} tokens",
                  flush=True)

    # ---- runtime: windowed qrun (cross-consistency + wall clock) --------
    if not args.skip_runtime:
        print(f"=== qrun windowed runtime (S={S}) ===", flush=True)
        eng = build_engine(args.qbin, args.model_dir, "bf16", tokenizer=tok,
                           ref=ref)
        result["runtime"] = {"windows": {}}

        # cross-consistency prompt (4K so windowed != full)
        pid = build_seq(tok, 4096)
        print(f"  full-attention baseline (20 tokens) ...", flush=True)
        full_tokens, _ = eng.generate(pid, args.cross_tokens, bootstrap=True)
        for W in windows:
            weng = WindowedEngine(eng, S, W)
            t0 = time.time()
            w_tokens, _ = weng.generate(pid, args.cross_tokens, bootstrap=True)
            gen_s = time.time() - t0
            div = [i for i in range(len(full_tokens))
                   if w_tokens[i] != full_tokens[i]]
            print(f"  W={W}: {len(full_tokens)-len(div)}/{len(full_tokens)} match; "
                  f"divergences at {div}", flush=True)
            result["runtime"]["windows"][str(W)] = {
                "full_tokens": full_tokens,
                "windowed_tokens": w_tokens,
                "divergence_positions": div,
                "n_divergences": len(div),
                "n_total": len(full_tokens),
                "generate_s": gen_s,
            }

        # wall clock: single decode step, full vs windowed (pos 4095)
        print(f"  wall clock (single decode_step @ pos 4095, "
              f"{args.wallclock_reps} reps) ...", flush=True)
        wc = {"reps": args.wallclock_reps, "full": {}, "windows": {}}
        token_id = int(pid[-1])
        # warmup
        eng.decode_step(4095, token_id)
        t0 = time.time()
        for _ in range(args.wallclock_reps):
            eng.decode_step(4095, token_id)
        wc["full"]["mean_s"] = (time.time() - t0) / args.wallclock_reps
        for W in windows:
            weng = WindowedEngine(eng, S, W)
            weng.decode_step(4095, token_id)   # warmup
            t0 = time.time()
            for _ in range(args.wallclock_reps):
                weng.decode_step(4095, token_id)
            mean_s = (time.time() - t0) / args.wallclock_reps
            wc["windows"][str(W)] = {
                "mean_s": mean_s,
                "speedup_vs_full": wc["full"]["mean_s"] / mean_s,
            }
            print(f"    W={W}: {mean_s:.2f} s/token "
                  f"({wc['full']['mean_s']/mean_s:.2f}x vs full "
                  f"{wc['full']['mean_s']:.2f} s)", flush=True)
        result["runtime"]["wallclock"] = wc

    # ---- performance model -------------------------------------------------
    print(f"=== qsim timing-model ceiling (S={S}) ===", flush=True)
    result["ceiling"] = ceiling_table(S, windows, seq_lens)
    for ctx, r in result["ceiling"]["baseline"].items():
        print(f"  baseline @{ctx}: overlap {r['overlap_tok_s']} tok/s | "
              f"double-buffer {r['double_buffer_tok_s']} tok/s", flush=True)
    for W, r in result["ceiling"]["windows"].items():
        print(f"  windowed W={W} (R={r['R_tokens']}): overlap "
              f"{r['overlap_tok_s']} tok/s | double-buffer "
              f"{r['double_buffer_tok_s']} tok/s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return result


if __name__ == "__main__":
    main()
