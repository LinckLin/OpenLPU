"""Full golden trace generation (J1 schema).

- prefill: seq=128 (first 128-token block), all 28 layers + global ops.
- decode:  seq=1 at cache L in {0, 512, 1024, 2048, 4096, 8192}, all 28 layers + global ops.
- linear_wq: representative layer-0 q_proj GEMM (PF seq=128 / DC seq=1), via gen_linear_wq.py.

Model path / device are parameterized:
  MODEL_DIR  (default ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B)
  DEVICE     (default cuda)
  --full     explicit full mode (also the default; no partial subset is generated)

Storage: bf16 -> exact fp32 (dtype_np="float32(bf16)", dtype_code=0); y_ref pure fp32.
y_ref semantics: GEMM ops -> fp32 GEMM (x.float() @ W.float().T); non-GEMM -> y.float().
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (Qwen3Ref, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, HIDDEN,  # noqa: E402
                   INTERMEDIATE, VOCAB, ATTN_SCALE, RMS_EPS, ROPE_THETA, rmsnorm)
from golden import (bf16_to_np, fp32_to_np, int32_to_np, bf16_field, fp32_field,  # noqa: E402
                    int32_field, write_op_dir)

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
GOLDEN_ROOT = "golden/qwen3-0.6b"
DEVICE = os.environ.get("DEVICE", "cuda")

PREFILL_SEQ = 128
DECODE_CACHE_POINTS = [0, 512, 1024, 2048, 4096, 8192]  # full acceptance subset

FIXED_TEXT = (
    "The transformer architecture processes sequences in parallel using self-attention. "
    "Query, key, and value projections transform each token into vectors that participate in "
    "scaled dot product attention. Rotary embeddings encode absolute positions into relative "
    "differences. Grouped query attention shares key and value heads among several query heads. "
    "Each decoder layer applies layer normalization before attention and before the feed-forward "
    "network. The SwiGLU activation multiplies a gated linear projection by an up projection. "
    "Residual connections stabilize training and let gradients flow through deep networks. "
)

# GEMM ops where y_ref = fp32 GEMM
GEMM_OPS = {"attn_qkv", "attn_score", "attn_ctx", "attn_o",
            "mlp_gate", "mlp_up", "mlp_down", "lm_head", "linear_wq"}


def build_seq(tok, target_len):
    base = tok(FIXED_TEXT, return_tensors="pt")["input_ids"][0]
    reps = (target_len + base.shape[0]) // base.shape[0]
    return base.repeat(reps)[:target_len]


# --------------------------------------------------------------------------- #
# per-op meta building
# --------------------------------------------------------------------------- #
def op_params(op, seq, cache_pos):
    return {
        "embed": {"seq": seq},
        "rmsnorm_in": {"eps": RMS_EPS},
        "attn_qkv": {"K": HIDDEN, "q_out": N_HEADS * HEAD_DIM, "kv_out": N_KV_HEADS * HEAD_DIM},
        "attn_qknorm": {"eps": RMS_EPS, "head_dim": HEAD_DIM},
        "attn_rope": {"theta": ROPE_THETA, "pos": cache_pos},
        "attn_score": {"attn_scale": ATTN_SCALE, "n_rep": N_HEADS // N_KV_HEADS, "cache_len": cache_pos + seq},
        "attn_softmax": {"dim": -1, "fp32_softmax": True},
        "attn_ctx": {"cache_len": cache_pos + seq},
        "attn_o": {"K": N_HEADS * HEAD_DIM, "out": HIDDEN},
        "residual_attn": {},
        "rmsnorm_mlp": {"eps": RMS_EPS},
        "mlp_gate": {"K": HIDDEN, "out": INTERMEDIATE},
        "mlp_up": {"K": HIDDEN, "out": INTERMEDIATE},
        "mlp_silu": {},
        "mlp_down": {"K": INTERMEDIATE, "out": HIDDEN},
        "residual_mlp": {},
        "final_norm": {"eps": RMS_EPS},
        "lm_head": {"vocab": VOCAB},
    }[op]


def dump_layer_op(root, layer, mode, op, tr, seq, cache_pos, w):
    """tr: {"inputs": {...}, "outputs": {...}}. Builds meta + npz for one layer op."""
    ri, ro = tr["inputs"], tr["outputs"]

    def arrs(d):
        out = {}
        for k, v in d.items():
            if v is None:
                continue
            if v.dtype == torch.int64 or v.dtype == torch.int32:
                out[k] = (int32_to_np(v), int32_field(list(v.shape)))
            else:
                out[k] = (bf16_to_np(v), bf16_field(list(v.shape)))
        return out

    inputs = arrs(ri)
    outputs = arrs(ro)

    # y_ref
    y_refs = {}
    if op in GEMM_OPS:
        if op == "attn_qkv":
            x = ri["x"].float()
            y_refs["q"] = fp32_to_np(x @ w.q_proj.float().T)
            y_refs["k"] = fp32_to_np(x @ w.k_proj.float().T)
            y_refs["v"] = fp32_to_np(x @ w.v_proj.float().T)
        elif op == "attn_score":
            q = ri["q"].float(); k = ri["k"].float()
            k_rep = k[:, None, :, :].expand(N_KV_HEADS, N_HEADS // N_KV_HEADS, k.shape[1],
                                            HEAD_DIM).reshape(N_HEADS, k.shape[1], HEAD_DIM)
            y_refs["scores"] = fp32_to_np(q @ k_rep.transpose(-1, -2) * ATTN_SCALE)
        elif op == "attn_ctx":
            p = ri["probs"].float(); v = ri["v"].float()
            v_rep = v[:, None, :, :].expand(N_KV_HEADS, N_HEADS // N_KV_HEADS, v.shape[1],
                                            HEAD_DIM).reshape(N_HEADS, v.shape[1], HEAD_DIM)
            y_refs["ctx"] = fp32_to_np(p @ v_rep)
        elif op == "attn_o":
            ctx = ri["ctx"]
            seqq = ctx.shape[1]
            ctx2 = ctx.transpose(0, 1).reshape(seqq, N_HEADS * HEAD_DIM).float()
            y_refs["o"] = fp32_to_np(ctx2 @ w.o_proj.float().T)
        elif op == "mlp_gate":
            y_refs["gate"] = fp32_to_np(ri["x"].float() @ w.gate.float().T)
        elif op == "mlp_up":
            y_refs["up"] = fp32_to_np(ri["x"].float() @ w.up.float().T)
        elif op == "mlp_down":
            y_refs["down"] = fp32_to_np(ri["x"].float() @ w.down.float().T)
    else:
        for k, v in ro.items():
            if v is not None and v.dtype != torch.int64:
                y_refs[k] = fp32_to_np(v)

    # append y_ref to outputs
    for k, arr in y_refs.items():
        outputs[f"{k}_ref"] = (arr, fp32_field(list(arr.shape)))

    write_op_dir(f"{root}/L{layer:02d}_{op}", {
        "op": op, "layer": layer, "mode": mode,
        "inputs": inputs, "outputs": outputs,
        "params": op_params(op, seq, cache_pos),
        "weights_ref": weights_ref_for(op, layer),
    })


def weights_ref_for(op, layer):
    p = f"model.layers.{layer}.self_attn."
    m = f"model.layers.{layer}.mlp."
    return {
        "rmsnorm_in": [f"model.layers.{layer}.input_layernorm.weight"],
        "attn_qkv": [p + "q_proj.weight", p + "k_proj.weight", p + "v_proj.weight"],
        "attn_qknorm": [p + "q_norm.weight", p + "k_norm.weight"],
        "attn_rope": [],
        "attn_score": [],
        "attn_softmax": [],
        "attn_ctx": [],
        "attn_o": [p + "o_proj.weight"],
        "residual_attn": [],
        "rmsnorm_mlp": [f"model.layers.{layer}.post_attention_layernorm.weight"],
        "mlp_gate": [m + "gate_proj.weight"],
        "mlp_up": [m + "up_proj.weight"],
        "mlp_silu": [],
        "mlp_down": [m + "down_proj.weight"],
        "residual_mlp": [],
    }[op]


def dump_global_op(root, mode, name, tr, seq, cache_pos, ref):
    ri, ro = tr["inputs"], tr["outputs"]

    def arrs(d):
        out = {}
        for k, v in d.items():
            if v.dtype == torch.int64 or v.dtype == torch.int32:
                out[k] = (int32_to_np(v), int32_field(list(v.shape)))
            else:
                out[k] = (bf16_to_np(v), bf16_field(list(v.shape)))
        return out

    inputs = arrs(ri)
    outputs = arrs(ro)
    y_refs = {}
    if name == "lm_head":
        y_refs["logits"] = fp32_to_np(ri["x"].float() @ ref.lm_head.float().T)
    else:
        for k, v in ro.items():
            if v.dtype != torch.int64:
                y_refs[k] = fp32_to_np(v)
    for k, arr in y_refs.items():
        outputs[f"{k}_ref"] = (arr, fp32_field(list(arr.shape)))

    weights_ref = {
        "embed": ["model.embed_tokens.weight"],
        "final_norm": ["model.norm.weight"],
        "lm_head": ["lm_head.weight"],  # tied to embed_tokens (verified bitwise)
    }[name]
    write_op_dir(f"{root}/{name}", {
        "op": name, "layer": None, "mode": mode,
        "inputs": inputs, "outputs": outputs,
        "params": op_params(name, seq, cache_pos),
        "weights_ref": weights_ref,
    })


# --------------------------------------------------------------------------- #
def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="model dir with config.json + model.safetensors "
                         "(default $MODEL_DIR or ~/.cache/.../models--Qwen--Qwen3-0.6B)")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--full", action="store_true",
                    help="explicit full acceptance subset (default; prefill + all decode points + linear_wq)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device).eval()
    ref = Qwen3Ref({k: v for k, v in model.state_dict().items()}, device=args.device)

    max_cache = max(DECODE_CACHE_POINTS)
    seq = build_seq(tok, max_cache + 1).to(args.device)

    # ===================== prefill seq=128 =====================
    print(f"prefill seq={PREFILL_SEQ} ...")
    root = f"{GOLDEN_ROOT}/prefill_seq{PREFILL_SEQ}"
    pseq = seq[:PREFILL_SEQ]
    dtr = []
    ref.forward(pseq, cache_pos=0, kv_cache=None, trace=dtr)
    dump_global_op(root, "PF", "embed", dtr[0][1], PREFILL_SEQ, 0, ref)
    for i in range(N_LAYERS):
        ltr = dtr[i + 1][1]
        w = ref.layers[i]
        for op in ltr:
            dump_layer_op(root, i, "PF", op, ltr[op], PREFILL_SEQ, 0, w)
    dump_global_op(root, "PF", "final_norm", dtr[N_LAYERS + 1][1], PREFILL_SEQ, 0, ref)
    dump_global_op(root, "PF", "lm_head", dtr[N_LAYERS + 2][1], PREFILL_SEQ, 0, ref)

    # ===================== decode seq=1 @ cache points =====================
    print(f"building {max_cache}-token cache ...")
    _, cache = ref.forward(seq[:max_cache])

    for L in DECODE_CACHE_POINTS:
        print(f"decode cache={L} ...")
        root = f"{GOLDEN_ROOT}/decode_seq1_cache{L}"
        if L == 0:
            layer_cache = None
            cache_pos = 0
            tok_id = seq[0:1]
        else:
            layer_cache = [{"k": c["k"][:, :L, :], "v": c["v"][:, :L, :]} for c in cache]
            cache_pos = L
            tok_id = seq[L:L + 1]
        dtr = []
        ref.forward(tok_id, cache_pos=cache_pos, kv_cache=layer_cache, trace=dtr)
        dump_global_op(root, "DC", "embed", dtr[0][1], 1, cache_pos, ref)
        for i in range(N_LAYERS):
            ltr = dtr[i + 1][1]
            w = ref.layers[i]
            for op in ltr:
                dump_layer_op(root, i, "DC", op, ltr[op], 1, cache_pos, w)
        dump_global_op(root, "DC", "final_norm", dtr[N_LAYERS + 1][1], 1, cache_pos, ref)
        dump_global_op(root, "DC", "lm_head", dtr[N_LAYERS + 2][1], 1, cache_pos, ref)

    # ===================== linear_wq_pf / linear_wq_dc =====================
    from gen_linear_wq import generate as gen_linear_wq  # noqa: E402
    gen_linear_wq(model, tok, ref, args.device)

    print("DONE full golden")


if __name__ == "__main__":
    main()
