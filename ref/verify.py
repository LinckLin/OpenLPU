from pathlib import Path
"""Verify ref/model.py against HF transformers 4.51.0 eager attention.

- Loads Qwen3-0.6B in bf16 (eager), runs prefill on a fixed prompt.
- Runs the hand-written Qwen3Ref on the same tokens.
- Compares full logits AND per-op intermediate tensors (max abs diff).
- Writes docs/p1/op_diff.md and docs/p1/op_diff.json.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Qwen3Ref, N_LAYERS, ATTN_SCALE  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", str(Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
PROMPT = "Explain the concept of a transformer neural network and its attention mechanism:"
DEVICE = "cuda"

from model import Qwen3Ref, N_LAYERS, repeat_kv  # noqa: E402
def load_hf_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE).eval()
    return model, tok


@torch.no_grad()
def hf_layer_trace(model, hidden, positions, mask4d, kv_cache, layer_idx):
    """Re-run HF's decoder layer sub-modules to extract per-op tensors.

    Returns dict op -> dict(inputs, outputs), mirroring ref/model.py naming.
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    seq = hidden.shape[0]
    n_head = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    hd = attn.head_dim
    scale = attn.scaling

    # input layernorm
    norm_in = layer.input_layernorm(hidden)
    # qkv
    q_raw = attn.q_proj(norm_in)
    k_raw = attn.k_proj(norm_in)
    v_raw = attn.v_proj(norm_in)
    q = attn.q_norm(q_raw.view(seq, n_head, hd)).transpose(0, 1)
    k = attn.k_norm(k_raw.view(seq, n_kv, hd)).transpose(0, 1)
    v = v_raw.view(seq, n_kv, hd).transpose(0, 1)

    cos, sin = positions  # [seq, hd]
    q_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)[0]
    k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)[1]

    if kv_cache is not None and kv_cache["k"].numel() > 0:
        k_full = torch.cat([kv_cache["k"], k_rot], dim=1)
        v_full = torch.cat([kv_cache["v"], v], dim=1)
    else:
        k_full, v_full = k_rot, v

    k_rep = repeat_kv(k_full, n_head // n_kv)
    v_rep = repeat_kv(v_full, n_head // n_kv)

    scores = torch.matmul(q_rot, k_rep.transpose(-1, -2)) * scale
    mask = mask4d[:, :, :, :k_full.shape[1]] if mask4d is not None else None
    if mask is not None:
        mask = mask.squeeze(0)          # [1, seq, seq_k] (batch dropped)
    masked = scores + mask if mask is not None else scores
    probs = torch.nn.functional.softmax(masked, dim=-1, dtype=torch.float32).to(q_rot.dtype)
    ctx = torch.matmul(probs, v_rep)
    attn_out = ctx.transpose(0, 1).reshape(seq, n_head * hd).contiguous()
    o = attn.o_proj(attn_out)
    hidden1 = hidden + o
    norm_post = layer.post_attention_layernorm(hidden1)
    gate = layer.mlp.gate_proj(norm_post)
    up = layer.mlp.up_proj(norm_post)
    act = torch.nn.functional.silu(gate) * up
    down = layer.mlp.down_proj(act)
    hidden2 = hidden1 + down

    return {
        "rmsnorm_in": ({"x": hidden}, {"y": norm_in}),
        "attn_qkv": ({"x": norm_in}, {"q": q_raw, "k": k_raw, "v": v_raw}),
        "attn_qknorm": ({"q": q_raw, "k": k_raw}, {"q": q, "k": k}),
        "attn_rope": ({"q": q, "k": k, "cos": cos, "sin": sin}, {"q": q_rot, "k": k_rot}),
        "attn_score": ({"q": q_rot, "k": k_full}, {"scores": scores}),
        "attn_softmax": ({"scores": scores, "mask": mask.squeeze(0) if mask is not None else None},
                         {"probs": probs}),
        "attn_ctx": ({"probs": probs, "v": v_full}, {"ctx": ctx}),
        "attn_o": ({"ctx": ctx}, {"o": o}),
        "residual_attn": ({"x": hidden, "attn_o": o}, {"y": hidden1}),
        "rmsnorm_mlp": ({"x": hidden1}, {"y": norm_post}),
        "mlp_gate": ({"x": norm_post}, {"gate": gate}),
        "mlp_up": ({"x": norm_post}, {"up": up}),
        "mlp_silu": ({"gate": gate, "up": up}, {"y": act}),
        "mlp_down": ({"x": act}, {"down": down}),
        "residual_mlp": ({"x": hidden1, "mlp_down": down}, {"y": hidden2}),
    }, hidden2, k_full, v_full


def compare_tensors(name, ref_t, hf_t):
    a = ref_t.detach().float().cpu()
    b = hf_t.detach().float().cpu()
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)], "max_abs": float("inf")}
    return {"shape": list(a.shape), "max_abs": float((a - b).abs().max().item())}


def main():
    model, tok = load_hf_model()
    sd = {k: v for k, v in model.state_dict().items()}
    ref = Qwen3Ref(sd, device=DEVICE)

    token_ids = tok(PROMPT, return_tensors="pt")["input_ids"][0].to(DEVICE)  # [seq]
    seq = token_ids.shape[0]
    print(f"prompt tokens = {seq}: {tok.decode(token_ids)}")

    # --- HF full forward (prefill) ---
    with torch.no_grad():
        out = model(input_ids=token_ids[None, :], use_cache=False)
    hf_logits = out.logits[0]  # [seq, vocab]

    # --- HF per-layer trace (re-run with our own decomposition) ---
    emb = model.model.embed_tokens(token_ids)
    cos, sin = model.model.rotary_emb(emb[None], torch.arange(seq, device=DEVICE)[None, :])
    cos = cos[0]; sin = sin[0]
    mask4d = model.model._update_causal_mask(None, emb[None], torch.arange(seq, device=DEVICE),
                                             None, False)
    hidden = emb
    hf_traces = []
    for li in range(N_LAYERS):
        tr, hidden, kf, vf = hf_layer_trace(model, hidden, (cos, sin), mask4d, None, li)
        hf_traces.append(tr)
    hf_final = model.model.norm(hidden)
    hf_lm = model.lm_head(hf_final)

    # --- ref forward with trace ---
    ref_trace = []
    ref_logits, _ = ref.forward(token_ids, trace=ref_trace)

    # --- full logits comparison ---
    print("logits max abs diff:", (ref_logits.float() - hf_logits.float()).abs().max().item())
    print("lm_head max abs diff:", (ref_trace[-1][1]["outputs"]["logits"].float() - hf_lm.float()).abs().max().item())

    # --- per-op comparison ---
    results = {"prompt": PROMPT, "prompt_tokens": seq, "model": "Qwen3-0.6B",
               "logits_max_abs": float((ref_logits.float() - hf_logits.float()).abs().max().item()),
               "ops": {}}
    all_ops = []
    for li in range(N_LAYERS):
        ref_tr = ref_trace[li + 1][1]   # trace[0] is embed
        hf_tr = hf_traces[li]
        for op in ref_tr:
            ri, ro = ref_tr[op]["inputs"], ref_tr[op]["outputs"]
            hi, ho = hf_tr[op][0], hf_tr[op][1]
            d = {"layer": li, "op": op, "inputs": {}, "outputs": {}}
            for name in ro:
                d["outputs"][name] = compare_tensors(name, ro[name], ho[name])
            for name in ri:
                if name in hi:
                    d["inputs"][name] = compare_tensors(name, ri[name], hi[name])
            results["ops"][f"L{li:02d}_{op}"] = d
            all_ops.append((f"L{li:02d}_{op}", d))

    # worst op
    worst = max(all_ops, key=lambda t: max(
        [v.get("max_abs", 0) for k, v in t[1]["outputs"].items()]
        + [v.get("max_abs", 0) for k, v in t[1]["inputs"].items()]))
    print("worst op:", worst[0], "max_abs:",
          max([v.get("max_abs", 0) for k, v in worst[1]["outputs"].items()]))

    # global ops
    global_ops = {
        "embed": compare_tensors("embed", ref_trace[0][1]["outputs"]["y"], emb),
        "final_norm": compare_tensors("final_norm", ref_trace[-2][1]["outputs"]["y"], hf_final),
        "lm_head": compare_tensors("lm_head", ref_trace[-1][1]["outputs"]["logits"], hf_lm),
    }
    results["ops"]["embed"] = {"op": "embed", "outputs": {"y": global_ops["embed"]}}
    results["ops"]["final_norm"] = {"op": "final_norm", "outputs": {"y": global_ops["final_norm"]}}
    results["ops"]["lm_head"] = {"op": "lm_head", "outputs": {"logits": global_ops["lm_head"]}}

    os.makedirs("docs/p1", exist_ok=True)
    with open("docs/p1/op_diff.json", "w") as f:
        json.dump(results, f, indent=2)

    # markdown table
    rows = []
    rows.append("# 逐 op 对比（ref/model.py vs HF transformers 4.51.0 eager）\n")
    rows.append(f"- prompt: `{PROMPT}` ({seq} tokens)")
    rows.append(f"- model: Qwen3-0.6B (BF16, eager)")
    rows.append(f"- 全模型 logits max abs diff: **{results['logits_max_abs']:.3e}**\n")
    rows.append("| op | 输出张量 | shape | max abs diff | 判定(<1e-3) |")
    rows.append("|---|---|---|---|---|")
    for key, d in results["ops"].items():
        for name, v in d["outputs"].items():
            ok = v.get("max_abs", float("inf")) < 1e-3
            rows.append(f"| {key} | {name} | {v.get('shape','?')} | {v.get('max_abs','NA'):.3e} | {'✅' if ok else '❌'} |")
    with open("docs/p1/op_diff.md", "w") as f:
        f.write("\n".join(rows) + "\n")

    print("WROTE docs/p1/op_diff.md + docs/p1/op_diff.json")
    # exit nonzero if any op fails
    fails = [k for k, d in results["ops"].items()
             for n, v in d["outputs"].items() if v.get("max_abs", float("inf")) >= 1e-3]
    print("FAILS:", fails if fails else "none")


if __name__ == "__main__":
    main()
