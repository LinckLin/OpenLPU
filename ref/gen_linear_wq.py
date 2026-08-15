"""Generate M2a join-point golden: linear_wq_pf (seq=128) and linear_wq_dc (seq=1).

Wq = q_proj.weight [2048, 1024] (bf16). x = real layer-0 q_proj input activation.
y (bf16) = F.linear(x, Wq); y_ref (fp32) = matmul(x.float(), Wq.float().T).

Model path / device are parameterized:
  MODEL_DIR  (default ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B)
  DEVICE     (default cuda)

Writes golden/qwen3-0.6b/linear_wq_pf/ and linear_wq_dc/.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Qwen3Ref, HIDDEN  # noqa: E402
from golden import bf16_to_np, fp32_to_np, bf16_field, fp32_field, write_op_dir  # noqa: E402

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
GOLDEN_ROOT = "golden/qwen3-0.6b"
DEVICE = os.environ.get("DEVICE", "cuda")

# fixed 128-token prompt (deterministic)
PF_TEXT = (
    "The transformer is a neural network architecture that relies entirely on the attention "
    "mechanism to draw global dependencies between input and output sequences. Unlike recurrent "
    "networks that process tokens one step at a time, the transformer attends to all positions "
    "simultaneously, which makes it highly parallelizable and efficient to train at scale. "
    "Each layer combines multi-head self-attention with position-wise feed-forward networks, "
    "layer normalization, and residual connections. Attention scores are computed by taking the "
    "scaled dot product of query and key vectors, followed by a softmax over keys, and finally "
    "weighting value vectors. Rotary positional embeddings inject relative position information "
    "without learned position matrices. Query, key, and value projections are followed by a "
    "per-head normalization step before the rotary transformation is applied. The grouped query "
    "attention mechanism lets several query heads share a single key-value head, reducing memory "
    "traffic during decoding while preserving generation quality across long sequences. "
)


def generate(model, tok, ref, device):
    """Write linear_wq_pf / linear_wq_dc using an already-loaded model."""
    Wq = ref.layers[0].q_proj                                   # [2048, 1024]

    # --- prefill seq=128: x = layer0 rmsnorm_in output ---
    ids = tok(PF_TEXT, return_tensors="pt")["input_ids"][0].to(device)
    ids = ids[:128]
    assert ids.shape[0] == 128, f"prompt only {ids.shape[0]} tokens"
    trace = []
    ref.forward(ids, trace=trace)
    x_pf = trace[1][1]["rmsnorm_in"]["outputs"]["y"]            # [128, 1024]

    y_pf = F.linear(x_pf, Wq)                                   # [128, 2048] bf16
    y_ref_pf = torch.matmul(x_pf.float(), Wq.float().T)         # [128, 2048] fp32

    write_op_dir(f"{GOLDEN_ROOT}/linear_wq_pf", {
        "op": "linear_wq", "layer": 0, "mode": "PF",
        "inputs": {"x": (bf16_to_np(x_pf), bf16_field([128, HIDDEN]))},
        "outputs": {"y": (bf16_to_np(y_pf), bf16_field([128, 2048])),
                    "y_ref": (fp32_to_np(y_ref_pf), fp32_field([128, 2048]))},
        "params": {"M": 128, "N": 2048, "K": HIDDEN, "seq": 128, "n_tiles_N": 16},
        "weights_ref": ["model.layers.0.self_attn.q_proj.weight"],
        "weights": {"wq": (bf16_to_np(Wq), bf16_field([2048, HIDDEN]))},
    })

    # --- decode seq=1: x = layer0 rmsnorm_in output of the next token ---
    logits_pf, cache = ref.forward(ids)
    next_id = logits_pf[-1].argmax(dim=-1)                     # greedy next token
    trace_dc = []
    ref.forward(next_id[None], cache_pos=128, kv_cache=cache, trace=trace_dc)
    x_dc = trace_dc[1][1]["rmsnorm_in"]["outputs"]["y"]         # [1, 1024]

    y_dc = F.linear(x_dc, Wq)                                   # [1, 2048] bf16
    y_ref_dc = torch.matmul(x_dc.float(), Wq.float().T)         # [1, 2048] fp32

    write_op_dir(f"{GOLDEN_ROOT}/linear_wq_dc", {
        "op": "linear_wq", "layer": 0, "mode": "DC",
        "inputs": {"x": (bf16_to_np(x_dc), bf16_field([1, HIDDEN]))},
        "outputs": {"y": (bf16_to_np(y_dc), bf16_field([1, 2048])),
                    "y_ref": (fp32_to_np(y_ref_dc), fp32_field([1, 2048]))},
        "params": {"M": 1, "N": 2048, "K": HIDDEN, "seq": 1, "n_tiles_N": 16},
        "weights_ref": ["model.layers.0.self_attn.q_proj.weight"],
        "weights": {"wq": (bf16_to_np(Wq), bf16_field([2048, HIDDEN]))},
    })

    # sanity: y_ref == fp32(y) for the bf16-rounded output (within bf16 rounding)
    for name, y, yr in [("pf", y_pf, y_ref_pf), ("dc", y_dc, y_ref_dc)]:
        d = (y.float() - yr).abs().max().item()
        print(f"{name}: y bf16 vs y_ref fp32 max abs diff = {d:.3e}")

    print("WROTE linear_wq_pf + linear_wq_dc")


def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device).eval()
    ref = Qwen3Ref({k: v for k, v in model.state_dict().items()}, device=args.device)
    generate(model, tok, ref, args.device)


if __name__ == "__main__":
    main()
