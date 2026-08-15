from pathlib import Path
"""HF baseline greedy decode (20 tokens) + ref greedy decode, token-by-token compare.

Writes docs/p1/baseline_tokens.txt (HF baseline) and
docs/p1/greedy_match.json / .md (HF vs ref token-by-token comparison).
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Qwen3Ref  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", str(Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
PROMPT = "Explain the concept of a transformer neural network and its attention mechanism:"
N_TOKENS = 20
DEVICE = "cuda"


@torch.no_grad()
def hf_greedy(model, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    gen = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    new_ids = gen[0][ids.shape[1]:]
    return ids[0], new_ids


@torch.no_grad()
def ref_greedy(ref, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].to(DEVICE)
    logits, cache = ref.forward(ids)
    new_tokens = []
    for _ in range(n):
        tok_id = logits[-1].argmax(dim=-1)          # [scalar]
        new_tokens.append(int(tok_id.item()))
        logits, cache = ref.forward(tok_id[None], cache_pos=ids.shape[0] + len(new_tokens) - 1,
                                    kv_cache=cache)
    return ids, torch.tensor(new_tokens, device=DEVICE)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE).eval()

    hf_ids, hf_new = hf_greedy(model, tok, PROMPT, N_TOKENS)
    hf_text = tok.decode(hf_new, skip_special_tokens=True)
    print("HF new tokens:", hf_new.tolist())
    print("HF text:", hf_text)

    ref = Qwen3Ref({k: v for k, v in model.state_dict().items()}, device=DEVICE)
    ref_ids, ref_new = ref_greedy(ref, tok, PROMPT, N_TOKENS)
    ref_text = tok.decode(ref_new, skip_special_tokens=True)
    print("ref new tokens:", ref_new.tolist())

    match = torch.equal(hf_new, ref_new)
    print("token-by-token match:", match)
    if not match:
        for i, (a, b) in enumerate(zip(hf_new, ref_new)):
            if a != b:
                print(f"  first mismatch at step {i}: hf={a} ref={b}")
                break

    os.makedirs("docs/p1", exist_ok=True)
    with open("docs/p1/baseline_tokens.txt", "w") as f:
        f.write(f"model: Qwen/Qwen3-0.6B (BF16, eager)\n")
        f.write(f"prompt: {PROMPT}\n")
        f.write(f"num new tokens: {N_TOKENS}\n")
        f.write(f"prompt token ids: {hf_ids.tolist()}\n")
        f.write(f"new token ids (HF greedy): {hf_new.tolist()}\n")
        f.write(f"decoded: {hf_text}\n")

    results = {
        "model": "Qwen3-0.6B", "prompt": PROMPT, "n_tokens": N_TOKENS,
        "hf_tokens": hf_new.tolist(), "ref_tokens": ref_new.tolist(),
        "hf_text": hf_text, "ref_text": ref_text,
        "token_by_token_match": bool(match),
    }
    with open("docs/p1/greedy_match.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("docs/p1/greedy_match.md", "w") as f:
        f.write("# greedy decode 对比（HF vs ref/model.py）\n\n")
        f.write(f"- prompt: `{PROMPT}`\n- 新 token 数: {N_TOKENS}\n")
        f.write(f"- **逐 token 一致: {'✅ 是' if match else '❌ 否'}**\n\n")
        f.write("| step | HF | ref |\n|---|---|---|\n")
        for i, (a, b) in enumerate(zip(hf_new, ref_new)):
            f.write(f"| {i} | {a.item()} | {b.item()} |\n")
        f.write(f"\nHF 解码文本: `{hf_text}`\n")

    print("WROTE docs/p1/baseline_tokens.txt + greedy_match.json/.md")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
