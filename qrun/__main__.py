"""qrun CLI: `python -m qrun <qbin> --prompt "..." [--ctx N] [--max-new N]
[--dtype int8|int4|bf16] [--weights-from-hf] [--model-dir DIR] [--device DEV]`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def resolve_model_dir(model: str, model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    name = model.replace("/", "--")
    base = os.path.expanduser(f"~/.cache/huggingface/hub/models--{name}")
    if os.path.isdir(base):
        if os.path.isfile(os.path.join(base, "model.safetensors")):
            return base
        for snap in (os.path.join(base, "snapshots"), base):
            if os.path.isdir(snap):
                for d in sorted(os.listdir(snap)):
                    p = os.path.join(snap, d)
                    if os.path.isfile(os.path.join(p, "model.safetensors")):
                        return p
    raise FileNotFoundError(
        f"cannot resolve HF cache for {model!r}; pass --model-dir")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qrun")
    ap.add_argument("qbin")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ctx", type=int, default=None, help="context length hint")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--dtype", choices=["int8", "int4", "bf16"], default="int8")
    ap.add_argument("--weights-from-hf", action="store_true",
                    help="BF16: load weights from safetensors instead of qbin")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from qrun.engine import build_engine
    from qrun.reference import load_hf

    model_dir = resolve_model_dir(args.model, args.model_dir)
    tok, hf, ref = load_hf(model_dir, args.device)

    t0 = time.time()
    engine = build_engine(args.qbin, model_dir, args.dtype, tokenizer=tok,
                          ref=ref, weights_from_hf=args.weights_from_hf)
    t_build = time.time() - t0

    prompt_ids = tok(args.prompt, return_tensors="pt")["input_ids"][0].numpy()
    bootstrap = prompt_ids.shape[0] > 128 or (
        args.ctx is not None and prompt_ids.shape[0] < args.ctx)
    if bootstrap and args.ctx is not None and prompt_ids.shape[0] < args.ctx:
        import numpy as np
        reps = (args.ctx + prompt_ids.shape[0]) // prompt_ids.shape[0]
        prompt_ids = np.tile(prompt_ids, reps)[:args.ctx]

    t1 = time.time()
    new_tokens, logits_list = engine.generate(prompt_ids, args.max_new,
                                              bootstrap=bootstrap)
    t_gen = time.time() - t1

    text = tok.decode(new_tokens, skip_special_tokens=True)
    print(f"[qrun] dtype={args.dtype} prompt_tokens={prompt_ids.shape[0]} "
          f"build={t_build:.1f}s generate={t_gen:.1f}s")
    print(f"[qrun] new tokens: {new_tokens}")
    print(f"[qrun] decoded: {text}")
    return 0



if __name__ == "__main__":
    sys.exit(main())
