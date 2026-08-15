"""qforge CLI: qforge compile Qwen/Qwen3-0.6B --target qcore-v1 --dtype int8.

Pipeline: config parse -> build graph -> safetensors load -> W8A8 quantize
(weight side per-128-K-group; activation scale compile-time constant) or BF16
pack (raw BF16 weights, scales omitted) -> weight repack -> tiling (N<=128,
K streamed) -> scheduling -> .qbin (00-container).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, config as C, graph, build
from .safetensors import SafeTensors

DEFAULT_OUT = "qwen3-0.6b.qbin"


def resolve_model_dir(model: str, model_dir: str | None) -> str:
    """Resolve a HF model id to a local directory containing config.json +
    model.safetensors."""
    if model_dir:
        return model_dir
    name = model.replace("/", "--")
    base = os.path.expanduser(f"~/.cache/huggingface/hub/models--{name}")
    if os.path.isdir(base):
        # standard layout: snapshots/<ref>/ ; fallback: files directly in base
        for cand in (base,):
            if os.path.isfile(os.path.join(cand, "model.safetensors")):
                return cand
        refs = os.path.join(base, "refs")
        if os.path.isdir(refs):
            for ref in os.listdir(refs):
                with open(os.path.join(refs, ref)) as f:
                    commit = f.read().strip()
                snap = os.path.join(base, "snapshots", commit)
                if os.path.isfile(os.path.join(snap, "model.safetensors")):
                    return snap
    raise FileNotFoundError(
        f"cannot resolve local HF cache for {model!r}; pass --model-dir")


def load_config(model_dir: str) -> dict:
    """Parse config.json and cross-check against the verified 0.6B card."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    card = C.MODEL_CFG
    checks = {
        "hidden_size": card["hidden"],
        "num_hidden_layers": card["layers"],
        "num_attention_heads": card["q_heads"],
        "num_key_value_heads": card["kv_heads"],
        "head_dim": card["head_dim"],
        "intermediate_size": card["intermediate"],
        "vocab_size": card["vocab"],
    }
    for key, expected in checks.items():
        got = cfg.get(key)
        if got != expected:
            raise ValueError(
                f"config.json {key}={got!r} != verified card {expected}; "
                f"refusing to compile an unverified config")
    return cfg


def cmd_compile(args: argparse.Namespace) -> int:
    dtype = args.dtype.lower()
    if dtype in ("int8", "w8a8"):
        quant_j = C.QUANT_INT8
        dtype_label = "W8A8"
    elif dtype == "bf16":
        quant_j = C.QUANT_BF16
        dtype_label = "BF16"
    else:
        print(f"qforge: dtype {args.dtype!r} not implemented "
              f"(int8/W8A8, bf16); INT4/FP8 land in P6", file=sys.stderr)
        return 2
    if args.target != C.TARGET:
        print(f"qforge: target {args.target!r} != {C.TARGET}", file=sys.stderr)
        return 2

    model_dir = resolve_model_dir(args.model, args.model_dir)
    cfg_json = load_config(model_dir)
    st_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(st_path):
        print(f"qforge: missing {st_path}", file=sys.stderr)
        return 2

    print(f"[qforge] model={args.model} target={C.TARGET} dtype={dtype_label} "
          f"(group=128, sym) -> {args.output}")
    print(f"[qforge] safetensors: {st_path}")

    st = SafeTensors.open(st_path)
    n_tensors = len(st.keys())
    print(f"[qforge] loaded {n_tensors} tensors (header parsed)")

    projs = graph.build_graph()
    print(f"[qforge] graph: {len(projs)} projections "
          f"({C.MODEL_CFG['layers']} layers x 5 + lm_head)")

    qb = build.build_model_qbin(args.output, st, quant_j,
                                C.ACTIVATION_SCALE_DEFAULT)
    total_weight = sum(len(t.data) for t in qb.tensors)
    total_scale = sum(len(t.scales) for t in qb.tensors if t.scales is not None)
    print(f"[qforge] wrote {args.output}: {len(qb.tensors)} tensors, "
          f"weights {total_weight} B, scales {total_scale} B, "
          f"pf {qb.header['pf_len']//16} inst, "
          f"dc {qb.header['dc_len']//16} inst")
    print(f"[qforge] container: magic={qb.magic!r} version={qb.version} "
          f"flags=0x{qb.flags:X} header_size={qb.header_size}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qforge",
                                description="QCore HF -> .qbin compiler")
    p.add_argument("--version", action="version",
                   version=f"qforge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="compile a HF model to .qbin")
    c.add_argument("model", help="HF model id (e.g. Qwen/Qwen3-0.6B)")
    c.add_argument("--target", default=C.TARGET, help="target (qcore-v1)")
    c.add_argument("--dtype", default="int8", help="weight dtype (int8|bf16)")
    c.add_argument("-o", "--output", default=DEFAULT_OUT,
                   help="output .qbin path")
    c.add_argument("--model-dir", default=None,
                   help="local dir with config.json + model.safetensors")
    c.set_defaults(func=cmd_compile)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
