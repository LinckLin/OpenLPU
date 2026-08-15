"""Golden-trace I/O helpers (J1 schema).

BF16 storage: numpy 2.2.6 has no native bfloat16 (ml_dtypes 0.5.4 registered dtype
does not round-trip through npz — it reloads as |V2). Per J1 fallback rule we store
every bf16 activation/weight as its exact fp32 representation with
dtype_np="float32(bf16)", dtype_code=0. y_ref is pure fp32 (no dtype_code).

dtype code map (frozen J1): bfloat16=0, float16=1, int8=2, int4=3, int32=4, int16=5;
float32 is comparison-only and carries no code.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

DTYPE_CODES = {"bfloat16": 0, "float16": 1, "int8": 2, "int4": 3, "int32": 4, "int16": 5}


def bf16_to_np(t: torch.Tensor) -> np.ndarray:
    """bf16 tensor -> exact fp32 numpy (lossless: bf16 subset of fp32)."""
    return t.detach().float().cpu().numpy()


def fp32_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def int32_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.int32)


def bf16_field(shape: list[int]) -> dict:
    return {"shape": shape, "dtype_np": "float32(bf16)", "dtype_code": DTYPE_CODES["bfloat16"]}


def fp32_field(shape: list[int]) -> dict:
    return {"shape": shape, "dtype_np": "float32", "dtype_code": None}


def int32_field(shape: list[int]) -> dict:
    return {"shape": shape, "dtype_np": "int32", "dtype_code": DTYPE_CODES["int32"]}


def write_op_dir(dirpath: str, meta: dict):
    """Write inputs.npz / outputs.npz / (weights.npz) / meta.json from a meta dict.

    meta keys:
      op, layer, mode, params, weights_ref,
      inputs : {name: (np_array, field)}, outputs : {name: (np_array, field)},
      weights : optional {name: (np_array, field)} -> written to weights.npz
    """
    os.makedirs(dirpath, exist_ok=True)

    inp = {}
    out = {}
    for name, (arr, field) in meta["inputs"].items():
        inp[name] = arr
    for name, (arr, field) in meta["outputs"].items():
        out[name] = arr

    np.savez(os.path.join(dirpath, "inputs.npz"), **inp)
    np.savez(os.path.join(dirpath, "outputs.npz"), **out)

    if meta.get("weights"):
        w = {name: arr for name, (arr, field) in meta["weights"].items()}
        np.savez(os.path.join(dirpath, "weights.npz"), **w)

    meta_json = {
        "op": meta["op"],
        "layer": meta["layer"],
        "mode": meta["mode"],
        "inputs": {name: field for name, (arr, field) in meta["inputs"].items()},
        "outputs": {name: field for name, (arr, field) in meta["outputs"].items()},
        "params": meta["params"],
        "weights_ref": meta["weights_ref"],
    }
    if meta.get("weights"):
        meta_json["weights"] = {name: field for name, (arr, field) in meta["weights"].items()}
    with open(os.path.join(dirpath, "meta.json"), "w") as f:
        json.dump(meta_json, f, indent=2)
