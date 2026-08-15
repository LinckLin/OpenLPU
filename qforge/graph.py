"""Graph: the full-model projection table (6 classes x 28 layers + lm_head).

Each Projection is one linear layer y = x @ W^T with PyTorch [out, in] weight.
The six classes are: qkv (fused QKV 4096x1024), o (1024x2048), gate (3072x1024),
up (3072x1024), down (1024x3072), lm_head (151936x1024).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config as C

# safetensors weight-key templates (layer is filled per projection)
_QKV_KEYS = (
    "model.layers.{L}.self_attn.q_proj.weight",
    "model.layers.{L}.self_attn.k_proj.weight",
    "model.layers.{L}.self_attn.v_proj.weight",
)


@dataclass
class Projection:
    """One linear projection (one weight tensor in the .qbin)."""
    kind: str                 # 'qkv' | 'o' | 'gate' | 'up' | 'down' | 'lm_head'
    name: str                 # qbin tensor name
    layer: int | None         # 0..27, or None for lm_head
    N: int                    # output dim (weight rows)
    K: int                    # input dim (weight cols)
    weight_keys: list[str]    # safetensors keys (3 for fused qkv, else 1)


def _layer_tables() -> list[Projection]:
    """28 layers x 5 per-layer projections (qkv/o/gate/up/down)."""
    out = []
    H = C.MODEL_CFG["hidden"]          # 1024
    I = C.MODEL_CFG["intermediate"]    # 3072
    for L in range(C.MODEL_CFG["layers"]):
        p = f"model.layers.{L}"
        out.append(Projection(
            kind="qkv", name=f"{p}.self_attn.qkv_proj.weight", layer=L,
            N=4 * H, K=H, weight_keys=[k.format(L=L) for k in _QKV_KEYS]))
        out.append(Projection(
            kind="o", name=f"{p}.self_attn.o_proj.weight", layer=L,
            N=H, K=2 * H, weight_keys=[f"{p}.self_attn.o_proj.weight"]))
        out.append(Projection(
            kind="gate", name=f"{p}.mlp.gate_proj.weight", layer=L,
            N=I, K=H, weight_keys=[f"{p}.mlp.gate_proj.weight"]))
        out.append(Projection(
            kind="up", name=f"{p}.mlp.up_proj.weight", layer=L,
            N=I, K=H, weight_keys=[f"{p}.mlp.up_proj.weight"]))
        out.append(Projection(
            kind="down", name=f"{p}.mlp.down_proj.weight", layer=L,
            N=H, K=I, weight_keys=[f"{p}.mlp.down_proj.weight"]))
    return out


def build_graph() -> list[Projection]:
    """Full 0.6B graph: 28 x 5 per-layer projections + lm_head (141 tensors)."""
    projs = _layer_tables()
    projs.append(Projection(
        kind="lm_head", name="lm_head.weight", layer=None,
        N=C.MODEL_CFG["vocab"], K=C.MODEL_CFG["hidden"],
        weight_keys=["lm_head.weight"]))
    return projs


# The six projection classes (used by M3 per-class quant-error verification)
CLASS_NAMES = ("qkv", "o", "gate", "up", "down", "lm_head")


def class_shapes(kind: str) -> tuple[int, int]:
    """Return (N, K) for a projection class."""
    H = C.MODEL_CFG["hidden"]
    I = C.MODEL_CFG["intermediate"]
    return {
        "qkv": (4 * H, H),
        "o": (H, 2 * H),
        "gate": (I, H),
        "up": (I, H),
        "down": (H, I),
        "lm_head": (C.MODEL_CFG["vocab"], H),
    }[kind]
