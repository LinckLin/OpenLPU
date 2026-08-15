"""Qwen3-0.6B model config — authoritative numbers from
docs/spec-src/01b-target-model-0.6b.md (verified against official config.json
+ safetensors header, 311 tensors all BF16).
"""

from __future__ import annotations

# Target identity
TARGET = "qcore-v1"

# dtype flag codes (00-container §2 flags bits[1:0])
DTYPE_FLAGS = {"BF16": 0, "FP16": 1, "INT8": 2, "INT4": 3}

# Qwen3-0.6B verified model card (§1)
MODEL_CFG = {
    "model": "Qwen3-0.6B",
    "hidden": 1024,
    "layers": 28,
    "q_heads": 16,
    "kv_heads": 8,          # GQA 2:1
    "head_dim": 128,
    "intermediate": 3072,
    "vocab": 151936,
    "rope_theta": 1_000_000.0,
    "rms_eps": 1e-6,
    "qk_norm": True,        # q_norm / k_norm [128] present, applied before RoPE
    "max_pos": 40960,
}

# Weight tensor shapes (01b §2). The per-layer six projection classes are:
#   qkv (fused QKV, 4096x1024 = q 2048 + k 1024 + v 1024),
#   o (1024x2048), gate (3072x1024), up (3072x1024), down (1024x3072),
#   plus global lm_head (151936x1024).
# The checkpoint physically stores q_proj/k_proj/v_proj separately; qforge fuses
# them into one qkv_proj tensor (spec §3.2 C13 "QKV 融合变体").

# Quantization defaults (00-container §2 quant dict): W8A8 (INT8 weights),
# W4A16 (INT4 weights + BF16 activations), BF16 (unquantized reference).
QUANT_INT8 = {"mode": "W8A8", "group": 128, "sym": True}
QUANT_INT4 = {"mode": "W4A16", "group": 128, "sym": True}
QUANT_BF16 = {"mode": "BF16", "group": 128, "sym": True}

# Compile-time activation-scale constant used for the skeleton qbin (the
# executor's dequant folds activation scale x weight scale into a single CD
# scale; runtime calibration lands in P5/qrun). Unit here means the stored
# scale equals the weight scale.
ACTIVATION_SCALE_DEFAULT = 1.0
