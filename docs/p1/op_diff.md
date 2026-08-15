# 逐 op 对比（ref/model.py vs HF transformers 4.51.0 eager）

- prompt: `Explain the concept of a transformer neural network and its attention mechanism:` (14 tokens)
- model: Qwen3-0.6B (BF16, eager)
- 全模型 logits max abs diff: **0.000e+00**

| op | 输出张量 | shape | max abs diff | 判定(<1e-3) |
|---|---|---|---|---|
| L00_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L00_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L00_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L00_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L00_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L00_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L00_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L00_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L00_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L00_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L00_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L00_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L00_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L00_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L00_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L00_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L00_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L00_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L00_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L01_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L01_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L01_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L01_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L01_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L01_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L01_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L01_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L01_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L01_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L01_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L01_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L01_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L01_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L01_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L01_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L01_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L01_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L01_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L02_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L02_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L02_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L02_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L02_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L02_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L02_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L02_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L02_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L02_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L02_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L02_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L02_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L02_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L02_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L02_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L02_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L02_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L02_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L03_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L03_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L03_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L03_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L03_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L03_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L03_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L03_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L03_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L03_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L03_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L03_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L03_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L03_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L03_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L03_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L03_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L03_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L03_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L04_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L04_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L04_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L04_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L04_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L04_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L04_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L04_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L04_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L04_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L04_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L04_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L04_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L04_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L04_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L04_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L04_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L04_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L04_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L05_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L05_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L05_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L05_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L05_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L05_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L05_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L05_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L05_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L05_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L05_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L05_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L05_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L05_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L05_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L05_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L05_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L05_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L05_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L06_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L06_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L06_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L06_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L06_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L06_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L06_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L06_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L06_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L06_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L06_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L06_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L06_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L06_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L06_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L06_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L06_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L06_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L06_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L07_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L07_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L07_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L07_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L07_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L07_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L07_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L07_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L07_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L07_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L07_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L07_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L07_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L07_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L07_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L07_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L07_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L07_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L07_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L08_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L08_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L08_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L08_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L08_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L08_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L08_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L08_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L08_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L08_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L08_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L08_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L08_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L08_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L08_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L08_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L08_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L08_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L08_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L09_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L09_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L09_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L09_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L09_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L09_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L09_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L09_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L09_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L09_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L09_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L09_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L09_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L09_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L09_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L09_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L09_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L09_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L09_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L10_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L10_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L10_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L10_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L10_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L10_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L10_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L10_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L10_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L10_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L10_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L10_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L10_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L10_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L10_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L10_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L10_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L10_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L10_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L11_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L11_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L11_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L11_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L11_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L11_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L11_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L11_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L11_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L11_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L11_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L11_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L11_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L11_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L11_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L11_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L11_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L11_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L11_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L12_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L12_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L12_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L12_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L12_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L12_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L12_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L12_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L12_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L12_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L12_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L12_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L12_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L12_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L12_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L12_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L12_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L12_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L12_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L13_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L13_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L13_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L13_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L13_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L13_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L13_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L13_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L13_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L13_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L13_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L13_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L13_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L13_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L13_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L13_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L13_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L13_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L13_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L14_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L14_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L14_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L14_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L14_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L14_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L14_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L14_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L14_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L14_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L14_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L14_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L14_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L14_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L14_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L14_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L14_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L14_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L14_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L15_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L15_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L15_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L15_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L15_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L15_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L15_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L15_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L15_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L15_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L15_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L15_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L15_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L15_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L15_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L15_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L15_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L15_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L15_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L16_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L16_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L16_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L16_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L16_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L16_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L16_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L16_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L16_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L16_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L16_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L16_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L16_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L16_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L16_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L16_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L16_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L16_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L16_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L17_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L17_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L17_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L17_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L17_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L17_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L17_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L17_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L17_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L17_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L17_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L17_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L17_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L17_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L17_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L17_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L17_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L17_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L17_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L18_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L18_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L18_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L18_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L18_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L18_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L18_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L18_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L18_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L18_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L18_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L18_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L18_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L18_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L18_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L18_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L18_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L18_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L18_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L19_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L19_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L19_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L19_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L19_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L19_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L19_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L19_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L19_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L19_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L19_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L19_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L19_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L19_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L19_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L19_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L19_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L19_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L19_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L20_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L20_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L20_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L20_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L20_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L20_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L20_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L20_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L20_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L20_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L20_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L20_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L20_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L20_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L20_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L20_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L20_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L20_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L20_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L21_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L21_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L21_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L21_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L21_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L21_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L21_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L21_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L21_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L21_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L21_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L21_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L21_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L21_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L21_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L21_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L21_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L21_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L21_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L22_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L22_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L22_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L22_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L22_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L22_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L22_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L22_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L22_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L22_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L22_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L22_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L22_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L22_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L22_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L22_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L22_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L22_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L22_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L23_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L23_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L23_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L23_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L23_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L23_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L23_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L23_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L23_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L23_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L23_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L23_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L23_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L23_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L23_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L23_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L23_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L23_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L23_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L24_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L24_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L24_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L24_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L24_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L24_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L24_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L24_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L24_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L24_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L24_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L24_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L24_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L24_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L24_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L24_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L24_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L24_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L24_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L25_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L25_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L25_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L25_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L25_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L25_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L25_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L25_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L25_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L25_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L25_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L25_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L25_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L25_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L25_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L25_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L25_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L25_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L25_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L26_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L26_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L26_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L26_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L26_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L26_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L26_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L26_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L26_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L26_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L26_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L26_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L26_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L26_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L26_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L26_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L26_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L26_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L26_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L27_rmsnorm_in | y | [14, 1024] | 0.000e+00 | ✅ |
| L27_attn_qkv | q | [14, 2048] | 0.000e+00 | ✅ |
| L27_attn_qkv | k | [14, 1024] | 0.000e+00 | ✅ |
| L27_attn_qkv | v | [14, 1024] | 0.000e+00 | ✅ |
| L27_attn_qknorm | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L27_attn_qknorm | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L27_attn_rope | q | [16, 14, 128] | 0.000e+00 | ✅ |
| L27_attn_rope | k | [8, 14, 128] | 0.000e+00 | ✅ |
| L27_attn_score | scores | [16, 14, 14] | 0.000e+00 | ✅ |
| L27_attn_softmax | probs | [16, 14, 14] | 0.000e+00 | ✅ |
| L27_attn_ctx | ctx | [16, 14, 128] | 0.000e+00 | ✅ |
| L27_attn_o | o | [14, 1024] | 0.000e+00 | ✅ |
| L27_residual_attn | y | [14, 1024] | 0.000e+00 | ✅ |
| L27_rmsnorm_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| L27_mlp_gate | gate | [14, 3072] | 0.000e+00 | ✅ |
| L27_mlp_up | up | [14, 3072] | 0.000e+00 | ✅ |
| L27_mlp_silu | y | [14, 3072] | 0.000e+00 | ✅ |
| L27_mlp_down | down | [14, 1024] | 0.000e+00 | ✅ |
| L27_residual_mlp | y | [14, 1024] | 0.000e+00 | ✅ |
| embed | y | [14, 1024] | 0.000e+00 | ✅ |
| final_norm | y | [14, 1024] | 0.000e+00 | ✅ |
| lm_head | logits | [14, 151936] | 0.000e+00 | ✅ |
