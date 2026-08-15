# 01 — 目标模型卡 (Qwen3-8B dense)

> 数据来源: Qwen/Qwen3-8B 官方 config.json (经 hf-mirror, etag 与官方 oid 一致)、
> model.safetensors.index.json (471 张量, total_size 逐字节核验)、官方 README、
> transformers v4.51.0 modeling_qwen3.py。

> **口径**：本卡为 **8B 设计上限**（架构按此尺寸设计）。验证模型 Qwen3-0.6B 的模型卡见
> `01b-target-model-0.6b.md`，0.6B 相关数值以其为准。

## 1. 关键结构参数 (契约第 10 条逐项核验: 12/12 命中, 0 硬偏差)

| 参数 | 值 | 出处 |
|------|-----|------|
| hidden_size | 4096 | config |
| num_hidden_layers | 36 | config |
| num_attention_heads | 32 | config |
| num_key_value_heads | 8 (GQA 4:1) | config |
| head_dim | 128 | config |
| intermediate_size | 12288 | config |
| vocab_size | 151936 | config |
| rope_theta | 1000000.0 | config |
| rms_norm_eps | 1e-6 | config |
| QK-norm | 无条件启用, per-head (128), **RoPE 之前** | modeling_qwen3.py (非 config 字段) |
| 权重 dtype | bfloat16 | config + index total_size |
| max_position_embeddings | 40960 (README 原生上下文 32768) | config |
| tie_word_embeddings | false → lm_head 独立权重 | config |
| attention_bias / MLP bias | 全无 | config + index (471 张量无 .bias) |
| sliding_window | null / 禁用, 36 层全 full attention | config |
| rope_scaling | null (YaRN 131K 为可选手工改动) | config |
| hidden_act | silu (SwiGLU) | config |

## 2. 每层 GEMM 与参数量

| 算子 | 权重 [out×in] | GEMM M×N×K | 参数 | BF16 字节 |
|------|---------------|------------|------|-----------|
| q_proj | 4096×4096 | seq×4096×4096 | 16,777,216 | 33,554,432 |
| k_proj | 1024×4096 | seq×1024×4096 | 4,194,304 | 8,388,608 |
| v_proj | 1024×4096 | seq×1024×4096 | 4,194,304 | 8,388,608 |
| QKV 合并 | 6144×4096 | seq×6144×4096 | 25,165,824 | 50,331,648 |
| o_proj | 4096×4096 | seq×4096×4096 | 16,777,216 | 33,554,432 |
| gate_proj | 12288×4096 | seq×12288×4096 | 50,331,648 | 100,663,296 |
| up_proj | 12288×4096 | seq×12288×4096 | 50,331,648 | 100,663,296 |
| down_proj | 4096×12288 | seq×4096×12288 | 50,331,648 | 100,663,296 |
| MLP 合计 | — | — | 150,994,944 | 301,989,888 |
| 每层 dense 合计 | — | — | 192,937,984 | 385,875,968 |
| 每层 4×RMSNorm | in/post 4096 + q/k 128 | — | 8,448 | 16,896 |

- 36 层 dense = 6,946,071,552 参数
- embedding + lm_head = 2 × 622,329,856 = 1,244,659,712
- **总参数 = 8,190,735,360 ≈ 8.19B** (README 8.2B ✅; 非嵌入 6.95B ✅)
- **BF16 总字节 = 16,381,470,720 B = 15.26 GiB** (与 index total_size 逐字节一致 ✅)
- 占比: MLP 66.4% | attention dense 18.4% | embedding+lm_head 15.2%

## 3. KV cache 容量

每 token = 36 × 8 × 128 × 2 (K+V) × 2 B = **147,456 B = 144 KiB**

| context | 总容量 |
|---------|--------|
| 4K | 576 MiB (0.5625 GiB) |
| 8K | 1152 MiB (1.125 GiB) |
| 每 (layer,kv_head) slab @8K | 8,192×128×2 = 2,097,152 B = 2 MiB 整 |

## 4. Attention GEMM 形状

QK-norm: per-head RMSNorm(head_dim=128) 作用于 q_proj/k_proj 输出, RoPE 之前。
attention 缩放 = 128^(-0.5)。GQA n_rep = 4。

| 场景 | 算子 | per-head GEMM | 说明 |
|------|------|---------------|------|
| prefill seq=128 | QK^T | 128×128×128 | 每层 32 个 (8 KV head 广播给 4 Q head) |
| prefill seq=128 | AV | 128×128×128 | 每层 32 个 |
| decode seq=1, cache=L | QK^T | 1×L×128 | L=4096 → 1×4096×128 |
| decode seq=1, cache=L | AV | 1×128×L | L=4096 → 1×128×4096 |

prefill 每层 attention MACs = 32 × 128³ × 2 = 134,217,728；
decode 每层 (L=4096) = 32 × 128 × 4096 × 2 = 33,554,432。

## 5. HBM 驻留张力 (重要约束)

HBM 16 GiB; BF16 权重 15.26 GiB + 8K KV 1.125 GiB > 16 GiB。

| 权重精度 | 权重字节 | +8K KV | 判定 |
|----------|----------|--------|------|
| BF16 | 15.26 GiB | 16.39 GiB | ❌ 溢出 0.39 GiB |
| INT8 | 7.63 GiB | 8.76 GiB | ✅ |
| INT4 | 3.81 GiB | 4.94 GiB | ✅ |

→ **8K context 必须 INT8/INT4 权重驻留**; BF16 模式限定 context ≤ 约 4K (0.56 GiB KV,
合计 15.82 GiB ✅, 余 0.18 GiB 给运行时)。此约束在 03-memory 中落地为布局规则。
