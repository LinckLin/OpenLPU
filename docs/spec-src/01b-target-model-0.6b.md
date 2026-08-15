# 01b — 验证模型卡 (Qwen3-0.6B dense)

> **本卡为验证模型（0.6B）。架构设计上限数值见 `01-target-model.md`（8B）。**
> 0.6B 是 8B 架构的子集（同 hidden/head_dim/vocab 口径，层数 28 vs 36、Q 头 16 vs 32）；
> 所有「设计上限 / 是否溢出」的判定以 8B 卡为准，本卡只提供 0.6B 的**权威数值事实**，
> 供 P1 roofline、P3 模拟器、P4 前端引用。

> 数据来源: Qwen/Qwen3-0.6B 官方 config.json (经 hf-mirror, sha `c1899de2...`) 与
> model.safetensors **单文件头**（无 index.json —— 该模型为单文件、未分片，故 404；
> 逐字节交叉验证改以 safetensors header 的 `total_size` 为准，与 HF API `safetensors.total` 一致）。
> 张量 311 个（28 层 × 11 + 3），全 BF16，无 `.bias`。

---

## 1. 关键结构参数 (逐项核验)

| 参数 | 值 | 出处 |
|------|-----|------|
| hidden_size | 1024 | config |
| num_hidden_layers | 28 | config |
| num_attention_heads | 16 | config |
| num_key_value_heads | 8 (GQA **2:1**) | config |
| head_dim | **128（显式，≠ hidden/heads = 64）** | config + 张量形状 |
| intermediate_size | 3072 | config |
| vocab_size | 151936 | config |
| rope_theta | 1000000.0 | config |
| rms_norm_eps | 1e-6 | config |
| QK-norm | 无条件启用, per-head (128), **RoPE 之前** | `q_norm`/`k_norm` 张量 `[128]` 存在 + modeling_qwen3.py（同 8B，非 config 字段） |
| 权重 dtype | bfloat16 | config (`torch_dtype`) + safetensors header（311 张量全 BF16） |
| max_position_embeddings | 40960 | config |
| tie_word_embeddings | **true**（但 checkpoint 物理存两份，见 §2 注） | config + safetensors header + 字节抽样比对 |
| attention_bias / MLP bias | 全无 | config + header（311 张量无 `.bias`，RMSNorm 仅 `.weight`） |
| sliding_window | null / 禁用, 28 层全 full attention | config（`use_sliding_window=false`） |
| rope_scaling | null | config |
| hidden_act | silu (SwiGLU) | config |
| max_window_layers | 28（`nextn` 稀疏预测窗口层数，`sliding_window=null` 时无实际作用） | config |

**关键结构差异（0.6B vs 8B，均非「预期值错误」而是 Qwen3 的 head_dim 显式化）：**

1. **`head_dim=128` 显式且与 `hidden/heads` 解耦**：Qwen3 的 `head_dim` 是 config 显式字段。
   0.6B 中 `16 头 × 128 = 2048 ≠ hidden(1024)`，故 **q_proj 输出 = 2048（2×hidden）**、**o_proj 输入 = 2048**，
   而非 hidden×hidden。8B 中 `32×128=4096=hidden` 恰好相等，掩盖了这一特性。
2. **GQA 比率 = 16/8 = 2**（8B 为 32/8 = 4）。KV.GATHER `broadcast=1` 硬件固定 ×4（02-isa §8.4），
   2:1 GQA 下仅 2 个 Q head 消费 → 冗余 2 副本、SRAM footprint 加倍（功能无碍，P6 优化项）。

---

## 2. 每层 GEMM 与参数量

| 算子 | 权重 [out×in] | GEMM M×N×K | 参数 | BF16 字节 |
|------|---------------|------------|------|-----------|
| q_proj | 2048×1024 | seq×2048×1024 | 2,097,152 | 4,194,304 |
| k_proj | 1024×1024 | seq×1024×1024 | 1,048,576 | 2,097,152 |
| v_proj | 1024×1024 | seq×1024×1024 | 1,048,576 | 2,097,152 |
| QKV 合并 | 4096×1024 | seq×4096×1024 | 4,194,304 | 8,388,608 |
| o_proj | 1024×2048 | seq×1024×2048 | 2,097,152 | 4,194,304 |
| gate_proj | 3072×1024 | seq×3072×1024 | 3,145,728 | 6,291,456 |
| up_proj | 3072×1024 | seq×3072×1024 | 3,145,728 | 6,291,456 |
| down_proj | 1024×3072 | seq×1024×3072 | 3,145,728 | 6,291,456 |
| MLP 合计 | — | — | 9,437,184 | 18,874,368 |
| attention dense (QKV+O) | — | — | 6,291,456 | 12,582,912 |
| 每层 dense 合计 | — | — | 15,728,640 | 31,457,280 |
| 每层 4×RMSNorm | in/post 1024 + q/k 128 | — | 2,304 | 4,608 |

- 28 层 dense = 440,401,920 参数；28 层 RMSNorm = 64,512；末层 norm = 1,024。
- embedding = 155,582,464；lm_head = 155,582,464（`[151936, 1024]`）。
- **物理总参数 = 751,632,384**（= dense 440,401,920 + RMSNorm 64,512 + embedding 155,582,464
  + lm_head 155,582,464 + 末层 norm 1,024）—— 与 safetensors header `total_size/2` **逐字节一致 ✅**，
  亦与 HF API `safetensors.total = 751632384` 一致 ✅。
- **BF16 物理总字节 = 1,503,264,768 B = 1.40 GiB**（header `total_size` 逐字节一致 ✅）。
- **tied 去重后有效参数 = 596,049,920 ≈ 0.596B**（"0.6B" 命名依据）。
- 物理占比: **embedding+lm_head 41.4%** | MLP 35.2% | attention dense 23.4%（RMSNorm 可忽略）。
  0.6B 是 **embedding 主导**（vocab 151,936 × hidden 1024 × 2 份 = 311M / 751M），与 8B 的 MLP 主导（66.4%）相反。

> **tie_word_embeddings 结论（明确）**：config 标 `tie_word_embeddings=true`，且
> `lm_head.weight` 与 `model.embed_tokens.weight` **逐字节一致**（对两个 `[151936,1024]` 张量抽样
> 头部 4 KiB、1/4 处 256 B、中部 64 B、尾部 4 KiB 四处全部相等），确证 tied。
> 但 checkpoint **物理存两份相同张量**（311 张量含独立 `lm_head.weight`），故磁盘/驻留口径取 751.6M；
> loader 可据此去重一份 155.58M（→ 596.05M 有效）。tie 只省 HBM 驻留，**不省 decode 每 token 的 logits 全读**（见 §6）。

---

## 3. KV cache 容量

每 token = 28 × 8 × 128 × 2 (K+V) × 2 B = **114,688 B = 112 KiB**

| context | 总容量 |
|---------|--------|
| 4K | 448 MiB (0.4375 GiB) |
| 8K | 896 MiB (0.875 GiB) |
| 每 (layer,kv_head) slab @8K | 8,192 × 128 × 2 B = 2,097,152 B = **2 MiB 整**（单 K 或 V 张量；`SLAB_SHIFT=21` ✅） |
| 全局 KV 区域 @8K | 28 × 8 × 2 × 2 MiB = **896 MiB = 0.875 GiB** |

> slab 口径与 8B 相同（`head_dim=128`、`kv_heads=8` 一致），故单 (layer,head,K或V) slab @8K 仍为 2 MiB，
> `SLAB_SHIFT=21` 无需因 0.6B 改动；仅层数 28（vs 36）使全局 KV 区域从 1152 MiB 缩为 896 MiB。

---

## 4. Attention GEMM 形状

QK-norm: per-head RMSNorm(head_dim=128) 作用于 q_proj/k_proj 输出, RoPE 之前。
attention 缩放 = 128^(-0.5)。GQA n_rep = **2**。

| 场景 | 算子 | per-head GEMM | 说明 |
|------|------|---------------|------|
| prefill seq=128 | QK^T | 128×128×128 | 每层 16 个 (8 KV head 广播给 2 Q head) |
| prefill seq=128 | AV | 128×128×128 | 每层 16 个 |
| decode seq=1, cache=L | QK^T | 1×L×128 | L=4096 → 1×4096×128 |
| decode seq=1, cache=L | AV | 1×128×L | L=4096 → 1×128×4096 |

prefill 每层 attention MACs = 16 × 128³ × 2 = 67,108,864；
decode 每层 (L=4096) = 16 × 128 × 4096 × 2 = 16,777,216。

---

## 5. HBM 驻留（0.6B 无溢出张力）

HBM 16 GiB；0.6B **全部精度模式（含 BF16 + 8K KV）均轻松容纳**，与 8B 的「BF16 溢出」约束相反。

| 权重精度 | 权重字节（物理 751.6M） | +8K KV | 合计 | 余量 | 判定 |
|----------|------------------------|--------|------|------|------|
| BF16 | 1.40 GiB | 0.875 GiB | 2.28 GiB | 13.72 GiB | ✅ |
| INT8 | 0.70 GiB | 0.875 GiB | 1.58 GiB | 14.42 GiB | ✅ |
| INT4 | 0.35 GiB | 0.875 GiB | 1.23 GiB | 14.77 GiB | ✅ |

> tied 去重（loader 共享 lm_head/embed_tokens 一份）后 BF16 降至 596.05M = **1.11 GiB**，余量更大。
> **结论：0.6B 验证模型不触发 8B 的 HBM 溢出张力，任何部署 dtype 均无需 context 压缩。**

---

## 6. decode 每 token 权重读

每 token 读（v0 decode 用 INT8 = 1 B/参数）：

| 部分 | 参数 | 说明 |
|------|------|------|
| 28 层 dense | 440,401,920 | 7 投影全读 |
| 28 层 RMSNorm | 64,512 | in/post + q/k |
| lm_head | 155,582,464 | logits 行全读 `151936×1024`（tied 仍需全读） |
| embedding | **0**（仅首 token 读单行 1024 元素 = 2 KiB） | 不重复读 |
| **合计** | **596,048,896** | — |

- **INT8 = 596,048,896 B = 0.596 GB**（= 0.555 GiB）。
- BF16 参考 = 1.19 GB（×2）。
- **相对 8B 设计上限（7,568,401,408 B = 7.57 GB）= 7.88%**。
  0.6B 的 decode 权重读约为 8B 的 **1/12.7**，是 P1 roofline 的下界参考点。

> tie 语义重申：`tie_word_embeddings=true` 时 lm_head 与 embedding **共享同一权重存储**，
> 但 logits 计算每 token 仍必须**全读** `151,936 × 1024` 权重行——tie 节省的是 HBM 驻留一份
> 155.58M（§2/§5），而非 decode 读流量。
