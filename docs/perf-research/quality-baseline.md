# 质量基线 — Qwen3-0.6B golden 长上下文 PPL + HellaSwag（perf-research §4 第 1 步）

> 用途：为 windowed-KV 优化项（**O1 StreamingLLM / O2 INT8 KV / O3 H2O**）建立 **golden 参照**。
> 口径：HF 原生 full-attention（`eager`，无窗口、无 stride），BF16 权重 + fp32 log-softmax；
> 每个被预测 token 都 attend 到完整 4K/8K 上下文 → 判别性成立（eval 长度 4096/8192 均 **>** 窗口候选 W=1024/2048）。

## 0. 环境（本次实测）

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-0.6B`（config 声明 `transformers_version: 4.51.0`，发布口径） |
| 推理框架 | **transformers 5.5.3**（现装版；同权重同架构，eager 因果注意力数学一致，数字对 minor 版本不敏感） |
| 权重精度 | BF16；logits→fp32 后 `log_softmax`（golden 口径，避免 BF16 softmax 精度损失） |
| attention | `attn_implementation="eager"`（HF 原生 full attention，非 sdpa/flash） |
| 设备 | CUDA 13.0 · torch 2.11.0+cu130 · RTX PRO 6000 |
| 数据集 | PG19 validation（`emozilla/pg19`，50 本）；HellaSwag validation（`Rowan/hellaswag`） |
| 随机种子 | 0（仅用于 HellaSwag 子集选择；PPL span 选择无随机，纯确定性） |

## 1. 长上下文 PPL 基线（golden full-attention）

**方法**：每本书取内部连续 span `tokens[L : 2L]`（跳过卷首标题/目录，span 完整落在单本书内，保证真实长程依赖）；
每个 span 用 full causal attention 一次前向，逐 token 计算 $\mathrm{NLL}_i = -\log P(x_{i+1}\mid x_{1..i})$，
$i$ 覆盖 span 内全部前文。PPL = $\exp(\text{pooled mean NLL})$。每长度 10 段（满足 ≥10 契约）。

| 长度 L | 段数 | 目标 token 数 | **PPL@L**（pooled） | per-sample mean-NLL 均值 ± 标准差 | 中位数 | 范围 [min, max] |
|---|---|---|---|---|---|---|
| 4096 | 10 | 40,950（4095×10） | **32.20** | 3.4718 ± 0.2944 | 3.4546 | [3.0286, 3.9150] |
| 8192 | 10 | 81,910（8191×10） | **27.43** | 3.3115 ± 0.3237 | 3.3686 | [2.5440, 3.7386] |

补充口径（per-sample PPL = exp(per-sample mean NLL)，逐段再取统计）：

| 长度 L | per-sample PPL 均值 ± 标准差 | 中位数 | 范围 [min, max] |
|---|---|---|---|
| 4096 | 33.60 ± 9.67 | 31.65 | [20.67, 50.15] |
| 8192 | 28.73 ± 7.94 | 29.09 | [12.73, 42.04] |

- **判别性成立**：eval 长度 4096/8192 > 窗口候选 W∈{1024, 2048}，windowed-KV 必丢 2048 之外的上文，ΔPPL 可判别。
- 长上下文一致性 sanity：L 从 4096→8192，PPL 32.20→27.43（更长的上下文降低了困惑度，符合因果 LM 预期）。

## 2. HellaSwag 零样本准确率（sanity）

| 指标 | 值 |
|---|---|
| **acc_norm**（长度归一化，主口径） | **0.5500**（55/100） |
| acc_raw（原始 log-likelihood 和，次口径） | 0.3900（39/100） |
| 方法 | 零样本多选：对 4 个 completion 各算给定 ctx 的条件 log-likelihood，取 argmax（greedy 选择）；不用 chat template |
| 子集 | validation 中 seed=0 确定性抽 100 例（indices 已存 JSON，可精确复现） |

> acc_norm 为 HellaSwag 标准口径（消除 raw sum 对短 completion 的偏好，raw 39%→norm 55% 即为此效应）。
> 0.55 与 Qwen3-0.6B 公开零样本水平一致，证明 scoring 实现正确。

## 3. 门槛预定义（O1 硬门槛 + sanity）

| 门槛 | 定义 | 判定 |
|---|---|---|
| **O1 硬门槛** | ΔPPL = (PPL_windowed − PPL_full) / PPL_full ≤ **2%**（相对，windowed vs full attention，同 span 同 token 逐 token 对比） | 不达即否决 O1 |
| **O2/O3 PPL 门** | 同上 ΔPPL ≤ 2% | 同 O1 |
| HellaSwag Δacc | Δacc = acc_windowed − acc_full，**仅记录不设达标线**（零样本 sanity，非 PPL 门槛） | 记录 |

- ΔPPL 计算以本基线 JSON 的 `per_token_nll` 为 **full 参照**；windowed 侧加载同 span 的 `token_ids` 重跑后逐 token 对比，
  得到 per-token ΔNLL → exp(mean ΔNLL)−1。
- 适用项：O1 StreamingLLM、O2 INT8 KV、O3 H2O（roadmap §4 第 1 步共同前置，§3 方法学第 2 条「质量门槛先行」）。

## 4. 数据文件（供 ΔPPL 比较）

| 文件 | 内容 |
|---|---|
| `docs/perf-research/quality-baseline/quality_baseline.json` | meta + PPL 每 span 的 `token_ids`（精确输入，免重 tokenize）与 `per_token_nll`（full-attention 逐 token NLL）+ HellaSwag 每例 scores/pred/label |

JSON 结构：

- `ppl["4096"|"8192"][i]` → `{sample_id, source_id(book index), offset_tokens, seq_len, token_ids[], per_token_nll[], mean_nll, ppl}`
  - `token_ids` 长度 = L；`per_token_nll` 长度 = L−1（`per_token_nll[i]` = 预测 `token_ids[i+1]` 的 NLL）。
- `hellaswag` → `{accuracy_norm, accuracy_raw, indices[], per_example[]}`（per_example 含 `label/pred/pred_raw/correct/scores_norm/scores_raw`）。
- `meta` → 完整环境（model/dtype/attention/device/torch/transformers/cuda/seed/ppl_dataset/o1_delta_ppl_gate）。

## 5. 复现命令

```bash
cd /home/lzl/project/newlpu

# 前置（自动缓存到 ~/.cache/huggingface）：
#   模型  Qwen/Qwen3-0.6B   —— 或 MODEL_DIR 指向本地
#   数据  emozilla/pg19、Rowan/hellaswag —— datasets 自动下载

python3 qrun/quality_baseline.py \
    --dataset pg19 --seq-lens 4096,8192 --n-samples 10 --hellaswag-n 100 --seed 0
```

输出：`docs/perf-research/quality-baseline/quality_baseline.json`（本报告数字即由该命令生成）。

本次实测环境：Python 3.10.12 · torch 2.11.0+cu130 · transformers 5.5.3 · datasets 5.0.1 · CUDA 13.0。
