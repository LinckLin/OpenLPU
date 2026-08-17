# O1 StreamingLLM 窗口化 KV — 实现 + 质量门槛 + 性能验证

> 目标（roadmap §2 第一梯队，零硬件改动）：decode 时 KV 读取拆两段 —— attention **sinks** `[0, S)`（`S=4`）
> + **滚动窗口** `[pos-W+1, pos]`（`W ∈ {1024, 2048}`）；`KV.APPEND` 维持全量写入，只收缩**读窗口**。
> 质量门槛：golden 长上下文 PPL 基线（`quality-baseline`）上 windowed vs full 的 **ΔPPL ≤ 2%**（硬门槛），
> HellaSwag 作 sanity，20 token 交叉一致作次级观察。

## 0. 结论摘要

| 判定 | 结果 |
|---|---|
| **O1 硬门槛 ΔPPL ≤ 2%** | **W=2048 通过**（4K 0.673% / 8K 1.663%）；**W=1024 部分通过**（4K 1.717% 过，**8K 3.316% 不过**）|
| HellaSwag Δacc（sanity，非门槛） | 0（100 例 ctx+ending 最长 138 token ≪ W，窗口化 ≡ 全注意力）|
| 20 token 交叉一致 vs BF16 全注意力基线（次级） | W=1024 **12/20**；W=2048 **20/20** |
| 性能天花板（qsim 时序模型，对照路线图 866/1009） | **W=2048 → 866 tok/s（✓ 逐位一致）；W=1024 → 1008.4（≈1009 ✓）** |
| 实测墙钟（Python 执行器，@4K 单 token） | full 28.72 s → W=1024 8.51 s（**3.37×**）/ W=2048 11.11 s（**2.58×**）|

**门槛判定：O1 以 W=2048 为落地档（4K/8K 双过硬门槛、20/20 交叉一致、性能 866 tok/s 与路线图预期逐位一致）；
W=1024 在 8K 超阈（ΔPPL 3.316% > 2%），按门槛纪律否决其 8K 应用（4K 仍可用）。**

## 1. 环境（本次实测）

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-0.6B`（BF16，eager） |
| 推理框架 | transformers 5.5.3 · torch 2.11.0+cu130 · CUDA 13.0 · Python 3.10.12 |
| 设备 | RTX PRO 6000（96 GB） |
| 质量参照 | `docs/perf-research/quality-baseline/quality_baseline.json`（full-attention per-token NLL，PPL@4096=32.20、@8192=27.43）|
| 窗口参数 | S=4 sinks，W ∈ {1024, 2048}；eval 长度 4096/8192（均 > W，判别性成立）|
| 结果文件 | `/tmp/o1-streamingllm.json`（脚本 `qrun/windowed_kv.py` 生成，含逐段对比表）|

## 2. 实现（`qrun/windowed_kv.py`，纯软件，零硬件）

- **DC attention 重发**：`_emit_attention_dc_windowed` 把每 KV head 的 KV.LOAD 拆成两段 ——
  sinks `KV.LOAD pos_start=0, count=S`（1 个 subtile，mask `[S,128)=−inf`）+ 滚动窗
  `KV.LOAD pos_start=0, count=W`（`W/128` 个 subtile，全有效）。当前 token 天然落在滚动窗内
  （经 13-bit `pos_start` 寻址），**不再需要** full 版的 VMOV 当前 token subtile。
- **per-token patch**：`patch_windowed_dc` 把滚动窗 KV.LOAD 的 `pos_start` 置为 `pos−W+1`
  （字段位 68，13-bit；sinks 与 `KV.APPEND` 的 C_KV_POS 不变）；tail mask 为静态（sinks 段
  `[0,S)` + 满窗 subtile）。
- **复用现有引擎**：`WindowedEngine` 包装 `build_engine` 产出的 `RunEngine`，只换 DC program /
  patch / mask，embedding、KV bootstrap、QMetal、lm_head 读回全部复用（m4/m6 同一条驱动路径）。
- **KV.APPEND 全量写入不变**：cache 完整性不丢（HBM 里仍存全序列 K/V），只有读窗口收缩 ——
  与 StreamingLLM 的 "cache 完整、attention 窗口化" 语义一致。

数值校验（窗口化 qrun vs HF 窗口化注意力，4K 末位 logits；证据
`docs/perf-research/o1-streamingllm.json` → `logit_compare`）：argmax 一致（token 279，
HF/qrun 均为 279），max|Δlogit| = 1.47（W=2048，实测 1.4688）/ 2.41（W=1024，实测 2.4062）——
BF16 28 层逐 op 舍入量级，实现正确。

## 3. 质量门槛：ΔPPL（硬门槛 ≤ 2%，逐 token ΔNLL）

口径：`ΔPPL = exp(mean over all tokens of (NLL_windowed − NLL_full)) − 1`，
full 参照 = baseline JSON 的 `per_token_nll`（同 span 同 token 逐 token 对比）。

| W | 长度 | 段数 | **pooled ΔPPL** | 判定 | pooled PPL（full → windowed）|
|---|---|---|---|---|---|
| 1024 | 4096 | 10 | **1.717%** | ✅ 过 | 32.20 → 32.75 |
| 1024 | 8192 | 10 | **3.316%** | ❌ **不过** | 27.43 → 28.34 |
| 2048 | 4096 | 10 | **0.673%** | ✅ 过 | 32.20 → 32.41 |
| 2048 | 8192 | 10 | **1.663%** | ✅ 过 | 27.43 → 27.88 |

### 3.1 逐段对比表（per-sample ΔPPL）

W=1024：

| 段 | ΔPPL@4K | ΔPPL@8K | | 段 | ΔPPL@4K | ΔPPL@8K |
|---|---|---|---|---|---|---|
| s00 | −1.829% | +6.141% | | s05 | +1.877% | +2.182% |
| s01 | +0.612% | +1.276% | | s06 | +3.539% | +4.427% |
| s02 | +2.879% | +3.849% | | s07 | −0.596% | +1.022% |
| s03 | +3.900% | +4.765% | | s08 | +1.209% | +2.717% |
| s04 | +4.642% | +3.114% | | s09 | +1.118% | +3.785% |

W=2048：

| 段 | ΔPPL@4K | ΔPPL@8K | | 段 | ΔPPL@4K | ΔPPL@8K |
|---|---|---|---|---|---|---|
| s00 | −0.249% | +3.243% | | s05 | +0.527% | +1.948% |
| s01 | +0.984% | +0.870% | | s06 | +0.426% | +2.621% |
| s02 | +0.642% | +1.678% | | s07 | −0.050% | −0.236% |
| s03 | +2.704% | +2.381% | | s08 | +0.112% | +0.618% |
| s04 | +1.584% | +1.024% | | s09 | +0.087% | +2.532% |

观察：W=1024 在 8K 有 8/10 段 ΔPPL > 2%（长程依赖丢失显著）；W=2048 在 8K 仅 s00/s03/s06/s09
四个段略超 2%（段级），pooled 1.663% 仍达标。ΔPPL 随 W 下降、随长度上升，符合因果 LM 预期。

## 4. HellaSwag 零样本准确率（sanity，非门槛）

| 口径 | acc_norm |
|---|---|
| full（baseline JSON，batched 评分） | 0.5500（55/100）|
| windowed W=1024 / W=2048（unbatched 评分） | **0.5600（56/100）** |

- 100 例 `ctx+ending` 最长 **138 token** ≪ W=1024 ⇒ 窗口化注意力 ≡ 全注意力，**窗口化的真实 Δacc = 0**。
- 表中的 `+0.01`（56 vs 55）是**评分方式差异**（baseline 的 4-ending batched + padding vs 本脚本
  per-ending unbatched），落在 idx=220 一个近并列样本的 argmax 翻转上（label=0，batched 判 1、
  unbatched 判 0），**非窗口化效应**：同一 unbatched 评分下 full 与 windowed 均得 56/100。已如实记录。

## 5. 20 token 交叉一致 vs BF16 全注意力基线（次级观察）

prompt：4K（`build_seq` 重复文本，bootstrap 后 decode 20 token），比较 windowed 与 full 的生成 token。

| W | 一致率 | 分歧位置 |
|---|---|---|
| 1024 | **12/20** | [12,13,14,15,16,17,18,19] |
| 2048 | **20/20** | — |

- W=1024 在第 12 token 起漂移并级联（重复文本 prompt 下窗口化退化为重复局部上下文）；W=2048 与
  full 全程 token 一致。这与 ΔPPL 结论一致：W=2048 质量余量充足，W=1024 在长上下文已丢失判别性依赖。

## 6. 性能验证

### 6.1 qsim 时序模型天花板（对照路线图预期 866/1009）

用 `qsim/timing_p6` 冻结常数，以窗口化 KV 重读桶 `R = S + W` 重算 decode 天花板
（`schedule_overlap` = 路线图/flashattn 的 `28×max(21849 + 4096·R/720, 4096·R/256) + 216087`）：

| 配置 | R (tokens/层) | overlap tok/s | double-buffer tok/s（P6 调度口径）|
|---|---|---|---|
| full @4K | 4096 | 487.5 | 481.1 |
| full @8K | 8192 | 257.3 | 255.2 |
| **W=2048** | 2052 | **866.0** ✅ | 845.9 |
| **W=1024** | 1028 | **1008.4 ≈ 1009** ✅ | 984.0 |

提升倍率（对照路线图 `4K 481→866、8K 255→866/1009`）：

| W | 4K（481.1→）| 8K（255.2→）|
|---|---|---|
| 2048 | 866.0 = **1.80×** ✓ | 866.0 = **3.39×** ✓ |
| 1024 | 1008.4 = **2.10×** ✓ | 1008.4 = **3.95× ≈ 4.0×** ✓ |

**与路线图预期逐位一致**（866.0 / 1008.4≈1009；倍率 1.80/3.39/2.10/3.95 vs 预期 1.8/3.4/2.1/4.0）。
注：32K 档 866（W=2048）/1009（W=1024）需 slab 扩容/paged KV 前置（v0 slab 上限 8K，05 §1.2/§8），
为 gen-2 门槛，不在本期实测范围。

### 6.2 实测墙钟（qrun Python 执行器，单 token decode @ pos 4095，5 次取均值）

| 配置 | 单 token 墙钟 | 相对 full 加速 |
|---|---|---|
| full（全窗口 65 subtile） | 28.72 s | 1.00× |
| W=1024（9 subtile） | 8.51 s | **3.37×** |
| W=2048（17 subtile） | 11.11 s | **2.58×** |

- 加速来自 attention subtile 数从 65 → 9/17（W=1024/2048，砍掉 86%/74% 的 KV 重读 staging），与
  `R` 比例（9/65≈0.14、17/65≈0.26）方向一致。Python 执行器是功能仿真（numpy），**不建模 HBM 带宽**，
  故墙钟只反映 KV 读窗口收缩的结构收益；**硬件的 HBM-bound 收益以 §6.1 时序模型为准**（255→866/1009）。

## 7. 需评审项

1. **W 档取舍**：W=1024 在 8K 超阈（ΔPPL 3.316%），是否仅以 W=2048 作为 O1 落地档（推荐），
   还是接受 4K-only 的 W=1024 档。
2. **早期 decode 带**：窗口化 DC 程序假设 `pos ≥ W+S−1`（窗口与 sinks 不相交）；短 prompt
   （<W）无 bootstrap 的 PF 路径需回退 full attention（本期评估全部走 4K/8K bootstrap，不触发）。
3. **sink 段粒度**：sinks 只 4 token、以 `count=S` 单段 LOAD（partial subtile + per-token mask）；
   与 StreamingLLM 原论文 "attention sink 4 token" 一致，S 是否可调由评审定。
4. **8K 边界**：窗口化滚动窗经 `pos_start` 直接寻址 pos 8192（无需 VMOV），仍依赖 `slab_shift=22`
   （4 MiB slab，m4 已列评审项）；32K 需 slab 扩容/paged KV（gen-2）。
5. **HellaSwag Δacc 记录口径**：窗口化真实 Δacc=0（ctx ≪ W）；报告中的 +0.01 为 batched/unbatched
   评分差异，建议基线后续统一为 per-ending unbatched 评分以消除该假差异。

## 8. 复现

```bash
cd /home/lzl/project/newlpu
python3 qrun/windowed_kv.py \
    --baseline docs/perf-research/quality-baseline/quality_baseline.json \
    --qbin /tmp/qwen3-0.6b-bf16.qbin \
    --sinks 4 --windows 1024,2048 --seq-lens 4096,8192 \
    --out /tmp/o1-streamingllm.json
```

前置：HF 模型缓存（`Qwen/Qwen3-0.6B`）、`Rowan/hellaswag` 数据集、BF16 qbin（`qforge compile
--dtype bf16` 产出）；ΔPPL 无需 qbin（HF eager + 4D 掩码），runtime 段需 qbin + `Qwen3Ref`。
