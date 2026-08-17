# A-Gate 实测：0.6B 贪心 n-gram / Lookup 草案接受率 α 与链式验证门槛

> 2026-08-16 实测文档。任务单：DECISION v2 §3.1 立项前置门槛 —— 在 0.6B（BF16，
> transformers eager）上测 host 侧 n-gram/Lookup 草案的贪心接受率 α，按
> α≥0.4 立项 / α∈[0.25,0.4) 降 gen-2 / α<0.25 关闭 裁决。
> 数值参照同源：quality-baseline（PPL@4K 32.195 / @8K 27.427、HellaSwag acc_norm 55%，
> seed=0、PG19 validation interior-span 口径）。
> 完整原始数据：`docs/perf-research/decision/alpha-measure.json`；脚本：`qrun/spec_alpha.py`。

## 0. 结论摘要（先给答案）

| 草案来源 | coverage | α_cov（给定草案的接受率） | α_eff（有效接受率） | 链式验证模拟加速 | 裁决 |
|---|---|---|---|---|---|
| ngram（静态 4-gram 字典） | 0.168 | **0.093** | **0.016** | ~1.01× | **关闭** |
| pld（Prompt Lookup 动态字典） | 0.385 | **0.795** | **0.306** | ~1.35× | **降级 gen-2** |

**总裁决：A 方向降级 gen-2，不立项。** 理由：DECISION 的 α≥0.4 门槛与其 2.6–3.2× 增益
校准（speculative.md §3 的 `(1+αγ)` 公式）**以 100% 草案覆盖率为隐含前提**。实测：

1. **静态 4-gram 字典（任务单字面指定路径）直接关闭**：即便有覆盖，6 MB 通用文本字典
   对 0.6B 贪心续写的命中率仅 9.3%（HellaSwag 8.7% / PG19 10.3%），且覆盖率只有 16.8%，
   有效 α=1.6%，几乎无增益（模拟 1.01×）。
2. **PLD（Lookup 草案）表面 α 很高（0.795）但被覆盖率稀释**：同一 3-gram 在 prompt+已生成
   文本中重复出现时，模型 79.5% 会照抄上次的续写（"copy" 行为）；但 3-gram 在短 ctx 语料上
   只重复 38.5% 的次数 → 有效 α=0.306，落入 gen-2 区间。
3. **lookup 草案的链不延伸**：γ=4 链式验证对 PLD 几乎不乘收益（见 §3），实际模拟加速仅
   1.28–1.36×（≈γ=1 档），远低于 α≥0.4 校准的 2.6–3.2×。

## 1. 方法

### 1.1 草案来源（均为 host 侧零模型成本、贪心验证 lossless）

- **`ngram`**：静态 4-gram 字典（context=最近 3 token → 最频续写 token），训练语料 =
  PG19 **train** split（与所有评测语料 disjoint）约 6 MB（2 本书、2,665,825 token、
  1,913,268 个 4-gram、1,243,080 个唯一 context）。
- **`pld`**：Prompt Lookup Decoding 动态字典，reference = prompt + 已接受 token；
  当前 3-gram 在 reference 中**最近一次**出现处的下一 token 即草案。零训练、覆盖随生成
  动态建立。

### 1.2 评测语料（与 quality-baseline 同 seed 策略）

- **HellaSwag**：validation 子集 100 例，`seed=0`、`sorted(rng.choice(10042,100,replace=False))`
  （与 quality_baseline `evaluate_hellaswag` 完全一致的索引选择），ctx 作 prompt，贪心续写 32 token。
- **通用文本**：PG19 **validation** 前 6 本合格书的 interior span（`[256:512)` token，与
  `collect_spans` 同口径但截短）作 prompt，贪心续写 128 token。
- 合计 106 个 prompt、**3,968 个生成 token**（α 估计标准差 ≈±0.7%）。

### 1.3 测量协议（离线、精确）

贪心 token 只生成一次（HF `generate`, `do_sample=False` → argmax，BF16，eager 注意力），
然后草案在**完全相同的贪心续写**上逐 token 回放比对——贪心验证 lossless 意味着生成序列与
纯贪心解码逐字一致，唯一变量是草案沿途被接受的比例。三个量：

- `coverage` = 该位置能提出草案的比例；
- `α_cov` = 提出草案时命中贪心 token 的比例（= 字面"接受率 α"）；
- `α_eff` = `α_cov × coverage` = 每个解码 token 的接受概率（`(1+αγ)` 增益公式真正需要的量）。

链式验证（γ=4）另行回放：候选生成 → 逐 token 贪心接受至首个失配，用冻结 qsim 时序
（`T_step=1,154,087 cyc/token`、`T_verify(n)=1,154,087+(n−1)×4,748`，DECISION §3 /
speculative.md §3）估算模拟墙钟加速（仅报告，不作门槛）。

## 2. 结果

### 2.1 分语料 α 表（`alpha_per_corpus`）

| corpus | source | n_pos | proposed | coverage | α_cov | α_eff |
|---|---|---|---|---|---|---|
| general (PG19-val) | ngram | 768 | 252 | 0.328 | 0.103 | 0.034 |
| general (PG19-val) | pld | 768 | 265 | 0.345 | 0.728 | 0.251 |
| hellaswag | ngram | 3200 | 413 | 0.129 | 0.087 | 0.011 |
| hellaswag | pld | 3200 | 1262 | 0.394 | 0.809 | 0.319 |
| **OVERALL** | ngram | 3968 | 665 | **0.168** | **0.093** | **0.016** |
| **OVERALL** | pld | 3968 | 1527 | **0.385** | **0.795** | **0.306** |

### 2.2 链式验证模拟墙钟（`chain_verify_wallclock`，冻结 qsim，仅报告）

| corpus | source | produced | α_chain | speedup | tok/s |
|---|---|---|---|---|---|
| general | ngram | 768 | 0.029 | 1.03× | 892 |
| general | pld | 768 | 0.450 | 1.28× | 1106 |
| hellaswag | ngram | 3200 | 0.024 | 1.01× | 873 |
| hellaswag | pld | 3200 | 0.582 | 1.36× | 1180 |
| **合计** | ngram | 3968 | — | **1.01×** | ~877 |
| **合计** | pld | 3968 | — | **1.35×** | ~1169 |

> 对照：DECISION 线性口径（α_eff 代入 `(1+αγ)`、恒验 n=5）给出 pld 2.19× / ngram 1.05×；
> 实测链式模拟更低（1.35×/1.01×），差在 lookup 草案的链深（见 §3）。

## 3. 裁决与机理

### 3.1 判据判定（DECISION §3.1）

- **ngram**：α_cov=0.093 <0.25 且 α_eff=0.016 <0.25 → **关闭**。双重判据一致否决。
- **pld**：字面 α_cov=0.795 ≥0.4 → 表面"立项"；但 **α_eff=0.306 ∈[0.25,0.4) → 降级 gen-2**，
  且链式模拟加速仅 ~1.35×，远离门槛档 2.6–3.2×。按 DECISION v2 明示的
  "门槛与增益对齐"口径（α∈[0.4,0.55] 须 ≈2.6–3.2×），**以有效 α 为准，裁决降级 gen-2**。

### 3.2 关键机理（对 DECISION 增益模型的两处订正）

1. **覆盖率是缺失因子**：`(1+αγ)` 假设每步都能提出 γ 个草案（coverage=1）。lookup 草案
   在短 ctx 语料（prompt ≤256 token）覆盖率仅 ~38%，把 pld 的"每提案 79.5% 命中"稀释成
   "每 token 30.6% 命中"。α_eff 才代入增益公式，而非 α_cov。
2. **lookup 链不延伸**：链式草案的第 2..γ 个 token 用**试探性 token** 作 context，而 PLD 的
   reference 只含 prompt+已接受 token → 试探 context 不在 reference → 链在第 1 个草案处即断。
   实测 pld 每步 768/600≈1.28 token（general），≈γ=1 档；γ=4 的批验证乘不出收益。
   这正是 DECISION 线性口径（2.19×）与实测（1.35×）的差距来源。

### 3.3 未改变的事实

- 贪心验证 lossless，输出分布与纯贪心解码逐字一致，α 测量与"20/20 交叉一致"质量门槛同构
  （speculative.md §2）——门槛问题纯粹是收益不够，不是正确性。
- B1 权重墙（866→1208）仍只有本方向可免训练攻击；本实测否定的是**当前 lookup 草案的 α 量级**，
  不是链式验证架构本身。
- 静态字典路径（任务单字面"4-gram 字典"）的失败是**词表级预测失效**（0.6B 分布平坦 +
  6 MB 语料覆盖稀疏），与 PLD 的 copy 行为正交，两者不可互救。

## 4. 复现

```bash
python3 qrun/spec_alpha.py \
    --hellaswag-n 100 --general-spans 6 --train-mb 6 \
    --hellaswag-gen 32 --general-gen 128 --gamma 4 \
    --out docs/perf-research/decision/alpha-measure.json
```

确定性：贪心生成无采样（do_sample=False），HellaSwag 索引 seed=0，PG19 span 确定性
interior 选择，冻结 qsim 常数硬编码于脚本顶部。单卡 RTX PRO 6000 全程 ~168 s。
