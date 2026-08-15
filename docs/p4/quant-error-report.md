# P4/M3–M4 逐层量化误差报告（6 类线性投影 × PF/DC：INT8 + INT4）

> 判据（沿用 M2a，见 `plans/p3-p4-plan.md` §4 任务 4b）：
> **INT32 逐位 bit-exact**（executor per-128-K-group 部分和 vs 独立 einsum 参考）
> 且 **dequant fp32 < 1e-6**（绝对值，或相对值——见文末注）。
> 输入取 golden 真实激活（`prefill_seq128` PF / `decode_seq1_cache1024` DC），权重取
> `model.safetensors` 真实权重（qkv 为 q/k/v 三张量融合 4096×1024）。量化：对称
> per-128-K-group（权重侧）+ 激活 scale 折叠进 CD scale。

## 结果总览：12/12 PASS

| 投影 | 形状 [N,K] | 模式 | M | INT32 bit-exact | dequant abs | dequant rel | 量化误差 abs | 量化误差 rel | PASS |
|------|-----------|------|---|-----------------|-------------|-------------|-------------|-------------|------|
| qkv  | 4096×1024 | PF | 128 | ✅ | 7.15e-07 | 2.29e-07 | 8.08e-02 | 1.41% | ✅ |
| qkv  | 4096×1024 | DC | 1   | ✅ | 2.38e-07 | 1.82e-07 | 3.83e-02 | 0.94% | ✅ |
| o    | 1024×2048 | PF | 128 | ✅ | 2.38e-07 | 9.94e-07 | 3.64e-02 | 1.17% | ✅ |
| o    | 1024×2048 | DC | 1   | ✅ | 9.69e-08 | 8.78e-07 | 7.68e-03 | 0.48% | ✅ |
| gate | 3072×1024 | PF | 128 | ✅ | 0.00e+00 | 0.00e+00 | 1.86e-01 | 2.71% | ✅ |
| gate | 3072×1024 | DC | 1   | ✅ | 1.19e-07 | 3.94e-08 | 1.14e-01 | 2.14% | ✅ |
| up   | 3072×1024 | PF | 128 | ✅ | 1.19e-07 | 4.31e-08 | 1.12e-01 | 3.20% | ✅ |
| up   | 3072×1024 | DC | 1   | ✅ | 1.19e-07 | 4.84e-08 | 4.87e-02 | 1.97% | ✅ |
| down | 1024×3072 | PF | 128 | ✅ | 1.19e-07 | 5.62e-08 | 9.82e-02 | 2.55% | ✅ |
| down | 1024×3072 | DC | 1   | ✅ | 4.47e-08 | 1.40e-07 | 8.52e-03 | 1.07% | ✅ |
| lm_head | 151936×1024 | PF | 128 | ✅ | 1.91e-06 | 1.11e-07 | 1.04e+00 | 3.40% | ✅ |
| lm_head | 151936×1024 | DC | 1   | ✅ | 4.77e-07 | 9.46e-08 | 3.06e-01 | 1.20% | ✅ |

## 判据说明

- **INT32 逐位 bit-exact**：executor 的 `int8_group_partials` 对每个 128-K-group 的
  INT32 部分和，与独立 `np.einsum`（int32）逐位相等 → 量化打包/重排/tiling 无位级错误。
- **dequant**：executor fp32 dequant 输出 vs 以**同一 CD scale**（BF16）独立 fp64 dequant
  参考的差异。lm_head PF 的 1.9e-6 绝对值来自 executor 的 fp32 跨组累加舍入（04 §1.5：
  「fp32 scale → fp32 跨组累加」），logits |y| 峰值 ≈ 30 时相对误差 ≈ 1.1e-7（≈ fp32 eps）。
  故判据保留双口径：**abs < 1e-6 或 rel < 1e-6**；所有例 rel ≤ 1e-6。

## 量化误差（vs fp32 golden，信息性）

- 6 类相对量化误差 **0.48%–3.40%**，均在 W8A8 对称 per-128-group 预期区间（远低于
  计划风险表「超预算回退 per-64-group」的触发线）。
- 最大出现在 **lm_head PF（3.40%）** 与 **up PF（3.20%）**——二者激活/权重动态范围较大，
  与对称 per-128-group 的保守选择一致；无需回退 per-64-group。

## 复现

```bash
python3 qforge/verify_m3.py            # 全量：构建 qbin + (a) round-trip/执行 + (b) 12 例
cat docs/p4/m3-results.json            # 机器可读结果
```

---

## INT4 (W4A16) 结果（plans/int4-plan.md §3 Q4a）

> 判据（M4 INT4 双轨）：**(a) 实现正确性** = 同 scale fp64 参考 dequant < 1e-6
> （dequant rel 按输出**峰值**幅度计，稳健于近零元素；见下方注）；**(b) 量化误差**
> vs fp32 golden **如实报告**（预期 8–15% rel 量级，以报告为准，不设 ≤1 ULP）。
> 另设**打包→executor 解包往返锁定**（位序权威 = `qsim/executor.py` 偶低/奇高）。

权重 INT4 对称 per-128-K-group（`sw = max|w_group|/7`，clip `[-7,7]`，round-to-nearest），
2b 打包（偶列低半字节/奇列高半字节）；激活保持 BF16（W4A16，无运行时 QUANT）；acc=FP32，
dequant=1（fp32 组内累加）。qbin tensors 表 `dtype=INT4`。

## 结果总览：12/12 PASS（锁定 + dequant 双口径）

| 投影 | 形状 [N,K] | 模式 | 打包往返 | dequant abs | dequant rel(峰值) | 量化误差 abs | 量化误差 rel | INT8 rel | PASS |
|------|-----------|------|---------|-------------|-------------------|-------------|-------------|---------|------|
| qkv  | 4096×1024 | PF | ✅ | 1.00e-06 | 1.74e-07 | 3.42e-01 | 5.97% | 1.59% | ✅ |
| qkv  | 4096×1024 | DC | ✅ | 4.17e-07 | 1.04e-07 | 1.36e-01 | 3.35% | 0.92% | ✅ |
| o    | 1024×2048 | PF | ✅ | 3.12e-07 | 1.00e-07 | 1.23e-01 | 3.95% | 1.30% | ✅ |
| o    | 1024×2048 | DC | ✅ | 7.36e-08 | 4.57e-08 | 8.11e-02 | 5.08% | 0.50% | ✅ |
| gate | 3072×1024 | PF | ✅ | 8.99e-07 | 1.31e-07 | 5.50e-01 | 8.03% | 2.67% | ✅ |
| gate | 3072×1024 | DC | ✅ | 4.56e-07 | 8.49e-08 | 3.51e-01 | 6.59% | 2.10% | ✅ |
| up   | 3072×1024 | PF | ✅ | 4.89e-07 | 1.42e-07 | 2.34e-01 | 6.72% | 3.20% | ✅ |
| up   | 3072×1024 | DC | ✅ | 1.56e-07 | 6.35e-08 | 1.63e-01 | 6.60% | 2.01% | ✅ |
| down | 1024×3072 | PF | ✅ | 4.08e-07 | 1.03e-07 | 1.19e-01 | 3.09% | 2.55% | ✅ |
| down | 1024×3072 | DC | ✅ | 1.04e-07 | 1.34e-07 | 6.96e-02 | 8.76% | 1.07% | ✅ |
| lm_head | 151936×1024 | PF | ✅ | 4.05e-06 | 1.32e-07 | 2.48e+00 | 8.10% | 3.39% | ✅ |
| lm_head | 151936×1024 | DC | ✅ | 1.67e-06 | 6.54e-08 | 1.74e+00 | 6.80% | 1.19% | ✅ |

## 判据说明

- **打包往返锁定**：qforge `pack_int4` 打包字节写入 executor HBM，经
  `_read_vector(DT_INT4)` 解包 == 原 INT4 矩阵，12/12 逐位一致 → 位序契约锁定
  （偶元素低半字节/奇元素高半字节，与 executor `_read_vector/_write_vector` 一致）。
- **dequant**：W4A16 实现（fp32）vs 以**同一 BF16 CD scale** 独立 fp64 参考的差异。
  相对误差按输出**峰值**幅度计（`max|Δ| / max|y|`），全部落在 4.6e-8–1.7e-7
  （≈ fp32 eps 1.2e-7），**均 < 1e-6**。lm_head 的 4.05e-6 绝对值来自 fp32 组内累加
  舍入（W4A16 在 fp32 域做组内累加，非 W8A8 的 INT32 精确部分和）；logits |y| 峰值
  ≈ 30.7 时相对误差 ≈ 1.3e-7，与 fp32 eps 同量级。判据保留双口径：abs < 1e-6 或
  rel < 1e-6。
- **量化误差（信息性，如实报告）**：INT4 相对误差 **3.09%–8.76%**，集中在 6–9%
  区间（gate/up/down/lm_head 高动态范围类；qkv/o 略低）。低于计划风险表「8–15%
  rel」的中位预期，但**以报告为准**——对称 per-128-group 的保守选择使小投影
  （qkv/o）误差低于 8%，无需回退 AWQ 式 per-group 搜索。最大出现在 down DC
  （8.76%）与 lm_head PF（8.10%）。
- **INT8 同口径对比**：INT4 误差为 INT8 的 **1.2×–10.2×**（down PF 最低 1.21×，其余 11 例 2.1×–10.2×；INT8 0.50%–3.39%），
  与 4-bit（15 个非零层级）对 8-bit（255 个非零层级）的粒度差一致；绝对量级仍
  远小于 token 级一致率风险线（Q4b 以 20 token ≥8/10 硬门槛另测）。

## 全模型 INT4 qbin（可装载）

- 产物：`/tmp/qwen3-0.6b-m4-int4.qbin`（flags=7 = INT4(3) | dual-mode(4)）。
- tensors 141 项全部 `dtype=INT4`；权重字节 297,992,192 B（= INT8 595,984,384 的
  ½，与 2b 打包一致）；scale 字节 9,312,256 B（与 INT8 相同，N×G×2 与权重位宽无关）。
- PF/DC 程序：W4A16 GEMM/GEMV 各 3,875 条（28 层 × 96 tile + lm_head 1,187 tile），
  QUANT 操作 0 条（激活保持 BF16）；指令全解码通过；权重/scale 经 executor HBM
  装载→读回逐位一致（round_trip_ok）。

## 复现

```bash
python3 qforge/verify_m4_int4.py                       # 全量：构建 INT4 qbin + 双轨 12 例
cat docs/p4/m4-int4-results.json                       # 机器可读结果
```

## AWQ 回退（plans/int4-plan.md §5 风险触发）

> Q4b 全模型 INT4 decode 首测 token 交叉一致率 3/20（< 8/10 硬门槛），触发 §5
> 风险表的回退方案：**AWQ 式 per-group 搜索（Q4a 增补）**。已落地于
> `qforge/quant.py:quantize_weight_int4_awq`（对称 `quantize_weight_int4` 的
> 同签名增补，仅 scale 取值不同；打包/lowering/qbin 契约不变）。

搜索法：对每个 `(输出行 n, K-group g)`，在 `s = β·max|w_group|/7`（β∈[0.4,1.2]
33 档）上最小化激活加权重建误差 `Σ_k ((w[n,k]−dequant(wq))·a[k])²`，其中
`a[k] = RMS_t(x[t,k])`（校准激活的逐列幅度，AWQ 式保护显著列）。此处校准为
**oracle**（用各自模式 golden 激活，逐投影验证）；全模型运行时校准（`qrun/weights.py
calibrate_act_inputs`，从 P1_PROMPT 捕获真实逐投影激活）已由 Q4b 落地并用于 §2.5 的
全模型 decode。

| 投影 | 模式 | logits RMSE 对称→AWQ | 比值 | max-rel 对称→AWQ |
|------|------|---------------------|------|------------------|
| qkv  | PF | 0.0250→0.0209 | 0.837 | 5.97%→6.41% |
| qkv  | DC | 0.0233→0.0189 | 0.812 | 3.35%→4.26% |
| o    | PF | 0.0180→0.0158 | 0.881 | 3.95%→3.42% |
| o    | DC | 0.0185→0.0155 | 0.840 | 5.08%→3.92% |
| gate | PF | 0.0644→0.0500 | 0.777 | 8.03%→5.27% |
| gate | DC | 0.0654→0.0493 | 0.754 | 6.59%→4.21% |
| up   | PF | 0.0449→0.0377 | 0.841 | 6.72%→7.44% |
| up   | DC | 0.0464→0.0372 | 0.801 | 6.60%→5.30% |
| down | PF | 0.0146→0.0125 | 0.854 | 3.09%→3.81% |
| down | DC | 0.0133→0.0103 | 0.775 | 8.76%→5.21% |
| lm_head | PF | 0.4022→0.3226 | 0.802 | 8.10%→6.81% |
| lm_head | DC | 0.3119→0.2461 | 0.789 | 6.80%→5.68% |

**如实结论**：AWQ 在 logits RMSE（token 一致率相关目标，AWQ 直接最小化的量）上
一致收窄 **12%–25%**（比值 0.75–0.88，lm_head 约 0.79–0.80）；但 max-abs
（最坏点）口径有升有降（gate/down DC/lm_head 明显改善，qkv/up PF 略升）——AWQ
优化的是均方误差而非最坏点。**Q4b 全模型 decode 实测（`qrun/m6_int4.py --awq`，
oracle-calibrated on P1_PROMPT，→ `docs/p5/int4-awq-results.json`）：交叉一致率
仍 3/20（< 8/10 硬门槛）**——AWQ 使前 3 token 全对（plain 第 1 位即分歧）、分歧
位置 logits rel 误差普遍降 20–30%（pos3 0.450→0.191、pos14 1.165→0.834），但
rel 误差 ~0.2–0.96 仍超出该 prompt 解码路径的 argmax margin。**结论：0.6B 4-bit
权重（即便激活感知 per-(n,g) 搜索）不足以在 28 层累积后维持 ≥8/10 贪心一致，
如实记录不虚报。**

**下一步选项**（供评审，均超出「CD 结构不变」的本期增量）：
1. **per-64-group**：K-group 128→64，scale 张量 [N,G] 翻倍（9.3 MB→18.6 MB），
   权重重构误差随组缩小进一步逼近逐行精度；代价是 CD 描述符与 RTL dequant
   通路需按 per-64 适配。
2. **混合精度 attention BF16**：attention 四投影（qkv/o）保持 BF16（权重字节
   增量小、decode 仍带宽 bound），仅 MLP（gate/up/down）+ lm_head 走 INT4——
   attention 是量化更敏感的通路，保持其精度可压低残差流累积误差。
```bash
python3 qforge/verify_m4_int4.py --awq        # 附 AWQ 对比（oracle 校准，约 2.5 分钟）
python3 qrun/m6_int4.py --awq                 # 全模型 AWQ decode（20 token 交叉一致率）
cat docs/p5/int4-awq-results.json             # 全模型一致率（3/20，见 §2.5）
```
