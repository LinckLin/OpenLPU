# B-Fold：QK-norm 折叠 + INT4 KV 数值验证（带符号 scale）—— 否决 INT4 K，折叠机制本身成立

> 目标（DECISION §3.2 / kv-outlier.md §3.3）：在 HF/ref（torch）参照上验证「INT4 KV + QK-norm 折叠」，
> 量化对象 = 单位范数 pre-weight 向量 `k_unit = rmsnorm(k_raw)`，`k_norm` 折入 dequant scale
> （`scale_c = k_norm[c] × s_q`，**带符号** BF16），RoPE 改为 on-read（pre-RoPE 存储）；V 保持
> per-token per-head INT4。判据：ΔPPL ≤2%（相对，vs golden full attention）**且** 20 token
> 交叉一致 ≥8/10（隔离口径：量化 vs windowed-BF16）。本轮只做数值验证，**不改 RTL**。

## 0. 结论摘要

| 判定 | 结果 |
|---|---|
| **ΔPPL 门槛（≤2%）** | **INT4 K 折叠否决**：4K **7.470%** / 8K **6.901%** ≫ 2% ❌ |
| **20 token 交叉一致（隔离：量化 vs windowed-BF16）** | INT4 K 折叠 **12/20** ❌（< 16/20） |
| 折叠机制正确性 | **INT8 K 折叠 + INT4 V = 1.266% / 1.995%，交叉一致 20/20 ✅** —— 折叠路径本身成立，问题只在 4-bit 精度 |
| 带符号 scale 验证 | k_norm **3.404% 负通道（122/3584），min −3.859** —— 与 DECISION「3.4% 负通道 min −3.86」逐位一致 ✓ |
| 性能天花板 | INT4 KV **1096.3（1.266×）/ 1097.9（1.268×）** 因质量门**不可达**；**可达修正档 = INT8 K（折叠）+ INT4 V = 1049.3（1.212×）** |
| RoPE 条件 | 1.27× 与一切 pre-RoPE 存储数字**以专用 on-read 旋转器为条件**；无旋转器口径 ≈4× 旋转开销（decode 变旋转绑定，见 §6） |

**门槛判定：B 轨按 DECISION 字面方案（INT4 K 折叠 + INT4 V）如实否决——INT4 K 的 ΔPPL 7.5%/6.9%
远超 2%、交叉一致 12/20 低于 8/10，双判据均不过。但否决的是 4-bit 精度本身，不是折叠机制：
INT8 K 折叠（同路径）1.27%/2.00% 过门槛、交叉一致 20/20，证明「把 k_norm 折出后 per-head 量化
恢复 per-channel 质量」的核心断言成立。可达修正档 = INT8 K（折叠）+ INT4 V（1049 tok/s，1.212×），
或保守 INT8 K+V（折叠，1006 tok/s，1.162×）。**

## 1. 方案与口径

- **QK-norm 折叠（K）**：`k_unit = rmsnorm(k_raw)`（单位 RMS，无通道离群，|k_unit| 有界 ≤√128≈11.3）；
  量化 `q = round(k_unit / s_q)`，`s_q = max|k_unit|/(2^bits−1)`（per-head-per-token，正）；
  dequant `k_hat = q × scale_c`，`scale_c = k_norm[c] × s_q`（**带符号** BF16，k_norm 可为负）。
  所有 scale 元数据 BF16 舍入（硬件忠实）。`k_hat` 为 **pre-RoPE**，RoPE 在 attention 时 on-read 施加。
- **V**：per-token per-head 对称 INT4（`s = max|v|/7`，`clip[-7,7]`；与 `qforge.quant` W4A16 同一对称规则）。
- **bits 参数化**：`--k-bits 4`（primary INT4）/ `--k-bits 8`（INT8 折叠锚，验证折叠机制正确性）。
- **参照**：golden full-attention PPL（`quality_baseline.json` `per_token_nll`，逐 token）+ torch ref
  windowed W=2048 forward（`qrun/fold_verify.py`）。
- **交叉一致隔离口径**（O2 §9.1 修正）：量化 vs **windowed-BF16**（隔离窗口化误差）；另并列 vs full 供审计。

## 2. 带符号 scale 验证（DECISION 锚点逐位复现）

| 量 | 数值 | 判定 |
|---|---|---|
| k_norm 通道数 | 28 层 × 128 = 3584 | — |
| **k_norm 负通道** | **122 / 3584 = 3.404%** | ✅ = DECISION「3.4% 负通道」 |
| **k_norm min / max** | **−3.8594 / 96.5** | ✅ = DECISION「min −3.86」/ O2「max 96.5」 |
| 折叠 scale_c = k_norm×s_q（INT4，s_q∈[0.273,1.586]） | 负通道 **3.404%**，min **−3.59** | s_q>0 保号，负通道比例与 k_norm 一致 |
| 折叠 scale_c（INT8，s_q∈[0.015,0.087]） | 负通道 3.404%，min −0.199 | 同上（s_q 更小，量级缩小） |

- **「min −3.86」指 k_norm（模型权重）本身**；`scale_c = k_norm[c] × s_q` 的 min 随 token 的 s_q 缩放
  （INT4 样本 −3.59）。dequant 必须按**带符号** scale 乘，否则 3.4% 负通道反号，与 DECISION 修订一致。

## 3. ΔPPL 门槛（windowed W=2048 + 量化 vs golden full，pooled，10 段）

| 配置 | 4K ΔPPL | 8K ΔPPL | 判定（≤2%） |
|---|---|---|---|
| windowed BF16（窗口化 only，O1/O2 参照） | 0.673% | 1.663% | ✅（逐位复现 O1/O2） |
| **INT4 K 折叠 + INT4 V（primary）** | **7.470%** | **6.901%** | ❌ **否决** |
| **INT8 K 折叠 + INT4 V（折叠机制锚）** | **1.266%** | **1.995%** | ✅（8K 贴线 1.995%，仍过） |

- **windowed BF16 0.673%/1.663% 与 O1/O2 逐位一致**，锁定 my forward（on-read RoPE 重构后）忠实性。
- INT8 K 折叠 1.266%/1.995% 与 O2 的 per-channel INT8 K + per-head V（0.701%/1.776%）同量级、同过门槛
  ——**证明「把 k_norm 折出后 per-head 量化即恢复 per-channel 质量」的核心断言成立**。残差（上界
  ≈0.6%/0.2%）同时含两个因子：① K per-head（折叠）比 per-channel 略粗；② 本配置 V 为 INT4（O2 对照为
  V INT8）略粗——故该残差是上界，per-head 折叠单因子误差更小。

## 4. 20 token 交叉一致（4K 重复文本 prompt）

| 对比 | 一致率 | 分歧位置 |
|---|---|---|
| windowed BF16 vs full | 12/20 | [12..19]（窗口化固有，复现 O2） |
| **INT4 K 折叠 vs windowed-BF16（隔离门槛）** | **12/20** ❌ | [12..19]（量化可见，翻转 8 处 near-tie） |
| INT4 K 折叠 vs full | 20/20 | []（巧合：量化扰动把 near-tie 翻回 full 方向） |
| **INT8 K 折叠 vs windowed-BF16（隔离门槛）** | **20/20** ✅ | []（量化不可见） |

- INT4 折叠「vs full = 20/20」是重复文本 near-tie 的混沌巧合（pos12 起为 near-tie，量化扰动方向恰与
  去窗口化一致），**不可作为质量证据**；隔离口径 12/20 正确反映「INT4 量化是大幅扰动」。
- 该 prompt 的 near-tie 主导使其判别力弱于 ΔPPL；两判据独立地都指向「INT4 K 不过」，结论稳健。

## 5. 根因：INT4 K 的重建误差（layer-0 前 256 token）

| 量化 | K 重建 mean\|err\| | mean rel err | 对 attention 的后果 |
|---|---|---|---|
| INT8 折叠 | 0.0346 | 8.6% | 可接受（ΔPPL 1.27%） |
| **INT4 折叠** | **0.5862** | **58.6%** | 灾难（ΔPPL 7.5%） |

- `k_unit` 元素典型幅值 ≈0.63（mean abs），INT4 步长 `s_q = max|k_unit|/7 ≈ 0.2–1.6` → 单元素
  误差 ≈0.3–0.6，相对 ≈50%。k_norm 折叠只解决**通道离群**（per-head 可用），不解决 **4-bit 步长
  本身**。这与 kv-outlier 自注「INT4 质量尚未重验 / 文献 4-bit 证据全在 7B+」一致——0.6B 无先例，
  实测即失败。

## 6. 性能天花板（qsim 时序模型，W=2048，R=S+W=2052；**on-read RoPE 条件**）

| 配置 | KV B/token/layer | overlap tok/s | 相对 866 |
|---|---|---|---|
| windowed BF16（O1 基线） | 4096 | 866.0 | 1.000× |
| INT8 K+V（折叠，data 2048+scale 32） | 2080 | 1006.2 | 1.162× |
| **INT8 K（折叠）+ INT4 V（可达修正档）** | 1536+32=1568 | **1049.3** | **1.212×** |
| INT4 K+V（折叠，data 1024+scale 32） | 1056 | 1096.3 | 1.266× ❌质量否决 |
| INT4 K+V（K scale 折入 k_norm，1024+16） | 1040 | 1097.9 | 1.268× ❌质量否决 |

- INT4 的 1096.3/1097.9 逐位复现 kv-outlier §4 表；但 **质量门否决后不可达**。
- 可达档落在 **INT8 K（折叠）+ INT4 V = 1049.3（1.212×）**（K 走折叠后 INT8、V 走 INT4，数据 1536 B/token）。
  该档 ΔPPL 1.266%/1.995% 过门槛、交叉一致 20/20（§3/§4 的 `--k-bits 8` 即此配置，V 恒 INT4）。
- **on-read RoPE 条件（本轮仅数值，未做硬件）**：pre-RoPE 存储把 RoPE 从 APPEND 一次（~48 cyc/token）
  变为每次窗口读在线旋转（2048 slots × 8 heads × 128 dims ≈ 2.1M 元素/层/步）。现有 Vector 引擎
  ≈131K cyc/层 ≈ **4× 绑定值**（33.5K cyc/层）——无旋转器口径下 decode 变旋转绑定
  （≈28×131K+216K ≈ 3.9M cyc/token → ≈258 tok/s，**低于 866 基线，收益变负**）。**1.27×（及修正档
  1.212×）以专用 on-read 旋转器微硬件（矩阵引擎旁路，≈128 cyc/层）为条件**；该条件不成立则 pre-RoPE
  存储整体不可行。本报告同列两口径，本轮未做硬件实现。

## 7. 判定与后续

- **INT4 KV（B 轨字面）：否决**。双判据均不过（ΔPPL 7.5%/6.9% > 2%；交叉一致 12/20 < 8/10）。
  根因是 4-bit 步长对 0.6B 的 K 太粗，非折叠实现缺陷（INT8 折叠同路径过门槛）。
- **折叠机制：成立（建议保留立项，但目标位宽改 INT8）**。`k_norm` 折出后 per-head 量化恢复 per-channel
  质量，带符号 scale 公式与实测（3.4% 负通道 / min −3.86）逐位一致。
- **修正可达档 = INT8 K（折叠）+ INT4 V = 866→1049（1.212×）**，ΔPPL 1.266%/1.995%、交叉一致 20/20；
  保守档 INT8 K+V（折叠）= 1006（1.162×）。两者均以专用 on-read 旋转器为条件。
- **双尺寸义务（D10）**：本报告仅 0.6B 实测。INT4 K 失败是「小模型 + 4-bit 敏感」的规模效应
  （rot-quant §1 同源：小模型 4-bit 显著掉点）；4B/8B 的 INT4 K 是否可行需放量时重检
  （DECISION §3.2 已预置该重检项）。带符号 scale 的 3.4% 负通道为 Qwen3 QK-norm 家族共性，8B 预期同向。

## 8. 复现

```bash
cd /home/lzl/project/newlpu

# primary：INT4 K 折叠 + INT4 V（否决档）→ docs/perf-research/fold-verify-results.json
python3 qrun/fold_verify.py --baseline docs/perf-research/quality-baseline/quality_baseline.json \
    --sinks 4 --window 2048 --seq-lens 4096,8192 --v-group 128 --k-bits 4 --cross-tokens 20

# 折叠机制锚：INT8 K 折叠 + INT4 V（过门槛档）→ docs/perf-research/fold-verify-int8.json
python3 qrun/fold_verify.py --baseline docs/perf-research/quality-baseline/quality_baseline.json \
    --sinks 4 --window 2048 --seq-lens 4096,8192 --v-group 128 --k-bits 8 --cross-tokens 20

# 天花板（快速，无需模型/GPU）
python3 qrun/fold_verify.py --ceiling-only --out /tmp/fold-ceiling.json
```

前置：HF 模型缓存（`Qwen/Qwen3-0.6B`）+ golden 基线 JSON。ΔPPL 走 torch ref（无需 qbin）；天花板走
`qsim/timing_p6` 冻结常数。数值锚点（7.470%/6.901% / 1.266%/1.995% / 1096.3 / 1049.3）已提交于
`docs/perf-research/fold-verify-results.json` 与 `fold-verify-int8.json`。
