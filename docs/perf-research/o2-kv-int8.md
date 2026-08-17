# O2 INT8 KV 量化 — 数值可行性判定（否决 primary + 备选 + RTL 立项依据）

> 目标（roadmap §2 O2，微硬件项先做数值验证）：KV（K/V）per-128-group 对称 INT8，评估
> windowed(W=2048)+INT8-KV 的 ΔPPL（门槛同 O1 ≤2%）与 20 token 交叉一致（门槛 ≥8/10），
> 并以 KV 读带宽减半（q=0.5）重算 decode 天花板（对照路线图 866→1009 @8K W2048）。
> 口径：HF/ref（torch）参照，executor/qrun 层数值验证，**不改 RTL**。

## 0. 结论摘要

| 判定 | 结果 |
|---|---|
| **ΔPPL 门槛（≤2%，windowed+INT8-KV vs golden full）** | **primary（per-head 对称 INT8）否决**：4K **4.543%** / 8K **5.196%** ≫ 2% |
| 根因 | **K 的 per-channel outlier**（QK-norm 权重诱导，K 通道幅值差 **~300×**）；V 良好（~2.4×） |
| 20 token 交叉一致（vs BF16 full） | 所有量化配置 **12/20**（**与 windowed-BF16 逐 token 相同** → 分歧 100% 来自窗口化，量化零增量） |
| 性能天花板（q=0.5，W=2048） | 路线图 **866→1009（1008.8）逐位复现 ✓**；但该数字对 per-head 量化**不可达**（质量门否决） |
| **备选（门槛判定）** | **仅 V 量化**：ΔPPL **0.641%/1.756% 过**（天花板 866→**930.8**）；**per-256-group**：方向错误（K 需更细非更粗） |
| **硬件立项突破口** | **QK-norm 折叠**：把静态 `k_norm` 权重从被量化 K 中折出（量化单位范数 pre-weight K），per-head INT8 才可恢复 866→1009 |

**门槛判定：O2 按路线图字面方案（per-128-group = per-head 对称 INT8 对 K+V）如实否决——ΔPPL
4.5%/5.2% 远超 2%，根因是 Qwen3 QK-norm 造成的 K per-channel outlier（非实现 bug，非 V 问题）。
近端落地档 = 仅 V 量化（过门槛，866→931，1.075×）；全量 866→1009 需先做 QK-norm 折叠微硬件项。**

## 1. 方案与口径

- **量化**：对称 INT8，`scale = bf16(max|x_group|/127)`、`q = round(x/scale)` clip `[-127,127]`、
  dequant `x≈q·scale`（scale 存 BF16 元数据）—— 与 `qrun.weights._group_scales` / `qforge.quant`
  同一对称校准规则，golden 激活 max-abs 统计。
- **粒度参数化**（`--k-group` / `--v-group`，沿 head_dim=128）：`128`=per-head（1 scale/head/token，
  == 路线图 "per-128-group" 字面义）、`1`=per-channel（per-dim，KVQuant 式）、`0`=BF16 不量化。
- **参照**：golden full-attention PPL 基线（`quality_baseline.json` 的 `per_token_nll`，逐 token 对比）
  + torch ref（`ref/model.py`，已验证与 HF 逐 token 一致）windowed W=2048 forward（`qrun/o2_kv_int8.py`）。
- 验证：windowed-BF16 ΔPPL **0.673%/1.663% 与 O1 报告逐位一致**（ref forward 忠实性锁定的旁证）；
  full-attention greedy decode 与 `ref_greedy_with_logits` **20/20 一致**（decode 路径正确性锁定）。

## 2. 数值结果：ΔPPL 门槛（windowed W=2048 vs golden full，pooled，10 段）

| 配置 | 4K ΔPPL | 8K ΔPPL | 判定（≤2%） |
|---|---|---|---|
| windowed BF16（窗口化 only，O1 参照） | 0.673% | 1.663% | ✅ |
| **per-head KV INT8（primary，k=v=128）** | **4.543%** | **5.196%** | ❌ **否决** |
| 仅 V 量化（k=0, v=128） | **0.641%** | **1.756%** | ✅ |
| per-channel K + per-head V（动态，k=1, v=128） | **0.701%** | **1.776%** | ✅ |
| per-channel K **静态**（128 token 校准） | 66.9%（4K 4 段） | — | ❌（RoPE 使通道幅值 token 相关，静态 scale 不泛化） |

## 3. 根因：K 的 per-channel outlier（QK-norm）

对 layer-0 的 K（post-RoPE）/V 测 128 通道幅值：

| 张量 | 通道 max 范围 | 通道 RMS max/min 比 | per-head INT8 重建 abs 误差 |
|---|---|---|---|
| **K post-RoPE** | **[1.95, 458]** | **~296×** | **1.80**（灾难性） |
| K pre-RoPE | [0.30, 458] | ~2472× | 1.80 |
| **V raw** | [0.33, 2.30] | **~2.4×** | 0.0088（近乎无损） |

- Qwen3 的 **QK-norm**（`self_attn.q_norm`/`k_norm`，per-channel [128] 权重）把 RMS 归一后的 K
  逐通道再放缩：`k_norm` 通道幅值 **max 96.5 / min 0.035 = ~2784×**（layer 0；layer 7 达 180859×，
  `q_norm` 达 5027×）。RoPE 旋转后 K 的通道幅值差达 **~300×**（pre-RoPE 更甚 ~2472×）。
- per-head 对称 INT8（1 个 scale 覆盖全部 128 通道）把小通道压到 0，**单元素重建误差 1.80**（相对 ~1.0），
  直接摧毁 attention score → ΔPPL 5.2%。
- V 无 QK-norm、幅值差仅 2.4×，per-head INT8 近乎无损（0.0088）——**V 是"干净"的，K 是问题所在**。

## 4. 消融：K vs V（4 段 @4K 子集）

| 消融 | ΔPPL@4K |
|---|---|
| windowed BF16（无量化） | 1.015% |
| K-only per-head INT8 | **5.321%**（主导误差） |
| V-only per-head INT8 | 0.975%（≈ 无量化，+0.0%） |
| KV per-head INT8 | 5.332% |

**结论：ΔPPL 失败 100% 来自 K 量化，V 量化近乎免费。**

## 5. 20 token 交叉一致（4K 重复文本 prompt，vs BF16 full）

| 配置 | 一致率 | 分歧位置 | per-pos rel err（分歧点） |
|---|---|---|---|
| windowed BF16（W=2048） | **12/20** | [12..19] | pos12=0.195 |
| per-head KV INT8 | **12/20** | [12..19]（**与 windowed-BF16 逐 token 相同**） | pos12=0.214 |
| 仅 V / per-channel-K | 12/20 | [12..19]（同上） | — |

- **量化对 greedy 轨迹零增量**：所有量化配置的 20 token 与 windowed-BF16 **逐 token 相同**
  （`windowed_quant == windowed_bf16: True`）。12/20 的分歧 100% 由 **W=2048 窗口化** 引起
  （pos12 是 near-tie，rel err 0.195 翻转 argmax 后级联）。
- **与 O1 报告的差异（如实记录）**：O1 §5 在 qrun numpy 执行器上测 W=2048 = 20/20；本报告在 torch
  ref 上测 12/20。二者窗口语义一致（本报告 prefill windowed NLL 与 HF 逐位一致、full decode 与
  reference 20/20 一致），差异来自 greedy decode 对 BF16 舍入的混沌敏感（重复文本 prompt 的 near-tie
  在不同执行器上翻转不同）。**O2 的交叉一致口径应改为「量化 vs windowed-BF16」隔离**（= 20/20，
  量化不可见），而不是「vs full」混入窗口化误差。

## 6. 性能天花板（qsim 时序模型，q=0.5，W=2048，R=S+W=2052）

| 配置 | KV 字节/token/layer | overlap tok/s | db tok/s |
|---|---|---|---|
| full @4K / @8K | 4096 | 487.5 / 257.3 | 481.1 / 255.2 |
| windowed BF16（O1 档） | 4096 | **866.0** | 845.9 |
| **INT8-KV q=0.5（K+V 全减半）** | 2048 | **1008.8 ≈ 1009 ✓** | 981.6 |
| 仅 V 量化（K BF16 2048 + V INT8 1024 + scale 16） | 3088 | **930.8** | 907.7 |
| per-channel K 动态（data 2048 + scale 2048） | 4096 | **866.0（无净收益）** | — |

- **866→1009 逐位复现**（1008.8，q=0.5 纯减半口径；含 32 B/token scale 元数据为 1006.2，二阶小量）。
- 但该数字对 **per-head 量化不可达**（质量门否决）。**仅 V 量化**只减半 V → 天花板 866→**930.8（1.075×）**。
- **动态 per-channel K 的 scale 元数据 = 128 scale/head/token = 2048 B/token = 数据本身 2×**，
  读带宽回到 BF16 口径（866，零净收益）——**per-channel K 必须静态化才可压缩，而静态化已证失败（§2）。**

## 7. 判定与备选

- **primary（per-128-group = per-head 对称 INT8 K+V）否决**：ΔPPL 4.5%/5.2% > 2%。
- **备选 ① 仅 V 量化（推荐近端落地）**：ΔPPL 0.641%/1.756% 过门槛；交叉一致与 windowed-BF16 相同
  （量化零增量）；天花板 866→**930.8**（非 1009）。RTL 改动最小（只 V 的地址步长 + scale）。
- **备选 ② per-256-group：否决**。方向错误——K 需要**更细**（per-channel）而非更粗的粒度；
  per-head(128) 已 5.3%，256 只会更差；且 head_dim=128，256 无法沿 head_dim 分组。
- **突破口（硬件立项）＝ QK-norm 折叠**：K 的 ~300× 通道幅值来自**静态** `k_norm` 权重。若把量化点
  移到 QK-norm 权重相乘**之前**（量化单位 RMS 的 pre-weight K，幅值有界 ≈[0,11]），`k_norm` 作为
  静态 per-channel scale 在 KV.LOAD 时施加（dequant 复用），per-head INT8 即可恢复，全量 866→1009 可达。
  这是比"per-channel 动态 scale（元数据=数据 2×）"更省的路径，应作为下一步微硬件立项核心。

## 8. RTL 改动清单（立项依据）

1. **KV 地址步长 2B→1B**（`rtl/kv_addrgen.sv`）：`out_base` 的 `pos_start << 8` → `<< 7`、
   `out_len = count*256` → `count*128`（INT8 数据 1 B/element）。对 V-only 与全量 INT8 均成立。
2. **scale 元数据**：KV.LOAD 需额外读 per-group BF16 scale。
   - V：per-head（1 scale/head/token，8×2B=16 B/token/layer）——V-only 档即够。
   - K：**per-head 不够**（§3）；需 QK-norm 折叠后的静态 per-channel scale（8×128×2B=2 KB/layer，
     共享全序列）或 per-channel 动态（不可压缩，否决）。
3. **dequant 复用**：复用 matrix_engine 现有 per-128-group dequant 数据通路（`i32_to_f32`/DEQUANT
   + fp32 scale 相乘，`rtl/matrix_engine.sv`、`rtl/vector_engine.sv` QUANT/DEQUANT 已落地），
   KV 读回后 `int8 × scale → BF16` 再入 attention BMM；无需新数据通路。
4. **KV dtype 字段**（ISA 05 §4 已预留 KV dtype code，v0 固定 BF16）：解锁 `dtype=INT8` +
   scale 描述符寻址，与 `CD dequant 描述符`（`[20]mode [19]scale_dtype [18:0]scale_base`，p2/dialects）
   对齐。
5. **（关键前置）QK-norm 折叠**：`k_norm` 从被量化 K 中折出，量化点前移。这是 O2 全量兑现 1009
   的必要 RTL 改动，超出路线图"小硬件"字面范围，需单独立项。

## 9. 需评审项

1. **O2 门槛口径修正**：交叉一致应改为「量化 vs windowed-BF16」（隔离窗口化误差），而非「vs full」；
   否则 W=2048 窗口化本身的 12/20（torch 口径）会误判量化。是否接受此口径修正？
2. **O1 cross-consistency 的 20/20 与本节 12/20 差异**（qrun numpy vs torch 的 BF16 舍入敏感性）
   是否需 O1 侧回注（O1Fix 正在重跑，可对照）。
3. **V-only 是否作为 O2 落地档**（收益 1.075×，远低于路线图 1.17×），还是等待 QK-norm 折叠立项后
   全量兑现 1009。
4. **QK-norm 折叠立项**：量化点前移 + 静态 per-channel scale（2 KB/layer）是否进入下一批微硬件项；
   需重验 pre-weight K 的单位幅值上界与 per-head INT8 的 ΔPPL（本报告未实现该折叠路径，仅定位根因）。
5. **双尺寸义务（D10）**：本报告仅 0.6B 实测；根因（Qwen3 QK-norm 的 per-channel outlier）为 Qwen3
   家族共性，8B 亦含 QK-norm，预期同向但幅度未测——8B 口径的 ΔPPL/天花板需在 QK-norm 折叠立项后
   一并重跑（roadmap §3.5）。


## 10. 复现

```bash
cd /home/lzl/project/newlpu
# primary（否决）
python3 qrun/o2_kv_int8.py --baseline docs/perf-research/quality-baseline/quality_baseline.json \
    --sinks 4 --window 2048 --seq-lens 4096,8192 --k-group 128 --v-group 128 --out /tmp/o2-primary.json
# 仅 V 量化（过门槛备选）
python3 qrun/o2_kv_int8.py --baseline docs/perf-research/quality-baseline/quality_baseline.json \
    --sinks 4 --window 2048 --seq-lens 4096,8192 --k-group 0 --v-group 128 --out /tmp/o2-vonly.json
# per-channel K + per-head V（过门槛，但动态 scale 不可压缩）
python3 qrun/o2_kv_int8.py --baseline docs/perf-research/quality-baseline/quality_baseline.json \
    --sinks 4 --window 2048 --seq-lens 4096,8192 --k-group 1 --v-group 128 --out /tmp/o2-kchannel.json
# §6 天花板六行（快速，无需模型/GPU）：V-only 930.8、per-channel-K 动态 866.0、q05 1008.8 等
python3 qrun/o2_kv_int8.py --ceiling-only --out /tmp/o2-ceiling.json
# 静态 per-channel K（§2 末行，66.9% 复现）→ docs/perf-research/o2-kv-int8-results.json
python3 qrun/o2_kv_int8.py --static-kchan --skip-ppl --skip-cross
```

前置：HF 模型缓存（`Qwen/Qwen3-0.6B`）+ golden 基线 JSON。ΔPPL 走 torch ref（无需 qbin）；天花板走
`qsim/timing_p6` 冻结常数。天花板六行与静态 per-channel K 的数值锚点（930.8 / 866.0 / 66.9%）已提交于
`docs/perf-research/o2-kv-int8-results.json`。
