# B' 落地 — INT8-K（QK-norm 折叠）+ INT4-V 的 KV 量化通路（微硬件 + 全链路）

> 目标（DECISION §5 裁决 / fold-verify）：把已过数值门槛的 **B' = INT8-K（QK-norm 折叠，带符号
> per-channel scale）+ INT4-V（per-token per-head）** 从数值验证落到 ISA 协议 / runtime / RTL
> 微硬件，并以两口径（有/无 on-read RoPE 专用旋转器）如实报告性能天花板。

## 0. 结论摘要

| 判定 | 结果 |
|---|---|
| **数值门槛（ΔPPL ≤2% + 交叉 ≥8/10）** | ✅ 复跑通过：ΔPPL **1.266% / 1.995%**（4K/8K，vs golden full）；交叉一致 **20/20**（隔离口径 fold vs windowed-BF16） |
| **性能天花板（qsim 时序模型，W=2048，R=S+W=2052）** | windowed-BF16 **866.0** → INT8-K 折叠 + INT4-V **1049.3（1.212×）**；保守 INT8 K+V 折叠 1006.2（1.162×） |
| **两口径（on-read RoPE 条件）** | **有专用旋转器**：1049.3（1.212×，条件成立）；**无旋转器**：≈257 tok/s（decode 变旋转绑定，低于 866 基线，收益变负） |
| **ISA/协议** | KV 指令通用头 `srcA[111:109]`=K dtype、`srcB[108:106]`=V dtype 解锁；`srcA/srcB=0`（BF16）保持 v0 逐字节向后兼容（只读引用 05 §4 dtype flags / C15 flags 编码） |
| **scale 元数据布局** | K 静态折叠 scale（`k_norm`，带符号 BF16）128×2B/head = **2 KB/layer，56 KB SRAM 常驻**；per-token `[s_q(2B), s_v(2B)]` = **4 B/token/(layer,head)**，HBM scale slab（8192×4 B=32 KB/(layer,head)） |
| **RTL 微硬件** | `rtl/kv_quantdequant.sv`（quant-on-write / dequant-on-read，复用 softfloat fp32 核）+ command_processor S_KVQD 通路；专用 co-sim 两用例 **0 ULP + trace 一致** |
| **qsim 功能级** | `qsim/test_bprime_kv.py`：K 折叠 / V INT4 dequant vs fp64 双轨 `<1e-6`，BF16 写回 ≤1 ULP，全通 |
| **默认回归** | qsim pytest 47 例全通；co-sim / golden3 逐周期不变（B' 为新增通路，默认参数路径未改动） |

**落地判定：B' 数据通路（量化/dequant/scale 元数据）已全链路落地并验证；1049 天花板仍以
on-read RoPE 专用旋转器为条件（本轮未落地该旋转器，如实列为需评审项）。**

## 1. 方案与口径

- **K 折叠（INT8）**：`k_unit = rmsnorm(k_raw)`（单位 RMS，pre-weight）；`q = round(k_unit/s_q)`，
  `s_q = bf16(max|k_unit|/127)`（正，per-head-per-token）；dequant `k_hat[c] = q[c] × (s_q × k_norm[c])`，
  `scale_c = s_q × k_norm[c]` **带符号** BF16（k_norm 可为负，实测 3.404% 负通道，min −3.859）。
  K 存 **pre-RoPE**（折叠要求量化点前移），RoPE on-read 施加。
- **V（INT4）**：`s_v = bf16(max|v|/7)`（per-head-per-token），`q4 = round(v/s_v)` clip `[-7,7]`，
  dequant `v_hat = q4 × s_v`。
- **字节预算（0.6B，8 KV heads，128 dim）**：K INT8 1024 + V INT4 512 = **1536 B/token/layer**；
  scale = K `s_q`(8×2B=16) + V `s_v`(16) = **32 B/token/layer**；静态 `k_norm` 2 KB/layer SRAM。
- **参照**：qsim executor（`qsim/executor.py` `_kv`）为功能黄金；fp64 双轨（M3 口径）；RTL co-sim
  与 executor 逐字节/≤1 ULP 对齐。

## 2. ISA / 协议侧

KV 指令此前 dtype 固定 BF16（05 §4「`[111:104]` dtype flags 在 v0 固定 BF16」）。B' 复用该通用头：

| 字段 | 语义 | 取值 |
|---|---|---|
| `srcA[111:109]` | K dtype | `0`=BF16（默认，向后兼容）；`2`=INT8（QK-norm 折叠） |
| `srcB[108:106]` | V dtype | `0`=BF16；`3`=INT4 |

`srcA/srcB=0` 保持 v0 逐字节路径不变（`qsim/test_bprime_kv.py::test_kv_bf16_default_unchanged` 验证）。
无新增指令（33 指令集不变）；scale 元数据寻址复用既有 AR/C 寄存器 ABI（`AR62`=per-token scale
slab HBM base、`C29`=静态 `k_norm` SRAM word base），dequant 复用 CD 描述符模式（`[20]mode [19]scale_dtype
[18:0]base` 同构）。此「dtype 组合解锁」为只读引用既有条款，不改 spec 分册。

### scale 元数据布局（bit 精确）

```
静态 k_norm（SRAM，56 KB）：addr = C_KVNORM_BASE×16 + (layer×8+head)×256    // 128 带符号 BF16
per-token scale（HBM）：    addr = AR_KV_SCALE_BASE + (layer×8+head)×32768 + pos×4
                               [s_q (2B BF16)][s_v (2B BF16)]
K data slab（INT8）：       addr = kv_base + (slab_index(K)<<SLAB_SHIFT) + pos<<7   // 128 B/token
V data slab（INT4）：       addr = kv_base + (slab_index(V)<<SLAB_SHIFT) + pos<<6   // 64 B/token packed
```

## 3. runtime（qrun）落地

`qsim/executor.py` 的 `_kv` 现支持 dtype 组合的 **quant-on-write / dequant-on-read**：

- `KV.APPEND` / `KV.STORE_BLOCK`（写）：BF16 K/V staging → K 折叠 INT8 + `s_q`、V INT4 + `s_v` → HBM
  数据 slab + scale slab。
- `KV.LOAD` / `KV.GATHER`（读）：HBM INT8/INT4 + scale + 静态 `k_norm` → dequant → BF16 staging
  （pre-RoPE；on-read RoPE 由程序既有 ROPE op 施加）。
- 叠加 O1 窗口化：窗口读 `KV.LOAD` 走 dequant；与 `windowed_kv` 的 W=2048 窗口正交（KV 元素 dtype
  与窗口 mask/分 tile 独立）。

dequant 数值与 `qrun/fold_verify.py` 的 torch 折叠逐位一致（同带符号 `scale_c = s_q×k_norm[c]`，
BF16 舍入忠实）。

## 4. RTL 微硬件落地

- **`rtl/kv_quantdequant.sv`**（新）：单 head（128 dim）× 单张量（K/V）的 quant/dequant 变换，
  FSM 复用 softfloat `i32_to_f32 / f32_to_i32_rne / fp32_mul / fp32_div / fp32_to_bf16`（同
  W4A16/INT8 尾数路径）。注意两处落点忠实性修正：
  1. BF16 读回须 `bf16_to_fp32`（此前零扩展位型被当 denormal，`amax`/`kn_f` 全错 → 见 §6 评审项 3）；
  2. `f32_to_i32_rne` 对 `|x|∈[0.5,1.0)` 返回 0（冻结 softfloat 的 RNE 漏洞），本模块以局部
     `quant_rne` 修正（不改冻结 softfloat，默认回归契约）。
  3. scale 在 BF16 舍入前按 executor 契约 clamp 到 float32 `1e-6`，全零 K/V 不再生成零 scale。
  4. Track 2.1 将四轮 Newton-Raphson 倒数按 `mul/sub/mul` 逐操作寄存，并将 element multiply
     与 RNE/clip 分级；保持原 softfloat 运算次序和一元素/周期稳态吞吐。SMIC28 全顶层
     data arrival：tt **4.59→1.57 ns**（217.9→636.9 MHz），ss **6.08→2.12 ns**
     （164.5→471.7 MHz）；均为 1 ns ideal-clock 综合探针，双角 top-10 均无量化器路径。
- **`rtl/command_processor.sv`**：KV 指令解码 dtype → 新增 `S_KVQD` 状态，逐 (tok, phase) 驱动
  kvqd 模块；BF16 张量走既有 DMA 回退；`KV.LOAD sel` 全组合（0=K/1=V/2=both）正确路由。
- **`rtl/qcore_pkg.sv`**：新增 `AR_KV_SCALE_BASE=62`、`C_KVNORM_BASE=29` 常量。
- **`rtl/tb/run_cosim_bprime.py`**：专用 B-feed co-sim 三用例（sink / rolling window / PF），
  覆盖 quantize-on-write 后的 INT8-K 旋转与 INT4-V 去量化，判据 trace+total 一致 + BF16 输出 ≤1 ULP。

## 5. 验证

| 层 | 用例 | 结果 |
|---|---|---|
| qsim 功能级 | `qsim/test_bprime_kv.py`（K 折叠/V INT4 dequant vs fp64 双轨 `<1e-6`，BF16 ≤1 ULP，roundtrip，scale 布局，BF16 默认不变） | **5/5 全通** |
| RTL co-sim | `rtl/tb/run_cosim_bprime.py`（sink / rolling window / PF） | **3/3 PASS，0 ULP，trace/cycles 一致** |
| RTL quant 定向 | 随机 + 全零 K/V，比较 INT8-K 128 B + INT4-V 64 B + scales 4 B | **两例 196 B 均与 executor 字节级一致** |
| qsim 回归 | `pytest qsim/test_isa_fields.py qsim/test_vector_kv.py qsim/test_int4.py` | **47 passed** |
| timing | `qsim/timing_p6.py`（ctx 4096/8192 双 PASS） | **PASS** |
| 数值复跑 | `qrun/fold_verify.py --k-bits 8`（ΔPPL + 交叉 + ceiling） | 复跑（见 §0） |

**回归注记（B' P0/P2 修复）**：`rtl/command_processor.sv` S_FETCH DMA 分支恢复
`dma_row_bytes_r <= imem[pc][91:76]` 与 `dma_num_rows_r <= imem[pc][54] ? imem[pc][75:60] : 16'd1`
两行（对齐 02-isa 字段位 RowBytes[91:76] / NumRows[75:60] / mode[54]）后——
(a) Verilator lint（top-module qcore_top / command_processor）无 UNDRIVEN；
(b) M2a linear（PF BF16）co-sim **PASS**（trace/cycles 一致，ulp_normal=1.0）；
(c) `run_cosim_bprime.py` 两例 ALL PASS（0 ULP，trace/cycles 一致）；
(d) vector 22 例 + KV 2 例子集 ALL PASS；
(e) `--ceiling-only` JSON 含 `int8_k_fold_int4_v`=**1049.3**、`int8_kv_fold`=**1006.2** 两行。

## 6. 需评审项

1. **on-read RoPE 专用旋转器未落地**：1049.3（1.212×）以矩阵引擎旁路专用旋转器（≈128 cyc/层）为
   条件；本轮仅落地 dequant 数据通路，on-read RoPE 由既有 vector ROPE op 施加（131K cyc/层）。
   无旋转器口径 ≈257 tok/s（< 866 基线，pre-RoPE 存储整体收益变负）。**主口径 = 无旋转器**，1049
   为条件口径，不虚报。
2. **`f32_to_i32_rne` 冻结软核 RNE 漏洞**：`|x|∈[0.5,1.0)` 返回 0 而非 1（round-to-nearest-even）。
   本模块以局部 `quant_rne` 规避；该漏洞影响既有 VECTOR `QUANT` op 的该区间，是否本轮一并修正需
   单独裁决（修它会影响 golden 一致性，风险自负）。
3. **BF16 读回零扩展陷阱**：co-sim 引擎读取 2B 数据后必须 `bf16_to_fp32`，零扩展位型会被当作
   denormal（幅值 ~2^-126），导致 max/scale 归零。`kv_quantdequant` 已修，建议作为 RTL 编码规范
   写入 docs/p7（若仍维护）。
4. **双尺寸义务（D10）**：0.6B 实测 1.266%/1.995% 过门槛；4B/8B 放量时需重检带符号 scale 负通道
   比例与 INT8-K 折叠 ΔPPL（Qwen3 QK-norm 家族共性，预期同向）。
5. **KV.GATHER 量化**：RTL 通路仅落地 APPEND/STORE_BLOCK/LOAD；`KV.GATHER` 携带非 BF16 dtype 时
   仍走 DMA（BF16）路径（0.6B decode 走 LOAD 单副本，GATHER 为非默认路径）。executor 层支持
   GATHER 量化，RTL 未覆盖，列为 backlog。

## 7. 复现

```bash
cd /home/lzl/project/newlpu
# qsim 功能级
python3 qsim/test_bprime_kv.py
# RTL co-sim（需先构建 obj_dir/Vqcore_top）
python3 rtl/tb/run_cosim_bprime.py
# 数值复跑（INT8-K 折叠 + INT4-V）
python3 qrun/fold_verify.py --baseline docs/perf-research/quality-baseline/quality_baseline.json \
    --sinks 4 --window 2048 --seq-lens 4096,8192 --v-group 128 --k-bits 8 --cross-tokens 20 \
    --out /tmp/bprime-fold-int8.json
# 天花板（快速，无需模型）
python3 qrun/fold_verify.py --ceiling-only --out /tmp/bprime-ceiling.json
# 默认回归
python3 -m pytest qsim/test_isa_fields.py qsim/test_vector_kv.py qsim/test_int4.py -q
python3 qsim/timing_p6.py
```
