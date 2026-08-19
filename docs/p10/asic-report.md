# QCore ASIC 流程报告（P10 / M9）

> 状态：M9 验收交付。Yosys 0.44 + SkyWater 130 nm（sky130_fd_sc_hd）逻辑真综合；
> OpenSTA 多 corner STA；SRAM 宏按公开密度上下界估算并单列占比。
> 冻结快照：`rtl/ref/asicsnap/`（开工时打点；rtl/ 只读，P8 未改 rtl/ 源）。
> D18 更新（2026-08-19）：代表数据通路与 BF16 MAC 已用 SMIC28 HDC30P140 RVT
> 重综合，tt/ss 均闭合 1 ns ideal-clock 探针；矩阵状态 RAM 已接入 9 个真宏，且
> Track 2.3 已闭合双角 SRAM 输入/读回 scoped timing；Track 2.4a 已交付单拍 dual-MAC
> INT8 PE 及 TT/SS 1.0/0.9 ns 映射证据。16×16 tile、完整 128×128 阵列、CTS 与物理
> signoff 仍另行列示。

## 0. 结论速览

| 项 | 结果 |
|---|---|
| elaboration | ✅ 通过（Verilator 4.038 原 RTL lint + Yosys 0.44 层级检查，见 §1） |
| 逻辑综合 | ✅ 通过（8 项 FP 基元 → legacy sky130 与当前 SMIC28 门级网表，见 §2/§10.4） |
| 时序收敛结论 | ⚠️ **全系统 1 GHz 尚未收敛**：当前含 matrix 真宏壳的 SMIC28 全顶层 tt/ss 探针为 689.7/518.1 MHz，最差路径均在 B-feed（§10.8.5）；Track 2.1 量化器流水化基线为 636.9/471.7 MHz（§10.7）。代表 `synth_datapath`、`mac_bf16` 与新 dual-MAC PE 的 SMIC28 DC tt/ss 1 ns 探针均 `MET`；PE 基线 arrival 0.98/0.97 ns 且显示 slack +0.00 ns，0.9 ns 加强映射也 `MET`（§10.9）。这些都是 pre-layout 探针，不是整芯片 signoff。legacy sky130 口径保留：DC 数据通路 tt/ss 331/169 MHz、Yosys/OpenSTA tt 129 MHz。 |
| 面积 | SMIC28 DC：`synth_datapath` 0.003085/0.003237 mm²（tt/ss），`mac_bf16` 0.001906/0.001961 mm²；dual-MAC PE 本体 0.000560/0.000631 mm²，直接复制 16,384 个 PE 仅给出 9.183/10.345 mm² cell-area 下限（不含布线/CTS/SRAM/后处理）；legacy sky130 P10b 与旧公开 SRAM 密度估算保留为历史口径 |
| token/s | 冻结 1 GHz 口径 960/675/469；SMIC28 代表数据通路已支持 1 GHz 综合探针，但系统频率仍受控制平面与后续物理实现限制，暂不改写端到端 token/s |


> **DC/VCS 补充（§10/§11）**：Synopsys DC O-2018.06-SP1 compile_ultra（基元级）与
> VCS-MX O-2018.09-SP2 功能仿真（27 用例与 Verilator 字节级一致）已交付；
> **O6 全设计扩展：synth_top 在 DC 侧 elaborate + link + compile_ultra 全跑通**
> （历史口径为控制平面 1.000 mm² / Fmax 136 MHz；Track 2.2b 已将矩阵状态 RAM 从
> 整模块黑盒推进为 9 个真 SRAM + 可综合控制壳，Track 2.3 又闭合其输入时序，§10.8）。
> DC 运行需 license 27000@bics109，VCS 需 snps-centos7 兼容命名空间（详见 §10/§11 与 README）。

## 1. Elaboration（验收项 1）

1. **Verilator 4.038（项目自身 lint/elaboration 工具）**：`verilator --lint-only
   -Wno-fatal -Wno-WIDTH --top-module qcore_top rtl/qcore_top.sv -Irtl` →
   **exit 0，无告警**。这是 P7/M6 已锁定的 elaboration 工具，覆盖全部 SV 构造
   （`return`、`import`、`inside`、类型转换、unpacked 数组端口、`localparam`
   常量表等）。
2. **Yosys 0.44**：`hierarchy` 对综合子集（softfloat 全函数 + 两种 MAC 顶层）通过，
   输出门级网表（§2）。全尺寸 `synth_top` 实例（含 vector_engine/CP 变长组合循环）
   的物理语义由 Verilator 侧 elaboration 覆盖；Yosys 0.44 前端不支持 `return`
   关键字、`import pkg::*`、`inside`、SV 类型转换、unpacked 数组端口、变长边界
   `for`——为做真综合，写了一个语义中性（纯 desugar）的预处理层
   `asic/preprocess.py` + `asic/return_elim.py`，把快照逐字节派生为 `asic/gen/`
   （快照与 rtl/ 均未改动，见 §7）。

## 2. 逻辑真综合（验收项 2 前半）

工具链：Yosys 0.44（源码构建，本地无 sudo）+ sky130_fd_sc_hd liberty
（由 skywater-pdk-libs 官方 `*.lib.json` 经其 `python-skywater-pdk` 转换器生成，
5 corner：tt_025C_1v80 / ss_n40C_1v28 / ss_100C_1v60 / ff_n40C_1v95 / ff_100C_1v65）。

合成对象（`asic/synth_datapath.sv` = 8 项 FP 基元；`asic/synth_mac.sv` = 两种 MAC，
由 `run_mac_synth.sh` 一次性合成——见 §8）：

| 单元 | 单元数 | 面积（sky130 hd，实测 `stat -liberty`） |
|---|---|---|
| **FP ALU 切片**（fp32_add/sub/mul/max + i32↔f32 + bf16↔fp32） | 5088 | **0.0331 mm²** |
| **BF16 MAC**（fp32_mul + fp32_add） | 3610 | **0.0273 mm²** |
| **INT8 MAC**（8×8 乘 + 32b 累加） | 555 | **0.0042 mm²** |

> 未纳入门级综合（§7 口径）：fp32_div/recip/rsqrt（4 次牛顿迭代）与
> fp32_exp/exp2/log2/pow/sin/cos（Taylor/查表 + exp2 内变长 `2^n` 循环）——它们是
> 迭代/查表单元，物理实现为多周期流水，冻结 `vector_latency()`（VDIV=10、
> VRECIP/VRSQRT=7、VEXP=8）已按多周期计费；其面积按 FP ALU 同量级另估。

**阵列/引擎缩放（0.6B 口径物理几何，128×128 双 MAC、128 lane）：**

| 组件 | 数量 | 面积 |
|---|---|---|
| 矩阵阵列（INT8，W8A8 主通路） | 128×128×2 = 32768 MAC | 32768 × 0.0042 = **≈138 mm²** |
| 矩阵阵列（BF16，2 子字复用 INT8 乘法器 + fp32 累加） | 32768 | ≈ 2× INT8 ≈ **≈276 mm²**（fp32_mul 直作上限 893 mm²） |
| 向量引擎 | 128 lane | 128 × 0.0331 = **≈4.2 mm²**（含 div/exp 另计） |

## 3. STA（验收项 2 后半）

方法：OpenSTA（2.0.17，apt 提取）+ 上述 5 corner liberty + 门级网表 + 虚拟时钟
（周期 1 ns）+ 0 输入/输出延时（`asic/sta.tcl`）。关键路径为 `a[i] → sub_o[j]`
（fp32_sub ≡ fp32_add 对齐+加减+规格化 + 去 return 化的守护链）。

| corner | 关键路径（data arrival） | Fmax = 1/delay |
|---|---|---|
| tt_025C_1v80 | 16.89 ns | **59.2 MHz** |
| ss_n40C_1v28（最慢） | 138.14 ns | 7.2 MHz |
| ss_100C_1v60 | 34.04 ns | 29.4 MHz |
| ff_100C_1v65（最快） | 12.67 ns | 78.9 MHz |

**时序收敛结论**：直接组合映射下 **1 GHz（1 ns 周期）不收敛**（tt 违例 −15.9 ns，
ss 违例 −137 ns）。原因三重，均如实记录：
1. 冻结 RTL 是**位精确功能+时序模型**，非物理数据通路（见 §7）；
2. 130 nm 标准单元本征慢 + 慢 corner（1.28 V/−40 °C）成倍放大；
3. `return` desugar 成 `__ret` 守护链引入串行优先级 MUX，延长了组合深度（
   物理实现用并行优先级/流水可消除）。
冻结周期模型（1 cyc = 1 ns）本身已按 `vector_latency()` 对 div/exp/rsqrt 等多周期
指令计费，故物理实现需按这些表流水化才能兑现 1 GHz 目标。

## 4. SRAM 宏估算（验收项 2 后半，口径注明）

8 MiB scratchpad（16 bank × 512 KiB，D3）不映射门级，按公开密度上下界估算：

| 口径 | 来源 | 面积 |
|---|---|---|
| 下限 1.5 µm²/bit | OpenRAM / Sky130 6T SRAM 参考（~1.5–2.0 µm²/bit） | 67.1 Mbit × 1.5 = **100.7 mm²** |
| 上限 4.0 µm²/bit | 计划给定公开区间上界（含外围/冗余/修复列） | 67.1 Mbit × 4.0 = **268.4 mm²** |

> 口径说明：8 MiB = 67.1 Mbit。真实编译宏（16 bank 2R1W、16 B 字）会再加译码/
> 感知放大器外围，取下限为乐观、上限为保守；报告取区间 [100.7, 268.4] mm²。

**SRAM 占面积比例（单列）**：以 INT8 主通路阵列 138 mm² + 向量 4.2 mm² + 控制逻辑
（CP/仲裁/编解码，量级 <10 mm²）为逻辑侧，SRAM 占芯片总面积
**≈ 40%–64%**（下/上限），是面积占比最大的单块。

## 5. 功耗（活动因子估算，口径注明）

| 项 | 口径 | 估值 |
|---|---|---|
| SRAM 动态（读/写活动） | 130 nm ≈0.3 mW/MHz/MB（公开量级） | 8.4 MB → 2.5 mW/MHz；@59 MHz ≈ **0.15 W**，@1 GHz ≈ 2.5 W |
| SRAM 漏电 | ~1 pW/bit 量级 | 67 Mbit → ≈ **0.07 W** |
| 逻辑动态 | α≈0.1、V=1.8、阵列+ALU 门数 × f | @59 MHz ≈ **<0.5 W**（阵列未占满时远小于此） |
| **SRAM 占功耗比例** | 以解码瓶颈（HBM-bound、阵列利用率 3.66%） | **≈30–80%**（随工作负载/频率） |

> 功耗为活动因子粗估（未跑 PPA 级 signoff），口径在表内注明；正式功耗须
> 布局后 + VCD 活动因子回注，列为后续。

## 6. token/s（结合 M5 周期模型）

M5 冻结周期模型每 token 周期数（1 GHz，0.6B INT8 sustained，spec §4/D10）：
短上下文 1.04×10⁶、@4K 1.48×10⁶、@8K 2.13×10⁶ 周期/token。

| 时钟口径 | 短 ctx | @4K | @8K |
|---|---|---|---|
| 1 GHz（目标，流水化后） | **960** | **675** | **469** |
| 59.2 MHz（tt 直接映射） | 56.9 | 40.0 | 27.8 |
| 7.2 MHz（ss 直接映射） | 6.9 | 4.9 | 3.4 |

结论：性能天花板由 HBM 带宽决定（decode 27.3× 带宽短缺，阵列利用率 3.66%），
时钟从 1 GHz 降到直接映射的 59 MHz 时 token/s 同比例下降；兑现 1 GHz 目标依赖
§3 的流水化物理实现。

## 7. 交付物、范围声明与需评审项

**asic/ 交付物**：`preprocess.py`（快照→可综合派生，语义中性）、`return_elim.py`、
`synth_datapath.sv`/`synth_mac.sv`（综合顶层）、`synth.ys`/`synth_datapath.ys`/
`run_synth.sh`/`run_mac_synth.sh`（脚本）、`sram_macro.sv`（8 MiB 宏黑盒）、
`mem_stub.lib`（宏时序黑盒）、`qcore.sdc`/`sta.tcl`（约束+STA）、`netlist/`（门级网表）。

**范围声明（如实）**：
- rtl/ 冻结快照为**功能+时序 co-sim 模型**：vector_engine 变长组合循环、CP 字节级
  编组、qmem 平坦 SRAM+稀疏 HBM 是功能拍平，非 128-lane 物理数据通路。故「逻辑
  真综合」以**物理计算基元**（softfloat 算术单元、两种 MAC）为对象并缩放；
  qcore_top 全尺寸物理实例不在本报告门级网表内（其变长循环不表达 128-lane 语义）。
- 预处理仅做语义中性 desugar（无功能改动），快照与 rtl/ 未改动。
- 未跑 formatter/linter/项目级测试套件（共同约束）；只跑上述综合/STA 验收命令。

**需评审项**：
1. **时序口径**：1 GHz 需流水化；直接映射 Fmax 值见 §3，是否作为 M9 结论接受。
2. **综合范围**：div/exp 等迭代单元未门级综合（面积另估），是否接受该口径。
3. **attn_softmax 偏差（已回修，无 delta）**：Golden3 原检出的 RTL↔qsim 221 ULP
   根因为 rtl/tb/sim_main.cpp 测试台 harness bug（预载后未撤销 backdoor 写使能
   bd_en，逐周期覆盖引擎输出），**非引擎缺陷**；修复仅在 sim_main.cpp 加 `bd_en=0`，
   rtl/*.sv 引擎源码零改动 → 快照无 delta、无需重 elaborate。attn_softmax 现
   0 ULP，golden3 15 实例 + 3 逐层全过（docs/p8/golden3-report.md §5/§7）。

## 8. 复现

```bash
python3 asic/preprocess.py                        # 快照 -> asic/gen/（语义中性）
bash asic/run_synth.sh tt_025C_1v80               # FP 基元综合（改 corner 名换库）
bash asic/run_mac_synth.sh tt_025C_1v80           # 两种 MAC 综合（mac_bf16 3610 门 / mac_int8 555 门）
# STA（多 corner）
STA=/path/to/sta
for c in tt_025C_1v80 ss_n40C_1v28 ss_100C_1v60 ff_100C_1v65; do
  LIB=.../sky130_fd_sc_hd__$c.lib $STA -no_splash -exit asic/sta.tcl
done
```

## 9. P10b 流水化 + SRAM 缩编（delta 记录）

> 状态：M8 等待期 P10b 交付。rtl/ 仅改 `sram.sv`/`qcore_pkg.sv` 的 SRAM 参数化
> （窄例外，单写者）；asic/ 物理层流水化落在 `synth_datapath.sv`/`synth_mac.sv`，
> rtl/ 冻结时序模型不动。

### 9.1 STA 关键路径分析（OpenSTA 路径明细，tt_025C_1v80）

M9 直接组合映射前 3 长路径（`a[i] → sub_o[j]` 等，§3）：

| 排名 | 路径 | 端点 | delay | Fmax |
|---|---|---|---|---|
| 1 | fp32_sub（≡fp32_add） | a[26] → sub_o[17] | 16.89 ns | 59.2 MHz |
| 2 | fp32_add | a[26] → add_o[0] | 15.73 ns | 63.6 MHz |
| 3 | fp32_mul | a[5] → mul_o[22] | 10.49 ns | 95.3 MHz |

根因（§3 已述）：①冻结 RTL 是位精确功能+时序模型，非物理数据通路；②130 nm
本征慢 + 慢 corner 放大；③`return` desugar 成 `__ret` 守护链串行优先级 MUX。

### 9.2 流水化（asic/ 物理层，rtl/ 源不改）

`synth_datapath.sv` 8 项基元全部改为显式时钟流水级，级数 ≤ 冻结计费 latency
（`qcore_pkg::vector_latency`）：

| 基元 | 级数 | 已计费 latency | 语义 |
|---|---|---|---|
| fp32_add / fp32_sub | 2 | VADD/VSUB = 2 | 逐位同 softfloat（截断加、denormal flush、Inf/NaN） |
| fp32_mul | 3 | VMUL/VSCALE = 3 | 逐位同 softfloat（24×24 尾数乘、截断） |
| fp32_max | 1 | VMAX = 2 | 符号幅值比较 |
| i32_to_f32（dequant） | 2 | DEQUANT = 5 | RNE |
| f32_to_i32（quant） | 3 | QUANT = 5 | RNE + 饱和 |
| fp32_to_bf16（写回） | 1 | — | RNE |
| bf16_to_fp32（输入） | 0 | — | 位重排 |

`synth_mac.sv`：mac_bf16 = fp32_mul3(3) + fp32_add2(2) = **5 级**；mac_int8 = 8×8 乘
+ 32b 累加 = **2 级**。吞吐 1 MAC/cycle（全流水）。K 维累加为**树**（log 深度），
故 MAC 流水 latency 不乘入 `matrix_pf_cycles`（其「+256」已计费 fill/drain）。

**功能等价（网表级回归，关键 op 向量）**：`asic/fp_equiv.sv`（Verilator）以 40 组
FP32/BF16/INT32 向量（含 normal/denormal/inf/nan/zero/±0/饱和）对拍流水化基元 vs
`softfloat_pkg` 逐位一致，**0 ULP**（`asic/run_fp_equiv.sh`）。

### 9.3 重综合 + STA（目标 Fmax(tt) ≥ 200 MHz，如实）

| corner | M9 直接映射 Fmax | P10b 流水化 Fmax(tt) |
|---|---|---|
| tt_025C_1v80 | 59.2 MHz | **≈ 129 MHz**（7.77 ns） |

流水化后 tt 关键路径（OpenSTA，1 ns 探针，Fmax = 1/arrival）：

| 模块 | delay | Fmax |
|---|---|---|
| fp32_add2 / fp32_sub2 | 7.77 ns | 128.7 MHz |
| fp32_mul3 | 7.65 ns | 130.7 MHz |
| mac_bf16（5 级，mul 级为限） | 8.51 ns | 117.5 MHz |
| f32_to_i32_3 | < 7.77 ns | — |

**200 MHz（5 ns）未达，如实记录**：物理级延迟受 130 nm 标准单元 + 冻结 latency 预算
双重约束——FP32 add（2 级，VADD=2）单级 7.77 ns，FP32 24×24 尾数乘（3 级，VMUL=3）
单级 7.65 ns，均 > 5 ns；进一步切分需 3/4 级，**超出冻结 latency 预算**。达 200 MHz
的路径是：向量引擎 BF16（8 位尾数，8×8 乘 ≈ 2 ns，可容 2/3 级）走 5 ns；FP32 尾数
乘只出现在 MAC 累加（矩阵引擎，`matrix_pf_cycles` 有充裕 headroom，可容 4–5 级）。
本报告代表性 FP32 数据通路口径如实 129 MHz；分级方案见 §9.5。

### 9.4 SRAM 缩编参数化（rtl/ 窄例外，默认参数不变）

`rtl/sram.sv`：`sram_top`/`sram_bank` 的 `BANK_BYTES`/`WORD_BYTES` 改实例参数，地址
宽度由 `$clog2` 派生；bank 选择位修正为冻结 D15 口径 `bank = word_addr[3:0]`（原
`[7:4]` 与 D15/03-memory §2.1 不一致，从未被 co-sim 路径实例化，属参数化过程中同步
修正，列为需评审项）。默认 512 KiB/bank → 8 MiB 不变。

缩编实例（16 bank × 缩编 bank 不变交错/端口结构）：

| BANK_BYTES | 总容量 | 占 4.75 MiB 片上预算 | 占 8 MiB 预算 |
|---|---|---|---|
| 512 KiB（默认） | 8 MiB | — | 100% |
| 256 KiB | 4 MiB | 84% | 50% |
| 128 KiB | 2 MiB | 42% | 25% |

`rtl/qcore_pkg.sv`：`SRAM_BYTES`/`SRAM_WORDS`/`BANK_WORDS` 本已由 `BANK_BYTES` 派生，
补注释说明缩编映射（co-sim `qmem` 保持 8 MiB 默认）。

**验证**：`asic/sram_check.sv`（Verilator 单测，`asic/run_sram_check.sh`）对默认 +
两个缩编实例各跑读写往返 / 字节使能 / 交错 bank 映射（写争用探针）/ 单 bank 双读，
**全过**。默认参数全回归（golden3 15 实例 + co-sim 4/4 + 单层 14/14）trace 逐周期
不变（qcore_top 用 `qmem` 平坦阵列，不实例化 `sram_top`，故 trivially 无 delta）。

### 9.5 分级方案（200 MHz 差距，供评审）

1. **BF16 向量数据通路**（5 ns 可达）：向量引擎 8 位尾数 8×8 乘 + 8 位加，2/3 级
   内每级 ≤ 5 ns。
2. **FP32 MAC 累加树**：24×24 乘拆 2 级 + 累加树（`matrix_pf_cycles` 有 headroom）。
3. **本报告口径**：以 FP32（24 位尾数）为代表性数据通路 → 129 MHz；若按 BF16 主
   通路重标定，可逼近 200 MHz。

### 9.6 delta 记录与需评审项

**delta（本节点 vs M9）**：

| 变更 | 文件 | 内容 | 影响 |
|---|---|---|---|
| 流水化 | `asic/synth_datapath.sv` | 8 基元全部时钟流水（级数 ≤ 计费 latency） | Fmax 59 → 129 MHz；fp_equiv 0 ULP |
| 流水化 | `asic/synth_mac.sv` | mac_bf16 5 级 / mac_int8 2 级 | MAC 单级 8.51 ns |
| 复现 | `asic/run_synth.sh`/`run_mac_synth.sh` | 自含源 + `fix_netlist.py` 接线 | 复现可跑 |
| STA | `asic/sta.tcl` | 时钟化设计（`clk` 端口） | — |
| SRAM 参数化 | `rtl/sram.sv`/`rtl/qcore_pkg.sv` | BANK_BYTES 实例参数 + `$clog2` 宽度 + D15 bank 位修正 | 默认不变；缩编单测全过 |
| 单测 | `asic/sram_check.sv`/`fp_equiv.sv`（+ `_main.cpp`/`run_*.sh`） | SRAM + 功能等价验证载体 | 全过 |

**需评审项**：
1. **bank 选择位修正**（`[7:4]`→`[3:0]`）：原 `sram.sv` 与冻结 D15 不一致（从未被
   co-sim 实例化），参数化过程中按 D15 同步修正——是否接受。
2. **200 MHz 未达口径**：FP32 代表性数据通路 129 MHz；达 200 MHz 需 BF16 主通路
   重标定（§9.5），是否作为 P10b 结论接受。
3. **FP32 mul 单级 7.65 ns**：24×24 尾数乘在 VMUL=3 预算内无法再切分；MAC 树口径
   （`matrix_pf_cycles` headroom）另计。


## 10. DC 综合流程（Design Compiler O-2018.06-SP1，P10 §10）

> 状态：DcSyn 交付；O6 全设计扩展。Synopsys Design Compiler（DC Ultra）
> O-2018.06-SP1 + sky130 5-corner liberty / SMIC28 HDC30P140 RVT tt/ss 库。代表数据通路
> （synth_datapath / mac_bf16）在双工艺下跑通 compile_ultra 并出 tt/ss 时序/漏电/面积；
> legacy sky130 结果与 OpenSTA 同口径交叉验证；
> **全设计 synth_top elaborate + link + compile_ultra 跑通**（存储/数字核黑盒化，
> 控制平面真综合，§10.5）；Track 2.2b 进一步综合矩阵状态/控制壳并链接 9 个真 SRAM，
> Track 2.3 闭合其双角输入/读回 scoped timing；Track 2.4a 已真综合 dual-MAC PE，
> 但 16×16 tile 与完整 128×128 算术核尚未接入全顶层，matrix/vector core 仍保留
> 物理宏边界（§10.8/§10.9）。

### 10.1 工具与 license（步骤 1）

| 项 | 实测 |
|---|---|
| 工具 | `dc_shell` O-2018.06-SP1（build Jul 19 2018），`$DC_HOME=/home/public/app/synopsys/syn/O-2018.06-SP1` |
| 调用链 | `dc_shell` → `dc2018` → `snps-centos7 $DC_HOME/bin/dc_shell` |
| license | `LM_LICENSE_FILE=27000@bics109`；`lmstat` 确认 `snpslmd: UP v11.14.1` |
| checkout | `get_license Design-Compiler` → 1（已持有）；`get_license DC-Ultra` → 1；feature 均 99 issued / 0 in use |

### 10.2 库准备（步骤 2）

sky130 liberty（`~/.eda/liberty/`，5 corner，452 cell，含 `internal_power` +
`cell_leakage_power` 表）**不能被 Synopsys LC 2018 直接读取**——skywater-pdk 转换器
产出的 bulk-well（`VNB` nwell / `VPB` pwell）元数据触发 fatal LBDB-27：

1. 6 个锁存单元（dlclkp/sdlclkp_{1,2,4}）内部节点 `M0` 的 `related_ground_pin` 指向
   `VNB`（nwell，非 primary_ground）；
2. 6 个 level-shifter tap 单元（lpflow_lsbuf_lh_{hl_,}isowell_tap_{1,2,4}）的
   `pg_pin(VPWR)` 带悬空 `related_bias_pin : "VNB"`（该单元无 VNB pg_pin）。

`asic/dc/clean_lib.py` 做语义中性清理（每 corner 精确 6+6 行：M0 ground 改指 VGND /
删悬空 related_bias_pin；不改任何 timing/power/function/leakage 值），
`lc_shell` 转 `.db`（tt/ss 各 ~4.2 MB）。**OpenSTA 继续读原始未改 liberty，基线不变。**

### 10.3 流程脚本与前端兼容（步骤 3）

`asic/dc/`：`run_dc.sh`（wrapper）、`dc_flow.tcl`（代表基元）、`dc_top.tcl`（全设计）、
`desugar_dc.py` / `hoist_dc.py`（DC 前端 desugar）、`clean_lib.py`、`sta_dc.tcl`、
`mem_stub.lib`（DC-local）、`db/`、`gen/`、`gen_full/`、`reports/`。

流程 = `elaborate → compile_ultra → report_timing/area/power`（1 ns 探针时钟、0 输入/
输出延时、Fmax = 1/arrival，与 sta.tcl 同口径）。DC Presto 前端对冻结 RTL 有 **5 类
不兼容**，前 3 类已 desugar（语义中性）；第 4 类（运行期 `for` + 推断 RAM）与第 5 类
（co-sim 端口宽度失配）在本节点（O6）一并处理，全设计就此 elaborate + link +
compile_ultra 跑通（见 §10.5）：

| # | 构造 | 位置 | DC 表现 | desugar |
|---|---|---|---|---|
| 1 | `k = -1;` 循环跳出（Yosys 无 `break` 的惯用法） | synth_datapath.sv×2、softfloat.sv×2 | **dc_shell 死循环**（6 行最小复现确认；`break` 瞬时返回） | `k = -1;` → `break;` |
| 2 | 变长边界 `for`（sticky-bit OR） | softfloat.sv×2 | ELAB-900「Loop exceeded maximum iteration limit」 | 定界 + 运行时 guard（同 preprocess.py） |
| 3 | 模块级「声明在使用之后」（SV declaration anywhere） | command_processor 等 8 模块 | VER-954/VER-956（先隐式声明后重定义） | 声明上提（hoist） |
| 4 | 运行期 `for (i < len)` 组合循环 + 推断 RAM | vector_engine/matrix_engine（co-sim 功能模型） | 资源/语义边界（~1.7 M 触发器 + 128-lane softfloat） | O6 先整核黑盒化（§10.5）；Track 2.2b 已用真 SRAM + 控制壳取代 matrix 整核黑盒（§10.8），vector/阵列核边界保留 |
| 5 | co-sim 端口宽度失配（vector_engine `len` 32b vs CP 16b） | vector_engine↔command_processor | LINK-3/LINK-25（linker 拒绝） | 黑盒端口收窄对齐 CP（§10.5） |

### 10.4 DC vs OpenSTA 交叉验证（步骤 4）

**先补跑 OpenSTA ss_100C_1v60 同口径基线**（sta.tcl + LIB 环境变量，1 ns 虚拟时钟、
Fmax=1/arrival，流水化 synth_datapath）——§9.3 只报了 tt，本节点补 ss：

| corner | OpenSTA 关键路径（data arrival） | Fmax = 1/arrival |
|---|---|---|
| tt_025C_1v80 | 7.7686 ns（u_add） | **128.7 MHz** |
| ss_100C_1v60 | 15.3851 ns（u_mul） | **65.0 MHz** |

legacy sky130 DC compile_ultra 结果（同口径 1 ns 探针时钟，`report_timing` data arrival）：

| design | corner | 关键路径 | Fmax | 面积 (µm² / mm²) | 漏电 |
|---|---|---|---|---|---|
| synth_datapath | tt | 3.02 ns | **331 MHz** | 61667.9 / 0.0617 | 29.83 nW |
| synth_datapath | ss | 5.90 ns | **169 MHz** | 62159.6 / 0.0622 | 38.73 µW |
| mac_bf16 | tt | 3.03 ns | **330 MHz** | 36029.6 / 0.0360 | 16.33 nW |
| mac_bf16 | ss | 5.64 ns | **177 MHz** | 38670.7 / 0.0387 | 23.76 µW |

**D18 SMIC28 数据通路重立（Track 2.2a）**：`dc_flow.tcl` 现与全顶层流程共用
`DC_TECH=sky130|smic28` 选择和 corner 映射；SMIC28 报告使用独立文件名，不覆盖 legacy
结果。四次 `compile_ultra` 均在 1 ns 约束下 `MET`：

| design | corner | 最差路径 | arrival / 1 ns 结果 | 面积 (µm² / mm²) | 漏电 |
|---|---|---|---|---|---|
| synth_datapath | tt | `u_add/sml0_reg[11]→u_sub/y_reg[8]` | 0.98 ns / **MET（≥1 GHz 探针）** | 3084.9 / 0.003085 | 22.28 µW |
| synth_datapath | ss | `i_in[0]→u_i2f/e0_reg[4]` | 0.98 ns / **MET（≥1 GHz 探针）** | 3236.7 / 0.003237 | 259.67 µW |
| mac_bf16 | tt | `u_mul/ma0_reg[17]→u_mul/prod_hi_reg[24]` | 0.99 ns / **MET（≥1 GHz 探针）** | 1905.6 / 0.001906 | 14.00 µW |
| mac_bf16 | ss | `u_add/sml0_reg[21]→u_add/y_reg[20]` | 0.98 ns / **MET（≥1 GHz 探针）** | 1961.4 / 0.001961 | 157.91 µW |

相对 legacy sky130 DC，同 RTL 的时序提升下界为 datapath tt/ss **≥3.08×/≥6.02×**、
mac_bf16 tt/ss **≥3.06×/≥5.76×**；面积分别缩小约 20.0×/19.2× 与 18.9×/19.7×。
这里不把 `1/0.98 ns` 写成极限 Fmax：compile_ultra 是按 1 ns 目标驱动优化，报告只证明
该约束在 ideal-clock、无提取互连寄生的综合口径闭合。ss 几乎无裕量，CTS/布线后是否仍
满足 1 GHz 必须由物理实现回答。

**逐项对比（legacy sky130 DC vs OpenSTA/Yosys）**：

| 量 | DC compile_ultra | Yosys abc + OpenSTA | 差异 |
|---|---|---|---|
| Fmax(tt) datapath | 331 MHz（3.02 ns） | 129 MHz（7.77 ns） | **2.57×** |
| Fmax(ss) datapath | 169 MHz（5.90 ns） | 65 MHz（15.39 ns） | **2.61×** |
| 面积(tt) datapath | 0.0617 mm² | 0.0609 mm²（§0） | +1.3%（同量级） |
| 面积(tt) mac_bf16 | 0.0360 mm² | ≈0.0349 mm²（§0） | +3%（同量级） |

**工具/effort 差异归因（如实）**：为区分「综合工具差异」与「STA 工具差异」，把 DC
网表交 OpenSTA 复算（`sta_dc.tcl`）：OpenSTA 对 DC 网表报 tt **3.0210 ns** / ss
**5.8959 ns**，与 DC 自身 report 一致（<0.1%）。**即 STA 引擎一致，2.6× 的 Fmax 差距
全部来自综合工具**——DC compile_ultra 用 DesignWare 算术（Wallace/Dadda 乘法器 +
时序驱动的 gate sizing/逻辑重构 + 等价寄存器合并），Yosys abc 仅做 techmap + 无时序
驱动。面积同量级（映射逻辑相同），仅时序结构更优。

**report_power 口径**：DC 只报**漏电 + 面积**（表内已列）；动态功耗沿用 §5 活动因子
估算（DC report_power 的动态值基于缺省翻转率，不采用；VCD 回注为后续）。
legacy sky130 漏电 ss(100 °C) 较 tt(25 °C) 高 ~1300×，与其 liberty 一致（NAND2
cell_leakage 0.0021 nW → 2.27 nW）；SMIC28 datapath/mac 的 ss/tt 漏电约 11.7×/11.3×，
跨工艺绝对漏电不可直接按面积缩放。

SMIC28 数据通路报告复现命令：

```bash
DC_TECH=smic28 bash asic/dc/run_dc.sh tt_025C_1v80 synth_datapath
DC_TECH=smic28 bash asic/dc/run_dc.sh ss_100C_1v60 synth_datapath
DC_TECH=smic28 bash asic/dc/run_dc.sh tt_025C_1v80 mac_bf16
DC_TECH=smic28 bash asic/dc/run_dc.sh ss_100C_1v60 mac_bf16
```

### 10.5 全设计 synth_top（步骤 5，全设计 elaborate + compile_ultra 跑通）

> 本节记录 O6 首次打通全顶层时的历史基线。Track 2.2b 已取代其中 matrix 的整模块
> 黑盒方案；当前矩阵状态/控制与 9 个 SRAM 的口径见 §10.8。

`synth_top` = command_processor@MAX_VEC=128 + 8 MiB scratchpad SRAM 黑盒。SRAM 按
**空模块黑盒 + set_dont_touch**（`sram_macro.sv` 空模块；`asic/dc/mem_stub.lib` 的
时序模型因 LC 2018 不支持标量 pin 上的 `bus_type`（LBDB-76）无法表达 23/8 位总线，
故 DC 侧不链时序库、以空模块黑盒为准；OpenSTA 继续用原 `asic/mem_stub.lib` 不变）。

**本节点更进一步（O6）**：§10.3 第 4 类不兼容（运行期 `for` + 推断 RAM）此前「不
desugar」，卡死在 matrix_engine 展开。根因是 co-sim 功能模型的**存储与数字核被 DC
展开成触发器 / 128-lane softfloat**（matrix_engine 4 组推断 RAM 共 53248×32 ≈ 1.7 M
触发器 + vector_engine 128-lane 组合展开）。`hoist_dc.py` 新增**第 4 类 desugar
（语义中性，只写 gen_full/，rtl/ 不动）**——把物理上本就是宏的存储与数字核映射为黑盒：

| # | 构造 | 位置 | 原 DC 表现 | desugar |
|---|---|---|---|---|
| 4a | 阵列 acc/partial/cin/scale 推断 RAM + 1-MAC/cycle 核心 | matrix_engine | 16384×3+4096 项 RAM → ~1.7 M 触发器，elaborate 卡死 | 整模块 `(* blackbox *)`（systolic array 宏；MAC 基元 mac_bf16/mac_int8 见 §10.4） |
| 4b | 128-lane × softfloat 运行期 `for` 组合核心 | vector_engine | 128-lane 组合展开（含 fp32_div/exp 迭代单元） | 整模块 `(* blackbox *)`（128-lane datapath 宏；FP 基元 synth_datapath 见 §10.4） |
| 4c | 指令流 `imem [0:4095]` 推断 RAM | command_processor | 4096×128 → ~512 K 触发器 | `bb_sram.sv`（1 同步写 + 1 组合读 SRAM 宏） |

另修复 **1 处 co-sim 端口宽度失配（新增第 5 类，如实记录）**：vector_engine 端口
`len` 声明 32 bit、CP 驱动 16 bit——Verilator 静默加宽、DC linker 报 LINK-3/LINK-25
（上一版 elaborate 从未走到 vector_engine 链接故未暴露）。黑盒端口 `len` 收窄为
16 bit 对齐 CP（co-sim rtl/ 不动，黑盒为 DC-only 副本）。
| design | corner | 关键路径 | Fmax | 面积 (µm² / mm²) | 漏电 | 单元 | 黑盒 |
|---|---|---|---|---|---|---|---|
| synth_top | tt_025C_1v80 | 7.33 ns（len_reg[2]→hbm_addr[33]） | **136 MHz** | 1000498 / **1.000 mm²** | 455.3 nW | 98548（21036 seq + 77512 comb） | sram_macro + matrix_engine + vector_engine + bb_sram（0 面积，不映射） |
| synth_top | ss_100C_1v60 | 13.91 ns（len_reg[1]→hbm_addr[34]） | **72 MHz** | 1031744 / **1.032 mm²** | 588.4 µW | 107021（21036 seq + 85985 comb） | 同上 |

面积 **1.000 mm² 为控制平面**（CP FSM/译码/取指/字节级 marshalling + AR/C/va/vb/
a_slice/b_slice 寄存器文件 + dma_engine + kv_addrgen），引擎与存储为黑盒（面积按 §4
SRAM 估算 + §10.4 基元口径另计）。关键路径 7.33 ns 是控制平面**组合地址译码深度**
（未流水化的 co-sim 模型），Fmax ≈ 136 MHz 如实——与 §10.4 流水化基元 331 MHz 分属
不同口径（控制平面 vs 数据通路基元）。

**结论（如实）**：全设计 synth_top 在 DC 侧 **elaborate + link + compile_ultra 全部
跑通**（前 3 类 desugar + 第 4 类存储/数字核黑盒化 + 第 5 类端口宽度修正），产出控制
平面时序/面积/漏电报告；矩阵/向量数字核与 SRAM 按物理宏边界黑盒（基元已在 §10.4
真综合），**比原现状（卡 matrix_engine、无全设计报告）更进一步**。

### 10.6 交付物与需评审项
`asic/dc/` 交付：`run_dc.sh` / `dc_flow.tcl` / `dc_top.tcl` / `desugar_dc.py` /
`hoist_dc.py`（第 4 类现为 matrix 宏壳替换 + vector 核黑盒，第 5 类端口宽度修正）/
`clean_lib.py` /
`sta_dc.tcl` / `mem_stub.lib`（DC-local）/ `bb_sram.sv`（指令流 SRAM 黑盒）+ `db/`
（tt/ss `.db` + mem_stub.db）+ `gen/` / `gen_full/`（desugar 产物；matrix 为真 SRAM
状态壳，matrix_compute_core/vector_engine 为黑盒）+ `reports/`（代表数据通路基元报告/
网表、dual-MAC PE 的 1.0/0.9 ns TT/SS 四份报告，以及全设计 synth_top 双角报告/网表；
门级 `.v` 为本地可再生成产物，不进入公开仓库）。

**需评审项**：
1. **Fmax 口径重标定**：DC compile_ultra 对同一 liberty 的 Fmax(tt)=331 MHz，显著高于
   Yosys/OpenSTA 的 129 MHz（§9.3），差因为综合工具（§10.4 已证 STA 一致）；是否以
   DC 结果作为物理数据通路新口径（与 §9.5 的 BF16 重标定并列）。
2. **全设计口径（O6 历史项）**：首次 synth_top 跑通时矩阵/向量数字核与 SRAM 均按宏
   边界黑盒，控制平面 1.000 mm² / Fmax 136 MHz；matrix 整核黑盒已由 Track 2.2b 的
   9 个真 SRAM + 控制壳取代（§10.8），不能再把该历史面积当当前全顶层面积。
3. **黑盒接口口径**：vector_engine 黑盒端口 `len` 由 32 bit 收窄为 16 bit 对齐 CP
   （co-sim 静默加宽、DC linker 严格），属黑盒副本（gen_full/）语义中性修正，是否接受。
4. **license/库兼容**：sky130 liberty 需 clean_lib.py 清理才可被 LC 读取、mem_stub.lib
   时序模型需 constraint 模板修正且 DC 侧改用空模块黑盒——两处库修正是否接受。

### 10.7 SMIC28 全流程重立（D18）+ 开源合规处置

> 状态：RebaselineAgent 交付（round-2 对 fmax-pipeline / smic28-rebaseline 两计划评审
> 一致后执行）。ASIC 工艺基线由 sky130 切换 **SMIC28（28HKCP，0.9 V）**，消除
> 「sky130 逻辑 + SMIC28 宏」的跨工艺混搭；商业 PDK 产物（宏 .lib/.lef 等）从公开仓库
> 与 git 历史中移除。

#### 10.7.1 双工艺参数化（dc_top.tcl + setup_smic28.sh）

`dc_top.tcl` 增加 `DC_TECH=sky130|smic28`：smic28 逻辑库 = 28HKCP HDC30P140 RVT 基本
库（corner 映射 `tt_025C_1v80→tt_v0p9_25c` / `ss_100C_1v60→ssg_v0p81_125c`，与宏
`tt_ctypical_0p90v_0p90v_25c` / `ssg_cworstt_0p81v_0p81v_125c` 同电压/温度/进程角）；
sky130 保留旧口径。宏 .db（kh4096x64/kn128x16，本就是 SMIC28 宏）两工艺下相同；
报告 tag 加 `smic28_` 前缀（`synth_top_smic28_${corner}.rpt`）避免覆盖 sky130 旧报告。

新增 `asic/smc28/setup_smic28.sh`：宏按 `asic/sram_macros/*/GEN.md` 从编译器包
（`SRAM_Ccompiler_ARM20240823`，经 skill `smic28-sram-compiler` 建 no-space + glibc 2.17
兼容 shim）再生成至稳定目录 `SMIC28_MACRO_DIR`（默认 `/home/public/PDK/SMIC28/macros_out`，
非临时 /tmp）；std cell .db 从库树预解压目录 `SMIC28_STD_DIR`（默认
`.../SCC28NHKCP_HDC30P140_RVT_V0p2`，0.9v basic 库）staged 至 `asic/dc/db/`；
`build_macro_db.sh` 改读 `SMIC28_MACRO_DIR`。已验证：vendor 基本 .db 可直接被
DC/LC O-2018.06-SP1 读取（`read_db` OK，无需 .lib 重转，兜底路径保留）。

#### 10.7.2 开源合规（含历史处置，P1 裁决）

| 步骤 | 结果 |
|---|---|
| 备份 | `git bundle create openlpu-backup-pre-filter.bundle --all`（3 refs，完整历史，存于本地 PDK 目录） |
| 移出 | 宏 `.lib`×6 + `.lef`×3 `git rm`；`.v`/`GEN.md`/`PORTS.md` 保留（RTL 仿真模型 + 生成记录，非商业 IP） |
| 忽略 | `.gitignore` 覆盖 `asic/sram_macros/**/*.{lib,lef,gds2,cdl,clf}` + `.dwsvf-*` |
| 重写 | `git filter-repo --invert-paths --path-glob 'asic/sram_macros/*/*.lib' --path-glob 'asic/sram_macros/*/*.lef'`（重写 3 commit，d8fb867 起） |
| 推送 | `git push --force origin main`（旧 d8fb867 → 新 674e40d，成功） |
| 验证 | 全历史扫描无 `asic/sram_macros/**/*.{lib,lef}`（仅项目自写 `mem_stub.lib` 保留，非 PDK 产物） |

#### 10.7.3 双基线 Fmax（如实，1 ns 探针时钟 / Fmax = 1/arrival）

| 口径 | corner | 关键路径 | data arrival | Fmax |
|---|---|---|---|---|
| legacy（**跨工艺可达性口径：sky130 逻辑 + SMIC28 宏**） | tt | `rope_sincos` 锥（`pos_base_reg[2]→row_reg[40][19]`） | 68.11 ns | 14.7 MHz |
| SMIC28 pre-pipeline（未流水化对照） | tt | `u_cp/u_kvqd/s_bits_reg[2]→hbm_wdata[2]`（kv_quantdequant；rope 锥 8.56 ns 紧随） | **8.59 ns** | **116.4 MHz** |
| SMIC28 pre-pipeline（未流水化对照） | ss | `u_cp/u_kvqd/s_bits_reg[5]→hbm_wdata[2]`（rope 锥 11.43 ns 紧随） | **11.49 ns** | **87.0 MHz** |
| **SMIC28 rope post-pipeline（锥流水化 14 级）** | tt | `u_cp/u_kvqd/s_bits_reg[5]→hbm_wdata[0]`（kv_quantdequant 量化器） | **4.59 ns** | **217.9 MHz** |
| **SMIC28 rope post-pipeline（锥流水化 14 级）** | ss | `u_cp/u_kvqd/s_bits_reg[1]→hbm_wdata[0]` | **6.08 ns** | **164.5 MHz** |
| **SMIC28 quant-pipeline（当前 RTL，Track 2.1）** | tt | `u_cp/len_reg[3]→hbm_addr[33]`（B-feed reg→reg 1.55 ns 紧随；top-10 无 kv_quantdequant） | **1.57 ns** | **636.9 MHz** |
| **SMIC28 quant-pipeline（当前 RTL，Track 2.1）** | ss | `u_cp/u_dma/row_reg[2]→hbm_addr[39]`（另 2 条 HBM 地址路径同为 2.12 ns；7 条 B-feed reg→reg 为 2.09 ns；top-10 无 kv_quantdequant） | **2.12 ns** | **471.7 MHz** |

**结论（如实）**：pre-pipeline 关键路径 = kv_quantdequant 量化器（8.59 ns tt / 11.49 ns
ss），`rope_sincos` 锥（28 nm 口径 8.56 / 11.43 ns，≈ sky130 68.11 ns 的 1/8）几乎并列。
14 级锥流水化先把 rope 锥切出关键路径，将控制平面推进到 4.59 / 6.08 ns（Fmax **tt
217.9 / ss 164.5 MHz**）。Track 2.1 随后将量化器的四轮 Newton-Raphson 倒数按
`mul/sub/mul` 逐操作寄存，并将 element multiply 与 RNE/clip 分级；scale 写回也先寄存。
INT8-K/INT4-V 仍保持一元素/周期的稳态吞吐与原有 softfloat 运算顺序。

当前 RTL 的完整 tt/ss `compile_ultra` 报告分别为
`asic/dc/reports/synth_top_smic28_tt_025C_1v80_kvqd_pipe.rpt` 与
`asic/dc/reports/synth_top_smic28_ss_100C_1v60_kvqd_pipe_ss.rpt`。tt 最差 data arrival
**1.57 ns**（**636.9 MHz**），相对上一 tt 基线 4.59 ns / 217.9 MHz 提升 **2.92×**；
ss 最差 data arrival **2.12 ns**（**471.7 MHz**），相对上一 ss 基线 6.08 ns / 164.5 MHz
提升 **2.87×**。tt 达到 Track 2.1 的 ≥300 MHz 目标，ss 探针值也高于 300 MHz。tt top-10 为 6 条
`len_reg→hbm_addr`（1.57 ns）与 4 条 B-feed reg→reg（1.55 ns）；ss top-10 为 3 条
DMA/长度寄存器到 `hbm_addr`（2.12 ns）与 7 条 B-feed reg→reg（2.09 ns）。量化器在双角
均完全退出 top-10。

tt 总 cell area 271824.6→266379.2 μm²（0.272→0.266 mm²，**-2.00%**）；ss 总 cell
area 272559.7→268726.7 μm²（0.273→0.269 mm²，**-1.41%**）。低努力功耗估计：tt dynamic
108.181 mW、leakage 1.540 mW；ss dynamic 91.339 mW、leakage 16.380 mW。ss 相对上一同角
基线的 dynamic/leakage 仅 +0.72%/+0.12%。

以上仍是 1 ns、ideal-clock 的综合探针，不是布局后 signoff：报告对 B-feed `clk` 的 30804
loads 使用 high-fanout=1000 延迟估算，且无 CTS/提取互连寄生。下一步须补 CTS/布局后 STA。
代表数据通路基元已经完成 SMIC28 重立并闭合 1 GHz 综合探针（§10.4）；Track 2.2b 又完成
matrix 状态 RAM 的真宏例化与接口报告（§10.8）。剩余边界是 128×128 matrix 算术核、
vector 数字核，以及 CTS/提取互连寄生后的整芯片 signoff。

> **pre-pipeline 收敛耗时（如实）**：未流水化的 `rope_sincos` 组合锥（28–31 op 级）使
> compile_ultra 收敛极慢——首轮 tt/ss 在 ~7 h 被外部终止、重跑 tt/ss 各 ~7.8 h 才完成
> Delay Optimization 并产出最终 report_timing（上表即最终值，非观测值）。rope 锥 68.11 ns
> sky130 → 流水化后非瓶颈的降路径结论不受收敛耗时影响。

### 10.8 矩阵状态 SRAM 真宏例化与输入切片（Track 2.2b/2.3）

Track 2.2a 已证明代表 BF16/INT8 基元在 SMIC28 1 ns 综合探针下可闭合，但 §10.5 的
`matrix_engine` 仍是整模块黑盒，既看不到内部状态 RAM 面积，也无法检查 SRAM 接口。
本节点保持功能 co-sim 的 `rtl/matrix_engine.sv` 不动，只在 DC 派生树中用
`asic/matrix_engine_sram.sv` 替换它；`matrix_compute_core` 作为 128×128 算术阵列边界，
继续使用显式黑盒。由此，报告覆盖矩阵状态/控制与真 SRAM 接口，但不虚构阵列内部面积
或时序。

#### 10.8.1 容量与 bank 映射

全部宏均为 SM18CA001 单口 `kh4096x64`，同步写、1-cycle 注册读，低有效 `CEN/WEN`，
`EMA=3'b011`、`EMAW=2'b01`、`EMAS=0`、`RET1N=1`。输出元素按
`bank=index[1:0]`、`word=index[13:2]` 交错：

| 状态 | 真宏 | 逻辑容量 | 物理容量 | 说明 |
|---|---:|---:|---:|---|
| `{partial, acc}` | 4 × 4096×64 | 128 KiB | 128 KiB | 两个 32-bit 状态打包，无浪费 |
| C seed | 4 × 4096×64 | 64 KiB | 128 KiB | 每字仅用低 32 bit |
| dequant scale | 1 × 4096×64 | 16 KiB | 32 KiB | 每字仅用低 32 bit |
| **合计** | **9 个宏** | **208 KiB** | **288 KiB** | 位利用率 **72.22%** |

frontend `elaborate + link` 精确识别 9 个矩阵宏；其宏面积为 **707049.9141 µm²
(0.707050 mm²)**。全顶层共保护 15 个 SRAM/黑盒实例，矩阵外还包括 B-feed 宏、指令/
scratchpad SRAM 及 matrix/vector 数字核边界。

#### 10.8.2 单口调度与读回协议

每个 accumulator bank 带一个 1-entry writeback queue。新读请求优先；若固定延迟算术核
的返回结果与同 bank 新读冲突，结果携带 `valid/bank/word/final` tag 入队，并在该 bank
下一空闲周期写回。`outstanding` 计数在全部队列真正落宏前禁止 C 读回，`done` 也只在
最终结果完成物理写入时拉高。契约要求 `N % 4 == 0` 且算术核延迟短于同一元素的 `M*N`
复用距离；冻结模式 `N=128`，行为压力核为 4 cycle，仿真最小复用距离为 8。

单口同步读还暴露了 CP 过渡风险：`S_MX_WAIT` 看到 `done` 时才把 `rd_ptr` 清零，若宏在
同一边沿仍采用旧地址，第一个输出会陈旧。壳层用 `c_prefetch_zero` 强制 WAIT→RDOUT
过渡周期读取元素 0；定向测试故意把等待态地址留在 127，验证首元素仍正确。

#### 10.8.3 验证与边界

`asic/run_matrix_sram_check.sh` 用同端口行为宏和 4-cycle tagged compute core 覆盖
INT8 accumulate、BF16+C seed、INT8 dequant(K=128)、INT4 dequant(K=128)，四例全部
bit-exact PASS。4-cycle 延迟特意制造 `latency % 4 == 0` 的读/返回同 bank 冲突，覆盖
writeback queue；陈旧 `c_raddr=127` 覆盖首元素预取。DC frontend 另为 9 个宏生成
`to SRAM inputs` 与 `through SRAM Q` 两组 scoped timing，避免只用实例计数代替接口证据。

#### 10.8.4 SMIC28 双角全顶层综合结果（1 ns ideal-clock probe）

两角均以 `DC_TECH=smic28`、`DC_TOP_COMPILE=1` 对当前派生顶层执行
`compile_ultra`；Fmax 按本报告统一口径 `1 / data arrival` 计算。全局最差路径和矩阵
宏接口的 scoped 结果如下。`total cell` 包含逻辑与所有受保护宏/黑盒；其中矩阵九宏的
层次局部面积仍为 707049.9141 µm²，顶层 macro/black-box 面积还包含其它 SRAM/黑盒。

| corner | 全局最差路径（data arrival / slack） | Fmax probe | total cell area | 逻辑/非宏面积 | macro/black-box area | dynamic / leakage | 矩阵 SRAM 输入 scoped | 矩阵 Q 读 scoped |
|---|---|---:|---:|---:|---:|---|---|---|
| `tt_025C_1v80` | `u_cp/C_reg[31][4] → hbm_addr[39]`，1.47 ns / **-0.47 ns** | **680.3 MHz** | 978016.837384 µm² | 190890.085675 µm² | 787126.751709 µm² | 121.2793 / 2.1169 mW | 0.80 ns / **+0.01 ns MET** | 0.48 ns / **+0.50 ns MET** |
| `ss_100C_1v60` | `u_cp/u_bfeed/s_ang2_reg[2][24] → s_r1_reg[2][16]`，1.93 ns / **-0.95 ns** | **518.1 MHz** | 980001.631439 µm² | 192874.879730 µm² | 787126.751709 µm² | 101.7342 / 16.7783 mW | 0.97 ns / **-0.18 ns VIOLATED** | 0.63 ns / **+0.34 ns MET** |

TT 的全局最差 slack 与 HBM 输出路径绑定，B-feed 路径紧随；SS 的全局最差路径在
B-feed。矩阵宏读出 Q 在两角均满足 1 ns 探针，但 SS 从控制逻辑到 SRAM 地址/写控制的
最差输入路径仍有 0.18 ns setup 违例，需要在下一阶段通过驱动、约束或物理实现处理。
因此 Track 2.2b 的结论是“真实状态 SRAM 已接入且读回接口可观测”，不是全顶层时序收敛。
上述数据没有 CTS、布局、提取互连寄生，也没有 128×128 算术核 Liberty；不能替代 signoff。

**边界声明**：`matrix_compute_core` 没有 Liberty，因此本节的全顶层结果只证明状态/控制壳
和真实 SRAM 接口，不代表 128×128 阵列内部时序/面积，更不能宣称整芯片 1 GHz signoff。
阵列代表 BF16/INT8 基元的独立 1 ns 探针见 §10.4；下一物理阶段仍需计算核 Liberty、
floorplan/CTS/布线与提取寄生 STA。

#### 10.8.5 Track 2.3 本地请求/预载寄存切片

Track 2.2b 的 SS 输入违例来自 CP 组合计数/地址逻辑直接驱动 SRAM 输入。物理壳
`asic/matrix_engine_sram.sv` 现加入本地 MAC 请求寄存切片，地址、bank、scale 地址与
计算 metadata 在同一边界捕获；同步宏 Q 返回后，再把对应 metadata 送入 compute pipe。
切片填满后仍保持每周期一个请求。首次切片综合消除 MAC 地址长路径后，C seed 预载数据
输入成为新的 SS 临界路径，因此 C seed 与 scale 写请求也在宏前增加一拍寄存。

功能协议保持不变：START 边沿仍消费上一拍捕获的最后一个预载写脉冲，随后清除脉冲，
不会泄漏进 MAC 阶段。四个定向用例继续 bit-exact PASS，并覆盖 4-cycle 同 bank 冲突、
writeback queue 与首元素预取；`pytest qsim/` 为 52 passed，B' co-sim 3/3 PASS、0 ULP，
`run_all_acceptance.sh --quick` 为 12 PASS / 0 FAIL / 2 SKIP。

正式报告由当前源重新运行 `python3 asic/dc/hoist_dc.py` 后再执行双角
`compile_ultra` 生成；也可使用会自动执行该步骤的 `asic/dc/run_dc.sh`。最终报告为：

- `asic/dc/reports/synth_top_smic28_tt_025C_1v80_mxram_slice_wreg.rpt`
- `asic/dc/reports/synth_top_smic28_ss_100C_1v60_mxram_slice_wreg.rpt`

| corner | 全局最差路径（data arrival / slack） | Fmax probe | total cell area | 逻辑/非宏面积 | macro/black-box area | dynamic / leakage | 矩阵 SRAM 输入 scoped | 矩阵 Q 读 scoped |
|---|---|---:|---:|---:|---:|---|---|---|
| `tt_025C_1v80` | B-feed `s_n2phi_reg[1][24] → s_r1_reg[1][16]`，1.45 ns / **-0.46 ns** | **689.7 MHz** | 978056.723348 µm² | 190929.971639 µm² | 787126.751709 µm² | 121.4191 / 2.1162 mW | 0.25 ns / **+0.57 ns MET** | 0.48 ns / **+0.50 ns MET** |
| `ss_100C_1v60` | B-feed `s_ang2_reg[3][25] → s_r1_reg[3][22]`，1.93 ns / **-0.95 ns** | **518.1 MHz** | 979851.887415 µm² | 192725.135706 µm² | 787126.751709 µm² | 102.0429 / 16.7331 mW | 0.41 ns / **+0.40 ns MET** | 0.63 ns / **+0.34 ns MET** |

相对 Track 2.2b，SS 矩阵 SRAM 输入 slack 从 **-0.18 ns** 提升到 **+0.40 ns**，
TT 输入裕量也从 +0.01 ns 提升到 +0.57 ns；两角 top-10 scoped 输入与 Q 路径全部
`MET`。这只闭合了当前真 SRAM 接口的 pre-layout 局部目标。全局 B-feed 仍违例，且报告
仍不含 `matrix_compute_core` Liberty、CTS、布局布线与提取寄生，不能据此宣称全芯片
1 GHz signoff。

### 10.9 Matrix dual-MAC PE 物理基元（Track 2.4a）

冻结规格要求 128×128 weight-stationary 阵列，每 PE 保存两份 INT8 权重、每拍接收两路
激活与北向 INT32 partial sum，并把激活向东、partial sum 向南各推进一个 PE。新增
`asic/matrix_int8_pe.sv` 实现同拍两次 signed `INT8×INT8` 加法与模 2^32 累加；输出边界
全部寄存，连续 valid 的吞吐为 2 MAC/PE/cycle。未增加隐藏流水，因此仍可组成规格中的
128 列水平传播 + 128 行垂直传播 = 256-cycle fill/drain。

`matrix_int8_pe_probe` 在 west/north 输入前加入 49 个真实 launch flop，避免顶层零 input
delay 掩盖相邻 PE 的 clock-to-Q；综合时保留 `u_pe` 层次，报告可分离 PE 与 probe 开销。
测试证据如下：

- 核心 Verilator：边界、bubble、权重重载、背靠背 valid、随机 INT32 psum，并让每路
  activation×weight 都覆盖完整 256×256 域，**65,544 checks / 0 failure**；
- registered probe：**964 checks / 0 failure**；
- TT 映射网表 + 官方 HDC30P140 functional standard-cell model：Icarus 独立 scoreboard
  **1,928 checks / 0 failure**。官方模型含 Verilog-1995 UDP，故门级执行使用 Icarus，
  RTL/probe 仍由 Verilator 验证。

SMIC28 流程兼容标签 `tt_025C_1v80`/`ss_100C_1v60` 在本节实际映射到
`tt_v0p9_25c` 与 `ssg_v0p81_125c`。`compile_ultra` 结果为：

| corner | 约束 | 最差路径 | arrival / slack | cells | PE 本体面积 | probe 总面积 | probe dynamic / leakage |
|---|---:|---|---:|---:|---:|---:|---:|
| TT | 1.0 ns | `weight0_reg[0] → psum_south_reg[31]` | 0.98 / **+0.00 ns MET** | 969 | 560.462 µm² | 680.610 µm² | 503.734 / 5.095 µW |
| SS | 1.0 ns | `launch_act0_reg[2] → psum_south_reg[17]` | 0.97 / **+0.00 ns MET** | 1232 | 631.414 µm² | 751.562 µm² | 433.185 / 62.898 µW |
| TT | 0.9 ns | `weight0_reg[0] → psum_south_reg[17]` | 0.87 / **+0.01 ns MET** | 1025 | 580.258 µm² | 700.406 µm² | 565.470 / 5.174 µW |
| SS | 0.9 ns | `launch_act1_reg[1] → psum_south_reg[18]` | 0.87 / **+0.00 ns MET** | 1307 | 691.194 µm² | 811.930 µm² | 500.769 / 69.795 µW |

1.0 ns 基线仅在报告显示精度上达到 `MET`，没有布局后余量。0.9 ns 加强映射相对基线
分别增加 TT **3.53%**、SS **9.47%** 的 PE cell area，证明前布局阶段可换取约 10% 周期
预算，但不等同于 CTS/布线后 1 GHz 收敛。功耗没有 VCD 回标，是 probe 级低努力估计，
不外推整阵列功耗。

按 1.0 ns PE 本体面积直接复制 16,384 份，TT/SS 为 **9.183/10.345 mm²**。这是只含
标准单元的理论复制下限，未包含时钟树、PE 间互连、拥塞、权重装载网络、9 个状态 SRAM、
32 KiB 双缓冲权重存储或后处理；不能冒充 tile 或整阵列布局面积。下一阶段必须先实现
16×16 结构 tile，验证 wavefront skew/valid/权重寻址并做分层综合，再组成 8×8 tile grid。

复现命令（商业库与官方门级模型只在本机使用，不提交仓库）：

```bash
bash asic/run_matrix_int8_pe_check.sh
DC_TECH=smic28 DC_LABEL=pe1c bash asic/dc/run_dc.sh tt_025C_1v80 matrix_int8_pe
DC_TECH=smic28 DC_LABEL=pe1c bash asic/dc/run_dc.sh ss_100C_1v60 matrix_int8_pe
DC_TECH=smic28 DC_PERIOD=0.9 DC_LABEL=pe1c_p090 \
  bash asic/dc/run_dc.sh tt_025C_1v80 matrix_int8_pe
DC_TECH=smic28 DC_PERIOD=0.9 DC_LABEL=pe1c_p090 \
  bash asic/dc/run_dc.sh ss_100C_1v60 matrix_int8_pe
bash asic/run_matrix_int8_pe_gate_check.sh
```

## 11. VCS 功能仿真（VCS-MX O-2018.09-SP2，P10 §11）

> 状态：VcsRun 交付。Synopsys VCS-MX O-2018.09-SP2 跑通 qcore_top 既有 co-sim
> 指令级用例（vector 22 + KV 2 + 单 tile GEMM/GEMV 矩阵 3），与 Verilator 结果
> **字节级一致**（trace 逐记录 + total_cycles + 最终内存逐字节），0 ULP（唯一例外为
> 冻结 ROPE pos=40960 Cody-Waite 边界残差，非门控）。

### 11.1 工具与 license（步骤 1）

| 项 | 实测 |
|---|---|
| 工具 | `vcs -ID` = **VCS-MX O-2018.09-SP2_Full64**（Build Feb 28 2019），`$VCS_HOME=/home/public/app/synopsys/vcs-mx/O-2018.09-SP2` |
| license | `LM_LICENSE_FILE=27000@bics109`；`lmstat` 确认 `snpslmd: UP v11.14.1`，vlog/vcsace/xsim 等 feature 99 issued / 0 in use |
| 编译冒烟 | simv 编译 **0 error**（6 模块 elaborate + link），见 §11.3 |

### 11.2 运行环境（CentOS 7 兼容层，如实）

VCS O-2018.09 预编译对象引用遗留 glibc 符号 `__pthread_unwind@GLIBC_PRIVATE`（glibc
≥ 2.34 已删除；本机 Ubuntu 22.04 / glibc 2.35）。因此**编译与 simv 运行都必须在
Synopsys CentOS 7 兼容命名空间内**（`snps-centos7`，glibc 2.17；与 §10.1 DC 同链）：

```bash
# 编译
snps-centos7 $VCS_HOME/bin/vcs -full64 -sverilog +v2k -timescale=1ns/1ps \
  +vcs+initreg+random +incdir+gen -top tb_qcore_top -o simv gen/qcore_top.sv tb_qcore_top.sv
# 运行
snps-centos7 ./simv +prog=… +preload=… +dump_req=… +trace=… +total=… +dump=… \
  +max_cycles=… +vcs+initreg+0 +vcs+initmem+0
```

### 11.3 SV 前端兼容（gen_rtl.py，语义中性）

VCS O-2018.09 前端对冻结 rtl/ 有 **2 类不兼容**（Verilator 均接受）；`asic/vcs/
gen_rtl.py` 快照 rtl/ → `asic/vcs/gen/` 并做语义中性改写（零功能改动）：

| # | 构造 | 位置 | VCS 表现 | 改写 |
|---|---|---|---|---|
| 1 | 模块级「声明在使用之后」（SV declaration anywhere） | command_processor.sv（kv_base / dma_* / op_* 三组） | VER-954/956「Identifier not declared」+「second declaration ignored」（隐式 wire 会吃掉后续 `logic` 宽度） | 声明上提（hoist） |
| 2 | 同一变量被两个 `always_ff` 驱动 | qmem.sv（engine 写 + backdoor 写 sram/hbm） | ICPD「Illegal combination of procedural drivers」 | 合并为单进程（co-sim 中 bd_en 仅 preload/dump、wr_en 仅执行，互斥，位级等价） |

**4 态 vs 2 态上电**：Verilator 2 态上电全 0，VCS 4 态上电为 X；co-sim preload 只发
非零 SRAM 连续段。编译加 `+vcs+initreg+random`、运行加 `+vcs+initreg+0 +vcs+initmem+0`
复现 Verilator 全 0 上电（计划口径：X 传播分歧即查，非放宽）。

### 11.4 测试台重写（步骤 2）

`asic/vcs/tb_qcore_top.sv` 重写 `rtl/tb/sim_main.cpp`（Verilator 专用 API；RTL 无 DPI）。
按二进制文件格式逐字节复刻（prog.bin 128 位 LE 指令 / preload.bin / dump_req.bin 记录流
/ trace.bin 6 字节记录 / total.bin 8 字节 LE / dump.bin），经 `$fopen("rb"/"wb")`+
`$fgetc`+`$fwrite("%c")` 读写，文件路径由 `+plusarg` 传入。驱动 `asic/vcs/
run_vcs_tests.py` 复用 `rtl/tb/run_cosim.py` 的用例构造（`run_rtl` monkeypatch 指向
VCS），对每用例分别跑 Verilator 与 VCS 并逐字节 diff。

### 11.5 用例与交叉验证（步骤 3，验收）

**判据 = trace 逐记录一致 + 最终内存逐字节一致**（与 cosim.py 精确比对口径一致）。
27 用例全部 **byte_exact（VCS == Verilator：trace + total_cycles + dump 逐字节）**：

| 组 | 用例 | 数 | 结果 |
|---|---|---|---|
| vector 指令级 | VADD/VSUB/VMUL/VDIV/VMAX、VADD/VMUL_bcast、VRECIP/VRSQRT/VSILU/VMOV、VEXP、VSCALE、VMASK、VREDUCE_SUM/MAX、ROPE(pos 42/1024/8192/40960)、RMSNORM、QUANT+DEQUANT | 22 | 全 byte_exact，0 ULP |
| KV 指令级 | KV.APPEND+LOAD、KV.STORE_BLOCK+GATHER | 2 | 全 byte_exact，0 ULP |
| matrix 单 tile | GEMM_1tile_BF16(M=128,N=128,K=128)、GEMM_1tile_INT8(M=8)、GEMV_1tile_BF16(DC,M=1) | 3 | 全 byte_exact，0 ULP |

唯一 ULP 非零：ROPE_pos40960_boundary（max_ulp=8）——冻结 Cody-Waite 归约残差
（run_cosim.py 已注明为记录值、非门控）；**VCS 与 Verilator 仍 byte_exact**（8 ULP 是
对 fp32 执行器的差异，非 VCS 差异）。4 态 X 传播未造成任何分歧。

### 11.6 矩阵口径（如实）

co-sim matrix_engine 每时钟 1 MAC（128×1024×128 tile = 16.7M 周期）；全尺寸 M2A 线性
（N=2048=16 tile，~2.7 亿周期）在 4 态 VCS（实测 ~10k 周期/s）不可行。按计划「matrix
子集 + 单 tile GEMM」口径，矩阵通路以单 tile GEMM/GEMV（M≤128、K=128、N=128 一 tile）
覆盖 PF/DC + BF16/INT8 数据通路，byte_exact 可比；全尺寸矩阵数值/时序由 Verilator
侧（§7/§9）与 DC 代表基元（§10）覆盖。

### 11.7 交付物与需评审项

`asic/vcs/` 交付：`tb_qcore_top.sv`（SV 测试台）、`gen_rtl.py`（VCS 前端 desugar 快照
生成器）、`run_vcs_tests.py`（VCS/Verilator 逐字节 diff 驱动）、`Makefile`（snps-centos7
+ initreg 编译）、`vcs_results.json`（27 用例逐项结果）、`gen/`（快照）、`simv` +
`compile.log`。

**需评审项**：
1. **矩阵用例口径**：全尺寸 M2A（16 tile）在 4 态 VCS 不可行，以单 tile GEMM/GEMV
   （M≤128）覆盖矩阵通路（§11.6），是否接受作为 P10 §11「matrix 子集」结论。
2. **前端 desugar 口径**：command_processor 声明上提 + qmem 双写合并（§11.3）为语义
   中性改写，是否接受（与 §10.3 DC 的第 3 类 desugar 同性质）。
3. **运行环境口径**：VCS 编译/运行须经 CentOS 7 兼容命名空间（glibc 2.17），与本机
   glibc 2.35 的 `__pthread_unwind@GLIBC_PRIVATE` 不兼容（§11.2），是否记录为环境前提。
