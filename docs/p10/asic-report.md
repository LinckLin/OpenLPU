# QCore ASIC 流程报告（P10 / M9）

> 状态：M9 验收交付。Yosys 0.44 + SkyWater 130 nm（sky130_fd_sc_hd）逻辑真综合；
> OpenSTA 多 corner STA；SRAM 宏按公开密度上下界估算并单列占比。
> 冻结快照：`rtl/ref/asicsnap/`（开工时打点；rtl/ 只读，P8 未改 rtl/ 源）。

## 0. 结论速览

| 项 | 结果 |
|---|---|
| elaboration | ✅ 通过（Verilator 4.038 原 RTL lint + Yosys 0.44 层级检查，见 §1） |
| 逻辑综合 | ✅ 通过（8 项 FP 基元 → sky130 门级网表，见 §2） |
| 时序收敛结论 | ⚠️ **1 GHz 未收敛**；M9 直接组合映射 Fmax ≈ 59 MHz（tt）。**P10b 流水化后 Fmax(tt) ≈ 129 MHz**（OpenSTA/Yosys 口径，add/sub 7.77 ns 限制）；**DC compile_ultra 口径：tt 331 MHz / ss 169 MHz**（基元级，§10——STA 引擎交叉验证 <0.1% 一致，差距归综合工具）；200 MHz 目标在 Yosys 口径未达、DC 口径达成（双口径如实并列） |
| 面积 | M9 组合映射口径（实测）：FP ALU 0.0331 / BF16 MAC 0.0273 / INT8 MAC 0.0042 mm²；P10b 流水化口径（实测，run_synth.sh）：ALU top 0.0609 / mac_bf16 ≈0.0349 / mac_int8 ≈0.0053 mm²；SRAM 100.7–268.4 mm²（估算） |
| token/s | 1 GHz 口径 960/675/469；P10b 流水化口径（129 MHz）≈ 124/87/60；DC 口径（331 MHz）≈ 318/223/155（见 §6/§9/§10） |


> **DC/VCS 补充（§10/§11）**：Synopsys DC O-2018.06-SP1 compile_ultra（基元级）与
> VCS-MX O-2018.09-SP2 功能仿真（27 用例与 Verilator 字节级一致）已交付；
> **O6 全设计扩展：synth_top 在 DC 侧 elaborate + link + compile_ultra 全跑通**
> （控制平面 1.000 mm² / Fmax 136 MHz，数字核/SRAM 按物理宏黑盒，§10.5）。
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
> O-2018.06-SP1 + sky130 5-corner liberty。代表数据通路（synth_datapath / mac_bf16）
> 跑通 compile_ultra 并出 tt/ss 两 corner 时序/漏电/面积；与 OpenSTA 同口径交叉验证；
> **全设计 synth_top elaborate + link + compile_ultra 跑通**（存储/数字核黑盒化，
> 控制平面真综合，§10.5）。

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
| 4 | 运行期 `for (i < len)` 组合循环 + 推断 RAM | vector_engine/matrix_engine（co-sim 功能模型） | 资源/语义边界（~1.7 M 触发器 + 128-lane softfloat） | **黑盒化**（数字核 + 存储 → `(* blackbox *)` 宏，见 §10.5） |
| 5 | co-sim 端口宽度失配（vector_engine `len` 32b vs CP 16b） | vector_engine↔command_processor | LINK-3/LINK-25（linker 拒绝） | 黑盒端口收窄对齐 CP（§10.5） |

### 10.4 DC vs OpenSTA 交叉验证（步骤 4）

**先补跑 OpenSTA ss_100C_1v60 同口径基线**（sta.tcl + LIB 环境变量，1 ns 虚拟时钟、
Fmax=1/arrival，流水化 synth_datapath）——§9.3 只报了 tt，本节点补 ss：

| corner | OpenSTA 关键路径（data arrival） | Fmax = 1/arrival |
|---|---|---|
| tt_025C_1v80 | 7.7686 ns（u_add） | **128.7 MHz** |
| ss_100C_1v60 | 15.3851 ns（u_mul） | **65.0 MHz** |

DC compile_ultra 结果（同口径 1 ns 探针时钟，`report_timing` data arrival）：

| design | corner | 关键路径 | Fmax | 面积 (µm² / mm²) | 漏电 |
|---|---|---|---|---|---|
| synth_datapath | tt | 3.02 ns | **331 MHz** | 61667.9 / 0.0617 | 29.83 nW |
| synth_datapath | ss | 5.90 ns | **169 MHz** | 62159.6 / 0.0622 | 38.73 µW |
| mac_bf16 | tt | 3.03 ns | **330 MHz** | 36029.6 / 0.0360 | 16.33 nW |
| mac_bf16 | ss | 5.64 ns | **177 MHz** | 38670.7 / 0.0387 | 23.76 µW |

**逐项对比（DC vs OpenSTA/Yosys）**：

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
估算（DC report_power 的 40.4 mW 动态值基于缺省翻转率，不采用；VCD 回注为后续）。
漏电 ss(100 °C) 较 tt(25 °C) 高 ~1300×，与 liberty 一致（NAND2 cell_leakage 0.0021 nW
→ 2.27 nW）。

### 10.5 全设计 synth_top（步骤 5，全设计 elaborate + compile_ultra 跑通）

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
`hoist_dc.py`（新增第 4 类存储/数字核黑盒化 + 第 5 类端口宽度修正）/ `clean_lib.py` /
`sta_dc.tcl` / `mem_stub.lib`（DC-local）/ `bb_sram.sv`（指令流 SRAM 黑盒）+ `db/`
（tt/ss `.db` + mem_stub.db）+ `gen/` / `gen_full/`（desugar 产物，含黑盒版
matrix_engine/vector_engine）+ `reports/`（基元 4 份 rpt + 4 份网表 + 全设计
synth_top tt/ss 2 份 rpt + 2 份网表 .v）。

**需评审项**：
1. **Fmax 口径重标定**：DC compile_ultra 对同一 liberty 的 Fmax(tt)=331 MHz，显著高于
   Yosys/OpenSTA 的 129 MHz（§9.3），差因为综合工具（§10.4 已证 STA 一致）；是否以
   DC 结果作为物理数据通路新口径（与 §9.5 的 BF16 重标定并列）。
2. **全设计口径（O6 更新）**：全设计 synth_top 现已 elaborate + link + compile_ultra
   跑通，但矩阵/向量数字核与 SRAM 按物理宏边界黑盒（基元在 §10.4 真综合）、控制平面
   1.000 mm² / Fmax 136 MHz 为如实口径——是否接受「全设计控制平面真综合 + 数字核/SRAM
   宏黑盒」作为 P10 §10 结论。
3. **黑盒接口口径**：vector_engine 黑盒端口 `len` 由 32 bit 收窄为 16 bit 对齐 CP
   （co-sim 静默加宽、DC linker 严格），属黑盒副本（gen_full/）语义中性修正，是否接受。
4. **license/库兼容**：sky130 liberty 需 clean_lib.py 清理才可被 LC 读取、mem_stub.lib
   时序模型需 constraint 模板修正且 DC 侧改用空模块黑盒——两处库修正是否接受。

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