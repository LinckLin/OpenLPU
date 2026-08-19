# Matrix 物理计算核推进计划（Track 2.4）

> 目标：把当前 `matrix_compute_core` 黑盒逐级替换为与冻结规格一致、可综合和可物理实现的
> 128×128 weight-stationary dual-MAC 阵列，并最终向全顶层提供可信 timing/area/physical
> view。`rtl/matrix_engine.sv` 是每拍 1 MAC 的数值 co-sim 模型，不能直接当作物理阵列。

## 1. 不可退让的规格约束

- 128×128 PE；每 PE 两个 signed INT8 乘法器，INT8 W8A8 为 2 MAC/PE/cycle；
- 每个 PE 驻留两个权重，激活向东、INT32 partial sum 向南传播；
- 每个方向每 PE 一拍，维持 128 列 + 128 行 = 256 cycle fill/drain；
- 稳态 32,768 INT8 MAC/cycle，对应 32.77 TMAC/s @ 1 GHz；
- 物理层新增流水不得偷偷改写上述吞吐/延迟。无法闭合时必须量化并回到规格裁决；
- 商业 SMIC28 `.lib/.db/.lef/.gds2/.cdl` 只在本机使用，不提交公开仓库。

## 2. 分阶段实现

### Track 2.4a：dual-MAC PE

实现单拍 `matrix_int8_pe`：两路 `INT8×INT8` 与北向 INT32 partial sum 在一个 PE hop
内合并，两份权重成对加载，激活/partial sum 同拍注册输出。用带真实 launch flop 的 probe
做 SMIC28 HDC30P140 RVT TT/SS 1 ns `compile_ultra`，避免零延时顶层输入掩盖相邻 PE
clock-to-Q。功能门槛为边界值、溢出模 2^32、bubble、权重重载、完整 8-bit
activation×weight 输入域及背靠背随机 partial sum 全部通过。

### Track 2.4b：16×16 结构 tile

以 16×16 tile 作为布局复用层级，而非缩小替代物；最终精确组成 8×8 tile grid =
128×128 PE。实现激活/partial-sum 波前 skew、每 PE 权重加载寻址和 tile 边界 valid，做
小矩阵逐周期 scoreboard 与 TT/SS 分层综合。报告 tile 实测面积，不用单 PE 简单缩放
冒充布线后面积。

### Track 2.4c：128×128 阵列与抽象模型

结构化例化 64 个 16×16 tile，加入 PF 权重双缓冲和 DC 16 lane×8 行重构网络；验证
32,768 INT8 MAC/cycle、256-cycle fill/drain 与 PF/DC 调度。DC 2018 无
`write_timing_model`/`extract_model`，因此抽象交付走经验证的 QTM `.db` 或物理工具导出的
Liberty；在模型与门级顶层交叉 STA 一致前，不把 quick timing model 称为 signoff Liberty。

### Track 2.4d：dtype 与后处理

在同一双乘法器上补 INT4 W4A8 四 MAC/PE/cycle、BF16 四部分积/两周期和 W4A16 路径；
阵列底部接 128-wide INT32→scale→FP accumulate→bias→convert 后处理。逐模式与
softfloat/qsim 做 bit-exact 或冻结 ULP 门槛验证。

### Track 2.4e：物理闭合

导入标准单元与 SRAM LEF，在 9 个状态 SRAM、32 KiB 权重存储、tile grid 和后处理之间
完成 floorplan、PG、CTS、布线与提取寄生 STA；TT/SS 之外补所需 signoff corners。最终
以布局后 Fmax、面积、功耗和拥塞报告替代 pre-layout probe。

## 3. Track 2.4a 验收

1. 单拍 PE 数据流和双 MAC 数值验证全过，连续 valid 吞吐 1 result/cycle；
2. Verilator lint 通过；SMIC28 TT/SS 1 ns probe 均生成正式 timing/area/power 报告；
3. 报告明确区分 PE 本体、probe launch flops 与 128×128 理论复制下界；
4. 若任一角落不满足 1 ns，不增加隐藏流水，记录违例并把 retiming/fill-drain 变化送入
   下一阶段架构裁决；
5. 更新 `docs/p10/asic-report.md` 与 `HANDOVER.md`，独立提交并推送 GitHub。

## 4. Track 2.4a 实测收口（2026-08-19）

状态：**完成**。`matrix_int8_pe` 保持单 PE hop 一拍，连续 valid 时每拍提交两次
signed INT8 MAC。核心穷举了每路完整的 256×256 activation/weight 域，共 65,544 次
检查；registered probe 随机/bubble 检查 964 次；TT 映射网表使用官方 HDC30P140
functional model 做 1,928 次独立门级检查，均为 0 failure。

SMIC28 corner 名沿用流程兼容标签；实际库映射为 TT 0.90 V/25 °C 与 SSG 0.81 V/125 °C。
功耗是无 VCD 的低努力 probe 总值，只作相对观察，不做阵列功耗外推。

| corner | 约束 | 最差 arrival / slack | PE 本体面积 | probe 总面积 | dynamic / leakage |
|---|---:|---:|---:|---:|---:|
| TT | 1.0 ns | 0.98 / **+0.00 ns MET** | 560.462 µm² | 680.610 µm² | 503.734 / 5.095 µW |
| SS | 1.0 ns | 0.97 / **+0.00 ns MET** | 631.414 µm² | 751.562 µm² | 433.185 / 62.898 µW |
| TT | 0.9 ns | 0.87 / **+0.01 ns MET** | 580.258 µm² | 700.406 µm² | 565.470 / 5.174 µW |
| SS | 0.9 ns | 0.87 / **+0.00 ns MET** | 691.194 µm² | 811.930 µm² | 500.769 / 69.795 µW |

1.0 ns 映射以报告显示精度刚好闭合，没有可声称的布局后余量；0.9 ns 映射证明可用
TT **+3.53%**、SS **+9.47%** 的 PE cell area 换取约 10% 的前布局周期预算，但仍不含
时钟树、互连和提取寄生。按 1.0 ns 的单 PE cell area 直接复制 16,384 个 PE，仅得到
TT/SS **9.183/10.345 mm²** 的标准单元面积下限，不能当作 tile 或整阵列布局面积。

下一门槛是 Track 2.4b：构造 16×16 真实互连 tile，逐周期验证 skew/valid/权重寻址，
并用分层综合及早暴露 256 个 PE 的扇出、布线和编译容量问题。

## 5. Track 2.4b 实测收口（2026-08-19）

状态：**完成**。`matrix_int8_pe_tile` 结构化例化 256 个 PE，边界把 activation 与
partial-sum valid 独立 skew：west row `r` 在 `launch+r` 进入，north column `c` 在
`launch+c` 进入，PE(row,col) 在 `launch+r+c` 汇合。权重用 `(row,col)` 单地址串行加载，
计算阶段不与加载重叠。该接口保持每个 PE 一拍 hop，不改变规格的 wavefront 深度。

验证门槛全部通过：

- Verilator 波前 scoreboard：**2,720 checks / 0 failure**，含两拍 bubble、全 16×16
  权重寻址、east activation payload 和 south INT32 累加；
- TT 映射网表 + 官方 HDC30P140 functional model：Icarus **1,168 checks / 0 failure**；
- Verilator lint、Icarus 语法检查、DC desugar 均通过。

SMIC28 HDC30P140 RVT 1 ns registered-boundary probe 结果：

| corner | 最差路径 | arrival / slack | tile `u_tile` cell area | probe 总 cell area | cells / sequential | dynamic / leakage |
|---|---|---:|---:|---:|---:|---:|
| TT | `gen_row[8].gen_col[4].u_pe/weight0_reg[0] → .../psum_south_reg[31]` | 0.98 / **+0.00 ns MET** | 143,261.986 µm² | 145,040.686 µm² | 229,229 / 17,465 | 64.166 / 1.067 mW |
| SS | `gen_row[13].gen_col[0].u_pe/act1_east_reg[1] → gen_row[13].gen_col[1].u_pe/psum_south_reg[17]` | 0.97 / **+0.00 ns MET** | 158,331.250 µm² | 160,110.930 µm² | 296,749 / 17,465 | 54.403 / 13.288 mW |

TT/SS 报告分别为 267,699/324,949 nets、3,254 ports；compile_ultra CPU time 为约
709/1,381 s，峰值 session memory 约 1.65/1.70 GB。DC 输出网表保留 `u_tile` 物理层次
边界，但 PE 逻辑被展平为 tile 内组合逻辑（timing path 仍保留 `gen_row/gen_col/u_pe`
来源名），因此不能把每个 `u_pe` 当作独立 hard macro。

按 64 个 tile 直接复制 `u_tile` cell area，得到 TT/SS **9.1688/10.1332 mm²** 的标准
单元复制下限；若把每 tile 的 registered probe 也算入，则为 **9.2826/10.2471 mm²**。
这些数字不含 tile-to-tile 边界寄存、8×8 权重/valid 分发、时钟树、SRAM、后处理、拥塞
和寄生。tile 内部单 PE 平均面积比独立 PE 报告低约 0.15%/2.05%，是 DC 跨实例优化结果，
不是物理布局收益，不能据此替代 P&R 面积。

两角报告都显示 `clk` 约 **17,465 loads**，DC 对高扇出路径采用 fanout=1000 的 wire-load
估算（TIM-134）。因此 1 ns 的 `MET` 仍是 pre-layout 数字，下一阶段必须在 64-tile 组装
前处理 clock distribution，并用 CTS/寄生 STA 重新裁决频率；不能把 tile 探针当作阵列
signoff。

Track 2.4c 的目标是例化 64 个 tile，验证 32,768 MAC/cycle 与 256-cycle fill/drain，
加入 PF 双缓冲权重接口和 DC lane 重构网络，再评估是否需要物理层级/时钟分区；实测收口
见下节。

## 6. Track 2.4c 实测收口（2026-08-19）

状态：**结构功能闭合，物理 signoff 未开始**。`asic/matrix_int8_pe_array.sv` 默认例化
8×8 个 `matrix_int8_pe_tile`，即 64 个 16×16 tile、16,384 个 dual-MAC PE。PF 权重
接口改为一整行并行写入：128 个列位置每拍各接收两路 INT8，共 **256 B/cycle**；一份
32 KiB bank 在 128 个 row-cycle 内完成，第二 bank 可在活动 bank 计算期间重载，随后以
`weight_commit` 原子切换；阵列用最长 hop drain counter 拒绝在途 wave 中的切换。

验证门槛：

- 2×2 tile-grid（32×32 PE）控制/算术/DC scoreboard：**8,960 checks / 0 failure**；
  覆盖双 bank、非活动 bank overlap load、PF wave、300-cycle MODE barrier 和 DC
  重构 wave；
- 完整 8×8 tile-grid registered-boundary scoreboard：**197,635 checks / 0 failure**；
  连续 256 个 PF wave 覆盖全 128×128 边界，最大同时活动 **16,384 PE = 32,768
  INT8 MAC/cycle**，wave 0 的最后输出槽对应 **256-cycle fill/drain**；
- `verilator --lint-only`（2×2 参数化与默认 full probe）及 `desugar_dc.py` 通过。

DC flow 已加入 `matrix_int8_pe_array` 入口并保留 tile 层次，但本轮不运行完整 64-tile
`compile_ultra`：DC 2018 对 packed DC lane reconstruction 会展开很大的寄存器管线，且
尚无可用的阵列 Liberty/LEF、CTS、布线和寄生。因而本节点只交付结构 RTL 证据，**不报告
完整阵列 PPA，也不把理想时钟下的 tile 复制面积当作 signoff**。DC 重构管线是后续物理
优化的明确风险点，应在 floorplan 前改成分 lane/分 tile 的局部 FIFO 或物理寄存器切片。

下一门槛是 Track 2.4d（dtype/后处理）与 Track 2.4e（阵列物理集成）之间的裁决：先确定
DC lane 网络的可综合物理实现，再导入真实 tile boundary、32 KiB 权重 SRAM、后处理和
CTS/寄生 STA；在此之前维持本节点的 32,768 MAC/cycle 只作为功能吞吐契约。
