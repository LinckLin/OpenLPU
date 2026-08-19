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
