# ASIC 专项：DC 流程与仿真（提案 v2，回应评审第 1 轮 6 条意见）

> v2 变更：①asic-report.md 共享写规则（各写 §10/§11、§0 与需评审项由主会话合并）；
> ②库路径订正为 /home/lzl/.eda/liberty（5 corner，含 power 表）；③ss corner 基线方案 =
> 先补跑 OpenSTA ss_100C_1v60 同口径基线再比（1 ns 虚拟时钟、Fmax=1/arrival）；④report_power
> 口径 = DC 只报漏电+面积，动态功耗沿用 §5 估算（VCD 回注为后续，不设跨流依赖）；⑤VCS 对齐
> 判据 = trace 逐记录一致 + 内存逐字节一致（ULP 仅作例外注明）；⑥synth_top 综合时 sram_macro
> 以 mem_stub.lib 黑盒 + set_dont_touch。
> 背景：用户指示板卡暂缓、聚焦 ASIC（含 DC 仿真）。环境：DC O-2018.06-SP1 + VCS + ICC2/FM/
> SpyGlass（license 27000@bics109 已实测）。现状：Yosys+OpenSTA sky130 流水化 129 MHz（tt）。

## 1. 范围与目标

1. **DC 综合流程**：sky130 liberty → DC elaborate → compile_ultra → report_timing/area/
   power 多 corner，对 asic/gen 派生设计（synth_datapath/synth_mac/synth_top），与
   Yosys/OpenSTA 基准交叉验证。
2. **VCS 功能仿真**：qcore_top 在 VCS 下跑既有 co-sim 用例，与 Verilator 结果**字节级一致**
   （trace 逐记录 + 内存逐字节）。
3. 全设计 DC 综合尝试（synth_top，SRAM 黑盒口径）；SV 前端兼容问题如实记录并界定。

## 2. 分工（两路并行）

| 流 | 代理 | 写目录 | 依赖 |
|----|------|--------|------|
| DcFlow | DcSyn | asic/dc/、docs/p10/（仅 §10 节） | 无 |
| VcsSim | VcsRun | asic/vcs/、docs/p10/（仅 §11 节） | 无 |

- **共享写规则**：两代理只追加各自 §10/§11；禁止改动 asic-report.md §0–§9；
  §0 结论速览与需评审项由主会话收流后合并。
- 两代理对 rtl/、asic/gen/ 只读。

## 3. 任务

### DcSyn（DC 综合）
1. 工具验证：dc_shell -version、license 实测（get_license/checkout + lmstat 27000@bics109）；
   记录 DC_HOME/版本。
2. 库准备：sky130 liberty（**/home/lzl/.eda/liberty/** 现有 5 corner，含 internal/leakage
   power 表）→ DC read_lib（或转 .db）；记录读入方式。
3. 流程脚本（asic/dc/）：elaborate（desugar 产物 read_verilog）→ compile_ultra →
   report_timing/area/power（tt_025C_1v80 + ss_100C_1v60 多 corner）。
4. 交叉验证：**先补跑 OpenSTA ss_100C_1v60 同口径基线（sta.tcl + LIB 环境变量，1 ns 虚拟
   时钟、Fmax=1/arrival）**；DC vs OpenSTA 对 tt 与 ss 两 corner 逐项对比，工具/effort
   差异如实记录。**report_power 口径：DC 只报漏电+面积；动态功耗沿用 §5 活动因子估算**
   （VCD 回注为后续，不设跨流依赖）。
5. 全设计：synth_top 综合（**sram_macro 以 mem_stub.lib 黑盒 + set_dont_touch**）；SV 前端
   兼容问题如实记录并界定。
6. 交付：asic/dc/（tcl 脚本 + 报告原文）+ docs/p10/asic-report.md 增 §10 DC 节（仅此节）。

### VcsRun（VCS 仿真）
1. 工具验证：vcs -ID 版本、license、编译冒烟。
2. 移植 co-sim 用例：SV testbench 重写（sim_main.cpp 为 Verilator 专用 API，RTL 无 DPI，
   重写工作量可控），跑既有指令级用例（vector/KV/matrix 子集 + 单 tile GEMM）。
3. 交叉验证：**判据 = trace 逐记录一致 + 最终内存逐字节一致**（与 cosim.py 精确比对口径
   一致）；任何 ULP 容忍仅作例外并注明原因。
4. 交付：asic/vcs/（makefile + tb + 报告）+ docs/p10/asic-report.md 增 §11 VCS 节（仅此节）。

## 4. 验收

- DcSyn：DC 综合跑通，tt+ss 两 corner 时序/漏电/面积报告产出，与 OpenSTA 同口径基线
  交叉验证记录完整；全设计综合结果（成功或兼容性界定）如实。
- VcsRun：VCS 编译跑通 + 用例字节级一致（trace + 内存），与 Verilator 结果一致证据。

## 5. 风险与对策

| 风险 | 对策 |
|------|------|
| DC license 不可用/受限 | 如实记录；受限则限定 ALU/MAC 级跑通流程 |
| DC 2018 前端 SV 语法不兼容 | preprocess desugar 已备；不兼容项如实记录并界定 |
| VCS 4 态 X 传播与 Verilator 2 态差异 | 字节级一致判据直接暴露差异；分歧即查（非放宽） |
| compile_ultra 时长 | ALU/MAC 级先验证；全设计设时间上限（记录中断点） |
