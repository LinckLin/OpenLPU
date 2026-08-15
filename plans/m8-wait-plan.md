# M8 等待期并行计划（提案 v2，回应评审第 1 轮 5 条意见）

> v2 变更：①P10b 只写 asic/+docs/p10/——rtl/ 保持只读（冻结时序模型），流水线化落在
> asic/ 物理层（synth_datapath.sv/synth_mac.sv），物理流水级 ≤ 已计费 latency，M6 周期
> 契约不动；SRAM 缩编参数化作为窄例外（sram.sv/qcore_pkg 参数化，单写者 P10b），附加门槛
> = 默认参数下 golden3/co-sim trace 逐周期不变、缩编实例由 sram 专用 Verilator 单测（P10b）+
> ddr_if 侧 fpga smoke（后续小任务，P9b 后）验证；
> ②dma_kv_stage 补定义（挂接点/CP 组合读接口保持/多 outstanding）；③pf-boot 补批量导入
> 通道与带宽预算（UART 寄存器流不可行）；④G10 抽验改 python3 -m qforge；⑤README 增组件索引表。
> 状态：M0–M7、M9 ✅；M8 阻塞于板卡采购。

## 1. 背景

- G10（完全开源）缺口：无 README/LICENSE/复现指南。
- P10：Fmax 59.2 MHz(tt)，1 GHz 未达；流水化杠杆在 asic/ 物理层（rtl/ 是冻结的位精确
  功能+时序模型，被测网表来自 asic/gen desugar 派生）。
- P9 冻结清单板卡无关三项：延迟容忍读路径、SRAM 缩编、PF 引导定义。

## 2. 分工（含 rtl/ 写权约束）

| 流 | 代理 | 写目录 | rtl/ 写权 |
|----|------|--------|-----------|
| OpenSrc | G10 | README.md、LICENSE、docs/、docs/zhihu-outline.md | 无 |
| Pipe10 | P10b | asic/、docs/p10/；**窄例外：sram.sv/qcore_pkg.sv 的 SRAM 缩编参数化（唯一写者）** | 仅缩编参数化，默认参数回归门槛 |
| FpgaPrep | P9b | fpga/、docs/p9/ | 无 |

- **rtl/ 冻结契约**：P10b 对 rtl/ 的改动仅限 SRAM 参数化，且默认参数下
  golden3（15 实例三向）+ co-sim（4/4）+ 单层（14/14）trace 必须逐周期不变（回归门槛）；
  缩编实例验证载体 = sram 专用 Verilator 单测（P10b 跑）；ddr_if 侧 SRAM_BYTES 缩编 =
  后续小任务（P9b 后 smoke）。其他任何 rtl/ 改动均禁止（需评审）。
- **asicsnap 流程**：P10b 改动 asic/ 后重跑 run_synth.sh 复现 + 重 elaborate 记录 delta。

## 3. 任务

### G10（开源打包）
1. `README.md`：项目定位（QCore：个人级开源 LLM 推理加速平台）、架构图、目录结构、
   里程碑状态表、快速上手（真实入口 `python3 -m qforge` / `python3 -m qrun`）、依赖、
   模型缓存路径；**组件索引表**：G10 每组件一行（目录 / 入口命令 / 验证命令 / 对应文档）——
   compiler(qforge/、python3 -m qforge、verify_m3.py、docs/p4)、ISA spec(docs/spec*.md、
   test_isa_fields.py)、asm(compiler/isa、round-trip 测试)、sim(qsim/、timing_p6.py、
   docs/p3)、runtime(qrun/、m4.py、docs/p5)、RTL(rtl/、run_cosim.py、docs/p7)、
   FPGA(fpga/、run_fpga_smoke.py、docs/p9)、ASIC(asic/、run_synth.sh、docs/p10)。
2. `docs/reproduction.md`：从零复现（golden → qsim → qrun → RTL co-sim → ASIC）逐段命令
   与预期输出；/tmp qbin 重建命令写明。
3. `LICENSE`：Apache-2.0。
4. `docs/zhihu-outline.md`：系列文章大纲（5 篇 × 3–5 节提纲）。
5. 验收：`python3 -m qforge --help` 与 `python3 qsim/timing_p6.py` 可执行且输出与文档一致。

### P10b（asic/ 物理层流水化 + SRAM 缩编参数化）
1. STA 关键路径分析（OpenSTA 报告路径明细），定位前 3 长路径。
2. **asic/ 物理层流水化**：synth_datapath.sv / synth_mac.sv 加流水级（FP ALU、MAC 累加
   树），每周期 1 操作兑现冻结计费（多周期指令内部级数 ≤ 已计费 latency）；rtl/ 源不改。
3. **SRAM 缩编参数化**（rtl/ 窄例外）：sram.sv BANK_BYTES 与 qcore_pkg 相关量改参数化
   （默认 512 KiB/bank 不变）；缩编实例示例（256 KiB/bank → 4 MiB ≈ 片上 4.75 MiB 的 84%；
   128 KiB/bank → 2 MiB ≈ 42%；片上总量 ≈ 8 MiB 预算的 59%）；**缩编实例验证载体 =
   sram 专用 Verilator 单测（默认+缩编参数，P10b 跑）**；ddr_if 侧 SRAM_BYTES 缩编 =
   后续小任务（P9b 后 smoke）；默认参数全回归（golden3+co-sim+单层）逐周期不变。
4. 重综合 + STA：目标 Fmax(tt) ≥ 200 MHz（sky130 现实目标；1 GHz 差距记录）。
5. 更新 docs/p10/asic-report.md（delta、新 Fmax、关键路径明细、缩编参数说明）。

### P9b（板卡无关前置）
1. `fpga/dma_kv_stage.sv` 定义（补全）：**挂接点 = ddr_if 引擎字节口内侧**（CP 组合读接口
   保持：stage 在**已计费 T_FIRST（=DDR_RD_LATENCY）时点内**交付数据——DDR_RD_LATENCY 即
   qcore_pkg 的 T_FIRST 常量，不得双计为 200，CP 侧零延迟语义不变）；
   **多 outstanding**：扩展 ddr_axi4_master 支持多事务在途 + 重排序缓冲（或 stage 内自带
   AXI 引擎——实现二选一，报告注明）；Verilator 单元测试（随机延迟注入 + 乱序应答）。
   本产物构成 porting.md §5.6 二选一（"ddr_if 侧吸收延迟"）的预裁决提案，报告注明。
2. `docs/p9/pf-boot.md`：PF 引导定义冻结提案——**qsim 侧 PF + KV 导入板卡**；导入通道 =
   **ZCU104 PS DDR 共享：权重与 KV 由 PS 侧预写入 PS DDR，QCore 经 AXI HP 口（PS DDR
   控制器）直读，零额外通道**（PS DDR 为硬化控制器直连、非 MIG；备选 PS-DMA/PCIe/SD）；
   **带宽预算表**：0.6 GB 权重 + 0.44 GiB(4K)/0.875 GiB(8K) KV × 通道带宽 → 导入时间；
   演示上下文建议 4K 起步。
3. 验收：dma_kv_stage 单测过；pf-boot.md 定义完整且带宽预算可执行。

## 4. 验收与评审

- 三流交付后各经评审一致（W1/W2/W3 小里程碑）。
- 板卡到位后 P9 立项直接引用（porting.md 冻结清单 3 项提前完成）。

## 5. 风险与对策

| 风险 | 对策 |
|------|------|
| P10b 物理流水改变网表语义 | 网表级回归：合成后网表 vs 原网表功能等价抽查（关键 op 向量） |
| SRAM 参数化破坏默认回归 | 默认参数全回归门槛（trace 逐周期不变）写进验收 |
| dma_kv_stage 多 outstanding 复杂度 | 若 stage 自带引擎过重，降级为扩展 master 2-outstanding（报告注明） |
| LICENSE 选型争议 | Apache-2.0 默认；异议走评审 |
| 200 MHz 目标不达 | 如实报告（分级方案与差距）；不虚构数字 |
