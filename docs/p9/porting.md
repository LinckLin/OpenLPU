# P9 板卡无关移植 — 接口层、载体映射与立项冻结清单

> 状态：接口层 + 集成 smoke 通过（bf16 ≤1 ULP、trace 一致，与 M6 co-sim 同判据）。
> 依据：`plans/p8-p10-plan.md` §5（P9 任务）、§7（风险）、`docs/p9/board-candidates.md`（板卡候选）。
> 本任务只写 `fpga/`（SystemVerilog 接口层 + 集成 + 测试）与 `docs/p9/`；`rtl/` 一字未改。

## 0. 结论摘要（TL;DR）

| 项 | 结论 |
|----|------|
| 三模块接口层 | `clock_reset.sv` / `host_if.sv` / `ddr_if.sv`（+ 支撑模块 `ddr_axi4_master.sv`）全部 Verilator 4.038 lint 通过、单元测试/集成 smoke 通过 |
| 集成 | `qcore_fpga_top.sv` 把未改动的 `command_processor` 挂在 host_if + ddr_if + clock_reset 上 |
| 集成 smoke | 4 用例 ALL PASS：VADD / RMSNORM / KV.APPEND+LOAD / GEMM_BF16，trace 全对齐、bf16 ≤1 ULP（实测 0.0） |
| 板卡无关性 | 宿主控制面 = 扁平寄存器接口；DDR = AXI4 风格窄接口（64B 突发、参数化延迟）；板卡到位后只适配物理 IP（UART/PCIe 桥 + MIG） |
| 首选载体 | ZCU104（ZU7EV）：UART 控制面 + DDR4 MIG（64-bit）；板卡候选见 `board-candidates.md` |

## 1. 文件布局

| 文件 | 职责 |
|------|------|
| `fpga/clock_reset.sv` | 时钟域骨架：单时钟域 + 异步复位低有效同步释放（立项约定） |
| `fpga/host_if.sv` | 宿主-设备控制面：配置/命令队列装载/qbin 装载/logits 回读（寄存器接口） |
| `fpga/ddr_if.sv` | DDR 内存子系统抽象（替代 qmem 的 HBM 模型）：SRAM 片上 + DDR 经 AXI4 窄接口 |
| `fpga/ddr_axi4_master.sv` | 支撑模块：AXI4 风格窄突发主机（64B 突发、参数化延迟模型） |
| `fpga/qcore_fpga_top.sv` | 集成顶层：clock_reset + host_if + ddr_if + command_processor |
| `fpga/tb/sim_fpga_main.cpp` | 集成 smoke 的 Verilator C++ 驱动（全程走 host_if 寄存器） |
| `fpga/tb/run_fpga_smoke.py` | 集成 smoke 驱动（qsim 基线对比，M6 判据） |
| `fpga/tb/clock_reset_tb.sv` + `sim_clock_reset.cpp` | clock_reset 单元测试 |
| `fpga/tb/ddr_axi4_tb.sv` + `sim_ddr_axi4.cpp` | ddr_axi4_master 单元测试（含行为级 DDR slave） |
| `fpga/tb/sram_shrink_tb.sv` + `sim_sram_shrink.cpp` + `run_sram_shrink.sh` | ddr_if SRAM_BYTES 缩编实例 smoke（读写往返 + 顶边界 + DDR 隔离 + 宿主口回环） |

> 本任务未改动 `rtl/`、`qsim/`、`qrun/` 等任何其它目录。`command_processor` 及其接口保持 M6 原样。

## 2. 接口定义

### 2.1 clock_reset.sv — 时钟域骨架

```
module clock_reset #(parameter int RST_STAGES = 2) (
  input  logic clk_i,            // 外部单时钟域时钟
  input  logic async_rst_n_i,    // 外部异步复位，低有效
  output logic clk,              // 单时钟域（直通）
  output logic rst_n             // 同步释放的复位，低有效
);
```

- **约定**（plans/p6-p7-plan.md §4.1）：单时钟域 1 GHz（1 cyc = 1 ns）；异步复位低有效、同步释放。
- **行为**：`async_rst_n_i` 的下降沿**异步**置低 `rst_n`（时钟未稳定时即可保持复位）；释放经 `RST_STAGES` 级同步器，`RST_STAGES` 个 `posedge clk_i` 后才拉高（防亚稳态）。
- 板卡映射：`clk_i` 接板卡时钟（或 MMCM/PLL 输出）；MIG 的 DDR 时钟在 P9 立项冻结时加入，仍属同一功能域。

### 2.2 host_if.sv — 宿主控制面（寄存器接口）

宿主侧是一个扁平 32-bit 寄存器文件（`host_addr[11:0]` 字地址、`host_wdata`、`host_wen`/`host_ren`、`host_rdata`、`host_ready` 恒 1 的 no-wait CSR 总线）。四个控制面功能：

| 功能 | 寄存器 | 说明 |
|------|--------|------|
| 配置 | `CTRL(0x00)` | `[0]`=start 脉冲、`[1]`=软复位 |
| 命令队列装载 | `PROG_LEN(0x06)`、`CMDQ_W0..W3(0x08..0x0B)`、`CMDQ_GO(0x0C)` | 128-bit Q-ISA 指令分 4 个 32-bit 字写入，`CMDQ_GO` 提交并自增地址 |
| qbin 装载 | `MEM_ADDR_LO/HI(0x10/0x11)`、`MEM_SEL(0x12)`、`MEM_WDATA(0x13)` | `[7:0]`=字节、`[8]`=提交（写字节 + 地址自增） |
| logits 回读 | `MEM_RDATA(0x14)`、`MEM_ADV(0x15)` | `[7:0]`=字节，`MEM_ADV` 写触发地址自增 |
| 状态 | `STATUS(0x01)`、`TOTAL_LO/HI(0x02/0x03)`、`TRACE(0x04)`、`TRACE_CYC(0x05)` | done/running、total_cycles、trace_valid/index/cycles |

设备侧接到 `command_processor` 的 imem 后门（`imem_waddr/we/wdata/prog_len`）与 `ddr_if` 的宿主端口（`hd_*`）。**物理载体（UART/以太网/PCIe）在 P9 立项冻结时映射到这套寄存器**——器件逻辑与载体无关。

### 2.3 ddr_if.sv — DDR 内存子系统抽象

```
module ddr_if #(
  parameter int SRAM_BYTES = 8*1024*1024,      // 片上 scratchpad 8 MiB
  parameter int DDR_BYTES  = 1024*1024*1024,   // DDR 窗口 1 GiB
  parameter int DDR_ADDR_BITS = 40, DDR_DATA_BYTES = 8,
  parameter int DDR_BURST_BYTES = 64,
  parameter int DDR_RD_LATENCY = 100, DDR_WR_LATENCY = 100  // = qcore_pkg T_FIRST
) ( ... );
```

三个端口面：

1. **引擎字节口**（与 qmem 完全一致）：`rd_sel/rd_addr → rd_data`（组合读）、`wr_en/wr_sel/wr_addr/wr_data`（posedge 写）。`rd_sel=0`=SRAM、`1`=DDR。**这是 M6 语义不变的保证**——未改动的 `command_processor` 原样接入。
2. **宿主字节口**（`hd_*`，替代 qmem 的测试后门 `bd_*`）：qbin 装载 / logits 回读，由 host_if 驱动。
3. **AXI4 风格窄主机口**（`m_axi_*`）：面向板卡 DDR 控制器（MIG）。64B 突发、`DATA_BYTES=8`（64-bit 窄接口，8 拍）、`DDR_RD/WR_LATENCY` 参数化延迟。

**功能模型 vs 物理接口**：co-sim（本构建）中引擎/宿主字节口由零延迟功能数组（SRAM 平坦数组 + DDR 稀疏关联数组，语义与 qmem 完全一致）服务，AXI4 主机空闲——因此集成 smoke 与 M6 co-sim 判据**逐位一致**。板卡到位后，DMA/KV 数据通路改走 AXI4（见 §5 冻结清单第 6 条），稀疏数组由真实 DDR 替代——**只适配物理 IP，不改器件逻辑**。

### 2.4 ddr_axi4_master.sv — AXI4 窄突发主机（支撑模块）

把一条 64B 请求转成 AXI4 INCR 突发（`arlen/awlen = 64/8-1 = 7`，`arsize/awsize = log2(8) = 3`，`wstrb` 全 1）。读/写固定延迟用计数器建模（`RD_LATENCY`/`WR_LATENCY`，默认 100 = `T_FIRST`）。单事务在途（`req_ready` 当 IDLE）。与冻结周期模型（DMA.STORE / KV.* = `T_FIRST + hbm_*_cycles`）对应——板卡真实 MIG 延迟在立项冻结时回填到这两个参数。

## 3. 集成（qcore_fpga_top.sv）

```
clk_i / async_rst_n_i ── clock_reset ──> clk / rst_n
host 寄存器总线 ── host_if ──> imem 后门 + start + (qbin/logits 经 hd_*)
command_processor <──> ddr_if（引擎字节口，qmem 等价）
ddr_if ── AXI4 窄主机口 ──> 板卡 DDR 控制器（MIG）
观测口 done/total_cycles/trace_*（镜像 qcore_top，供 testbench；宿主经 STATUS/TOTAL/TRACE 读同一值）
```

集成 smoke（`fpga/tb/run_fpga_smoke.py`）与 M6 判据一致，全程只经 host_if 寄存器（无测试后门）：

```
[PASS] VADD             trace=True max_ulp=0.0 cycles=5 (expected 5)
[PASS] RMSNORM          trace=True max_ulp=0.0 cycles=20 (expected 20)
[PASS] KV_APPEND_LOAD   trace=True max_ulp=0.0 cycles=211 (expected 211)
[PASS] GEMM_BF16        trace=True max_ulp=0.0 cycles=266 (expected 266)
P9 FPGA smoke: ALL PASS
```

覆盖：vector 引擎（VADD/RMSNORM，SRAM 编组）、DMA+KV 地址生成（APPEND 写 DDR → LOAD 读回）、matrix 引擎（GEMM M=4/N=128/K=128，周期 266 与 M6 报告一致）。判据 = M6 口径：逐指令周期 trace 精确相等 + 最终内存 bf16 ≤1 ULP（`|y|≥1e-3` 元素）。

单元测试：`clock_reset_tb`（异步置位 + 同步释放 3 级）、`ddr_axi4_tb`（64B 突发写+读回 + 延迟下限检查）均 PASS。

## 4. 载体映射方案（首选 ZCU104）

首选板卡 ZCU104（XCZU7EV，见 `board-candidates.md`）：

| 抽象 | 板卡映射 | 备注 |
|------|----------|------|
| 宿主控制面 | **UART**（PS→PL 或 PL 侧 UART 桥）→ host_if 寄存器 | 控制面带宽低（程序 + qbin 装载 + logits 回读），UART 足够；PCIe/以太网为后续可选升级 |
| DDR（rd_sel=1） | **DDR4 MIG**（64-bit，2 GB = 1.86 GiB ≥1.5 GiB 下限） | ddr_if 的 AXI4 窄主机口接 MIG 的 S_AXI 口；`DDR_RD/WR_LATENCY` 回填 MIG 实测延迟 |
| SRAM（rd_sel=0） | **片上 UltraRAM + BRAM**（URAM 27 Mb ≈ 3.38 MiB + BRAM 11 Mb ≈ 1.37 MiB ≈ 4.75 MiB） | 8 MiB → 片上需缩编 ~59%（见 §5 第 4 条） |
| 时钟 | PL 时钟（MMCM/PLL 生成 1 GHz 或工程时钟）→ `clk_i` | MIG 时钟独立生成，同功能域 |
| 复位 | 板卡复位源 → `async_rst_n_i` | clock_reset 同步释放 |

## 5. P9 立项冻结清单

> 下列项在板卡到位后的 P9 立项会议冻结；当前接口层已把它们全部隔离为参数/接口，冻结不回流到器件逻辑。

1. **物理载体映射**：宿主控制面选 UART（ZCU104）还是 PCIe/以太网；映射到 host_if 寄存器地址空间的桥接规格冻结。
2. **DDR 控制器 IP 适配**：MIG 配置（64-bit、突发长度、时钟比）冻结；把 MIG 实测读/写延迟回填 `DDR_RD_LATENCY` / `DDR_WR_LATENCY`，并与冻结周期模型 `T_FIRST`/`HBM_*_BPC` 重标定口径对齐（D3 已声明「FPGA 以 DDR 替代 HBM」）。
3. **DDR 带宽重标定**：以板卡 DDR4（ZCU104 64-bit）实际带宽重标定 `HBM_READ_BPC/HBM_WRITE_BPC`，更新 qsim timing.py 的 FPGA 口径（不动 RTL 器件逻辑）。
4. **SRAM 缩编口径**：8 MiB → 片上 ≈4.75 MiB（ZCU104，~59%）；缩编后 SRAM 容量/寻址与 16-bank 结构的关系冻结（KV260 备选则 ≈2.88 MiB，缩编更激进）。
   ✅ **缩编 smoke（P9b 后，后续小任务）**：`ddr_if.SRAM_BYTES` 缩编实例——256 KiB/bank（4 MiB）与 128 KiB/bank（2 MiB）读写往返 / 顶边界寻址（$clog2 宽度）/ DDR 隔离 / 宿主口回环全过（`fpga/tb/sram_shrink_tb.sv`）；集成 smoke 以 SRAM_BYTES=4 MiB 重跑，M6 判据（逐指令 trace 一致 + bf16 ≤1 ULP，实测 0.0）ALL PASS，与默认 8 MiB 逐位一致。
5. **PF 引导定义**：PF 由 qsim 侧执行、KV 导入板卡内存，还是板载真跑（二选一，plans §5.2「P9 立项冻结」）。
   ✅ **预裁决（P9b）**：qsim 侧 PF + KV 导入板卡；导入通道 = ZCU104 PS DDR 共享（PS 预写、QCore 经 AXI HP 口直读、非 MIG），4K 起步。见 `docs/p9/pf-boot.md`。
6. **DMA/KV 读路径延迟容忍**：板卡 DDR 有真实延迟，`command_processor` 的组合读（`rd_data`）需改为延迟容忍——在 ddr_if 内实现 64B 写合并行缓冲 + 读 staging，并把 AXI4 主机的 `req_*` 接口接到 DMA/KV 数据通路（当前 co-sim 中该主机空闲，仅单元测试覆盖）。这是板卡适配的核心工作量，立项时确认「CP 不改、只在 ddr_if 侧吸收延迟」还是「DMA 升级为 AXI4 master」。
   ✅ **预裁决（P9b）**：选「CP 不改、只在 ddr_if 侧吸收延迟」；实现 = `fpga/dma_kv_stage.sv`（挂接 ddr_if 引擎字节口内侧，CP 组合读接口保持，在已计费 `T_FIRST` 时点内交付、不双计）。**多 outstanding 实现选择 = stage 自带 AXI 引擎**（N_OUT=4 事务在途 + arid/rid 重排序缓冲，非扩展 ddr_axi4_master）；唯一集成点 = `rd_stall`/`wr_stall` 门控 DMA 字节流。Verilator 单测（随机延迟注入 + 乱序应答 + 写回环）过，见 `fpga/tb/dma_kv_tb.sv`。
7. **板卡/许可确认**：ZCU104（首选，$1,678、8 周、含 Vivado BASIC 一年许可）下单核验；Alinx AXU7EV 为成本替选（购前核验板级 DDR）；KV260 仅预算备选。

## 6. 复现命令

```bash
# lint 全部模块
cd fpga && for f in clock_reset ddr_axi4_master ddr_if host_if; do
  verilator --lint-only -Wno-fatal -Wno-WIDTH --top-module $f $f.sv; done
verilator --lint-only -Wno-fatal -Wno-WIDTH -I. -I../rtl --top-module qcore_fpga_top qcore_fpga_top.sv

# 集成 smoke（编译 + 运行）
cd fpga/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
  --top-module qcore_fpga_top -I../../rtl -I.. ../../fpga/qcore_fpga_top.sv \
  sim_fpga_main.cpp --Mdir obj_dir
python3 run_fpga_smoke.py

# 集成 smoke（SRAM 缩编 256 KiB/bank → 4 MiB；M6 判据）
cd fpga/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
  --top-module qcore_fpga_top -I../../rtl -I.. -GSRAM_BYTES=4194304 \
  ../../fpga/qcore_fpga_top.sv sim_fpga_main.cpp --Mdir obj_dir_shrink
FPGA_SIM=obj_dir_shrink/Vqcore_fpga_top python3 run_fpga_smoke.py

# 单元测试
cd fpga/tb && verilator --cc --exe --build -Wno-fatal -Wno-WIDTH \
  --top-module clock_reset_tb -I.. clock_reset_tb.sv sim_clock_reset.cpp --Mdir obj_cr \
  && ./obj_cr/Vclock_reset_tb
cd fpga/tb && verilator --cc --exe --build -Wno-fatal -Wno-WIDTH \
  --top-module ddr_axi4_tb -I.. ddr_axi4_tb.sv sim_ddr_axi4.cpp --Mdir obj_axi \
  && ./obj_axi/Vddr_axi4_tb

# ddr_if SRAM_BYTES 缩编实例 smoke（P9b 后；256 KiB/bank + 128 KiB/bank）
cd fpga/tb && bash run_sram_shrink.sh
```

## 7. 需评审关注点

1. **AXI4 主机在集成 smoke 中空闲**：co-sim 由功能数组服务（保证 M6 判据），AXI4 主机仅单元测试覆盖。立项时需确认「AXI4 数据通路 + 延迟容忍 DMA」作为板卡适配工作量在 §5 第 6 条内显式立项，而非隐含遗漏。
2. **延迟参数口径**：`DDR_RD/WR_LATENCY` 默认 100 = `T_FIRST`（固定延迟），但板卡 DDR 的持续带宽（B/cyc）与突发效率在 §5 第 3 条重标定——两者是不同的冻结对象，勿混。
