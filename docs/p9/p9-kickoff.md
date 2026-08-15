# P9 立项预冻结（执行计划冻结）

> 状态：执行计划冻结；仅硬件实测参数待板卡到位后填写（§2 待填参数清单）。
> 依据：`plans/p9-kickoff-plan.md`（批准计划，评审一致）；`docs/p9/porting.md`（接口层与冻结清单）、
> `docs/p9/pf-boot.md`（PF 引导预裁决）、`docs/p9/board-candidates.md`（板卡候选核验）。
> 原则：本文件只做裁决汇总与执行冻结；细节引用原文，不复制。

## 1. 已冻结裁决汇总

| # | 冻结对象 | 裁决 | 依据 |
|---|----------|------|------|
| 1 | 板卡首选 | ZCU104（XCZU7EV）；同级备替 Alinx AXU7EV；预算备选 KV260（SK-KV260-G） | `board-candidates.md` §1/§4 |
| 2 | 宿主闭环 | 复用 qrun 宿主（tokenizer/采样/embedding/权重装载）+ UART 控制面（宿主↔板卡载体 = UART，仅控制面非数据通道） | `porting.md` §5.1/§2.2 |
| 3 | PF 引导 | qsim 侧 PF + KV 导入板卡：PS 预写权重+KV 入 PS DDR，QCore 经 AXI HP 口直读；**QCore 运行时 DDR 窗口与权重/KV 同驻 PS DDR（硬化 DDRC，非 MIG）**，MIG 退为回退 | `pf-boot.md` §1/§2/§7 |
| 4 | DMA/KV 延迟容忍 | `fpga/dma_kv_stage.sv`（CP 不改、ddr_if 侧吸收延迟、多 outstanding + 重排序缓冲） | `porting.md` §5.6 预裁决 |
| 5 | SRAM 缩编 | 参数化；**两档已验证（4 MiB / 2 MiB）**（sram_shrink_tb + 集成 smoke @4 MiB 与默认 8 MiB 逐位一致）；更小档如需再验证 | `porting.md` §5.4/§6 |

> 第 3 条与本汇总第 2 条的边界：UART 只承载控制面（程序/qbin 装载 + logits 回读，带宽低）；
> 权重与 KV 属数据通道，走 PS DDR + AXI HP（UART 数据装载已排除，见 `pf-boot.md` §3）。

## 2. 待填参数清单（板卡到位实测后填写）

每项给出冻结对象（回填位置）与测量方法；实测值填入本表后即为执行口径。

| # | 待填参数 | 冻结对象（回填位置） | 测量方法 |
|---|----------|----------------------|----------|
| 1 | MIG/DDRC 读写延迟 | `ddr_if.DDR_RD_LATENCY / DDR_WR_LATENCY`（与 `qcore_pkg` 的 `T_FIRST` 口径对齐） | 首选测 PS DDRC（HP 口）：经 ddr_if AXI4 主机口（或 dma_kv_stage）发起单 64B 读/写事务，用 Vivado ILA 或片上计数器采样 AXI 通道 req→rsp（valid→valid+last）周期数，多次采样取 p50/p99；若回退 MIG 则同法测 MIG 延迟 |
| 2 | DDR 有效带宽（重标定系数） | 回填 `HBM_READ_BPC / HBM_WRITE_BPC`（qsim `timing.py` FPGA 口径，不动 RTL 器件逻辑） | DMA 全速连续读/写固定窗口 N MiB，片上计数器计周期数 ÷ PL 实测频率 = 持续 GB/s；与 HP 口名义带宽（2.4/4.8 GB/s，`pf-boot.md` §5）对比得重标定系数（可持续 ≠ 峰值） |
| 3 | UART 控制面参数（波特率/流控） | 与 `host_if` 寄存器流匹配（no-wait CSR 的字节→寄存器映射） | 确定宿主↔板卡 UART 链路波特率与流控（XON/XOFF 或硬件 RTS/CTS）；经 host_if 寄存器回环（§3 步骤③）验证映射正确、吞吐满足控制面装载预算 |
| 4 | PL 时钟实测频率 | 决定 tok/s 换算（cycles→秒）与带宽重标定（B/cyc→GB/s） | 板上 MMCM/PLL 生成工程时钟后，逻辑分析仪/频率计测 CLK 引脚频率，或片上计数器 vs 已知参考时钟计数；与 1 GHz 模型约定（1 cyc = 1 ns，`porting.md` §2.1）比对，按 §5.3 口径折算 tok/s |
| 5 | PS/PL 上电顺序与复位时序 | 对上板时钟方案：`clock_reset.async_rst_n_i` 接板卡复位源、`RST_STAGES` 同步释放级数匹配 | 示波器双通道采样 PS 上电复位（PS_POR_B/PS_SRST_B）与 PL 电源轨/时钟/复位，记录上电顺序与复位释放时序；确认同步释放级数与之匹配、无亚稳态窗口 |

> 参数 1/2 是两项不同冻结对象（固定延迟 vs 持续带宽），勿混（`porting.md` §7.2 同口径）。

## 3. bring-up 首日清单（八步，按序执行，每步有判据）

| 步 | 操作 | 判据 |
|----|------|------|
| ① 到货验收 | 上电前目视核验：板卡型号/料号（EK-U1-ZCU104-G）/器件（XCZU7EV）/板上 DDR 容量（2 GB）/Vivado BASIC 许可券/配件（电源、散热、线缆），对照 `board-candidates.md` §1.1；随后上电做 PS 冒烟 | 型号/料号/容量/许可券核对 board-candidates.md §1.1；配件以 AMD 官方套件页（EK-U1-ZCU104-G kit contents）核对；无外观损伤；PS 冒烟：串口有 U-Boot/内核输出（PS 最小系统可启动） |
| ② 电源/时钟/复位冒烟 | 测各电源轨电压（VCCINT/VCCAUX/VCCO 等）；PL 工程时钟起振；复位源释放，`clock_reset` 的 `rst_n` 同步释放 | 电源轨在规格内（DS925）；PL 时钟起振且实测频率记录（回填 §2 参数 4）；`rst_n` 拉高且保持（同步释放级数符合 `RST_STAGES`） |
| ③ host_if 寄存器回环 | 宿主经 UART 桥写 host_if 寄存器（CTRL/STATUS/CMDQ/MEM），回读比对；`CMDQ_GO` 提交 + STATUS 状态翻转 | 写读寄存器值一致（如写 PROG_LEN=0x12345678 回读相同）；`CMDQ_GO` 自增地址正确、STATUS done/running 正确——此为 §2 参数 3 的实测载体 |
| ④ ddr_if 读写往返 | 经宿主口（hd_*）写 DDR 窗口已知 pattern → 读回逐字节比对；同时采样 DDR 延迟 | 写读往返数据逐字节一致；DDR_RD/WR_LATENCY 实测值记录（回填 §2 参数 1） |
| ⑤ 权重装载（0.6 GB） | PS 预写 0.6 GB INT8 权重入 PS DDR（SD/网络/宿主），QCore 经 AXI HP 口抽样校验 | 权重校验和与 qbin 一致；装载时长 ≤ 预算量级（`pf-boot.md` §5） |
| ⑥ KV 导入 | qsim 侧 PF 产出 KV cache 导出 → 导入 PS DDR → QCore 经 HP 直读校验（4K 起步） | KV checksum 一致、导入时长符合预算（`pf-boot.md` §5） |
| ⑦ 单 token decode | 权重+KV 就位后 QCore 跑单 token decode，logits 经 host_if 回读 | 产出 logits 与 qsim 单 token 一致（M4 口径 INT8），无 hang |
| ⑧ 20 token 逐 token 一致（M8 判据） | 完整 decode 会话 20 token，逐 token 与 qsim 对齐 | **M8 判据**：上板生成真实 token，输出与 qsim 逐 token 一致（M4 判据 INT8 口径）；fpga/ 工程可复现（`p8-p10-plan.md` §5 验收） |

> 步骤 ①–④ 为「上板冒烟 + 参数实测」，⑤–⑥ 为数据就位，⑦–⑧ 为 decode 验收；任一步不达判据即停下定位，
> 不回退到器件逻辑（`rtl/` 冻结契约，见 `m8-wait-plan.md` §2）。

## 4. 集成测试驱动接口契约（qrun 宿主 ↔ FPGA）

本驱动是 bring-up 步骤 ③④⑦⑧ 的宿主侧执行器；与 `run_all_acceptance.sh`（组件全量验收）区分。

| 流 | 契约（引用，不复制） | 宿主侧角色 |
|----|----------------------|------------|
| 命令流 | `host_if` CMDQ（PROG_LEN/CMDQ_W0..W3/CMDQ_GO）装载 Q-ISA 程序 + CTRL start（`porting.md` §2.2） | 下发 128-bit 指令、启动/软复位 |
| 日志/回读流 | `host_if` STATUS/TOTAL/TRACE/TRACE_CYC + MEM_RDATA/MEM_ADV logits 回读（`porting.md` §2.2） | 读 done/running、total_cycles、trace、logits |
| 权重流 | 权重+KV 经 PS DDR 预写 + AXI HP 直读（**非** host_if 寄存器；qbin/程序装载走 MEM 寄存器）（`pf-boot.md` §2） | PS 预写权重/KV、QCore HP 直读 |

- 契约要素：命令队列装载协议、logits 回读协议、权重/KV 装载通道三者的字节序/地址/握手，均以 `porting.md` §2 与 `pf-boot.md` §2 为准；本驱动不新增接口，只实现已冻结的宿主侧协议。
- 观测判据统一 = M6 口径（逐指令周期 trace 精确相等 + 最终内存 bf16 ≤1 ULP，`|y|≥1e-3` 元素）。

## 5. 风险与回退

| 风险 | 回退 |
|------|------|
| KV260 备选 SRAM 上限 ≈2.88 MiB（`board-candidates.md` §1.3） | 缩编档更激进：2 MiB 档（=16 bank × 128 KiB/bank）已逼近上限；若仍不敷则更低档（需再验证，见 §1 第 5 条）；演示 ctx 上限相应下调；首选仍为 ZCU104（≈4.75 MiB） |
| DDR 带宽低于预算（HP 口实测可持续 < 名义 2.4/4.8 GB/s） | 降档：150 MHz→300 MHz→多 HP 口并行→回退板卡独立 DDR4 MIG（`porting.md` §4）；降档后果 = 更长导入时间 / 更小演示 ctx（`pf-boot.md` §7 同口径） |
| PL 时钟实测 < 1 GHz（模型约定） | tok/s 重算口径：`tok/s = f_实测 / cycles_token`（1 GHz 时 `tok/s = 1e9 / cycles_token`，M5 报告同式），按实测频率线性缩放，回填 qsim `timing.py` FPGA 口径；不伪造 1 GHz |
| 步骤④ 实测延迟与 `T_FIRST` 默认值（100）不符 | 回填 §2 参数 1 到 `DDR_RD/WR_LATENCY`，与冻结周期模型 `T_FIRST` 重标定口径对齐；`dma_kv_stage` 在已计费时点内交付、不双计（`porting.md` §5.6） |

## 6. 偏离已批准计划的处理

本文件严格按 `plans/p9-kickoff-plan.md` §1 五项内容与口径落盘；若后续执行发现需偏离
（例如板卡到位实测推翻某待填参数口径、或 bring-up 步骤需增删），停下列为需评审项，不得静默改动。
