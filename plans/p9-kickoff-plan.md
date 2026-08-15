# P9 立项预冻结 + 全量验收脚本（提案 v2，回应评审第 1 轮 7 条意见）

> v2 变更：①验收清单补 verify_m3/m4/run_mac_synth；②--quick 跳过集与 SKIP 标注定义；
> ③待填参数补 UART/PL 时钟实测/PS-PL 上电时序；④bring-up 加到货验收步；⑤qsim 三文件
> 显式执行方式（test_m2a 非 pytest 用例单独跑）；⑥SRAM 两档（4/2 MiB，已验证）；⑦ASIC 段
> lint/synth/STA 探测统一定义。
> 状态：27/28 完成，M8 阻塞于板卡采购。

## 1. P9 立项预冻结（docs/p9/p9-kickoff.md）

内容（执行计划冻结，仅硬件实测参数待填）：
1. **已冻结裁决汇总**：板卡 ZCU104 首选（Alinx AXU7EV 备选、KV260 预算备选）；宿主闭环 =
   复用 qrun 宿主 + UART 控制面；PF 引导 = qsim 侧 PF + KV 导入（PS 预写 PS DDR、QCore 经
   AXI HP 直读）；DMA/KV 延迟容忍 = fpga/dma_kv_stage.sv；SRAM 缩编 = 参数化，**两档已验证
   （4 MiB / 2 MiB）**，更小档如需再验证。
2. **待填参数清单**（板卡到位实测后填写，每项含测量方法）：
   MIG/DDRC 读写延迟（覆盖 ddr_if 的 DDR_RD/WR_LATENCY）、DDR 有效带宽（vs 时序模型口径
   的重标定系数）、**UART 控制面参数（波特率/流控，与 host_if 寄存器流匹配）**、
   **PL 时钟实测频率（1 GHz 为模型约定，ZU7EV 实测频率决定 tok/s 换算与带宽重标定）**、
   **PS/PL 上电顺序与复位时序（对上板时钟方案）**。
3. **bring-up 首日清单**（按序，每步验收判据）：
   ①到货验收（上电前：板卡型号/料号/DDR 容量/PS 冒烟/许可券与配件，对照 board-candidates.md
   §1.1）→ ②电源/时钟/复位冒烟 → ③host_if 寄存器回环 → ④ddr_if 读写往返（填 DDR 延迟实测）
   → ⑤权重装载（0.6 GB，时长预算）→ ⑥KV 导入 → ⑦单 token decode → ⑧20 token 与 qsim
   逐 token 一致（M8 判据）。
4. **验收脚本**：qrun 宿主 ↔ FPGA 的集成测试驱动（接口契约：命令/日志回读/权重流）。
5. **风险与回退**：KV260 备选的 SRAM 上限（2.88 MiB）对应缩编档与演示 ctx 上限；DDR 带宽
   低于预期时的降档表；PL 时钟实测低于预期时的 tok/s 重算口径。

## 2. 全量验收脚本（run_all_acceptance.sh）

内容：
1. 组件清单（对照 README 8 组件索引，逐条覆盖）：
   - qsim：`python3 -m pytest qsim/test_isa_fields.py qsim/test_vector_kv.py -q` +
     **`python3 qsim/test_m2a.py`（显式单跑，非 pytest 用例，4 case 判据）**；
   - M5：`python3 qsim/timing_p6.py`（双 target PASS）；
   - compiler：**`python3 qforge/verify_m3.py`（12 例 = 6 类 × PF/DC + 结构）**；
   - runtime：**m4 默认证据文件校验（docs/p5/m4-report.md 存在且 BF16 四判据 PASS 行在场）+
     `--full` 显式全量重跑（小时级）**；
   - 三级 golden：`python3 rtl/tb/run_golden3.py`；
   - RTL：`python3 rtl/tb/run_cosim.py`；
   - FPGA：`python3 fpga/tb/run_fpga_smoke.py` + sram 缩编（`asic/run_sram_check.sh`）；
   - ASIC：**verilator --lint-only（rtl/ 顶层）+ `bash asic/run_synth.sh` + OpenSTA（三者均
     工具链探测：在位则跑、缺失则 SKIP(工具缺失) 并注明）**；`bash asic/run_mac_synth.sh`。
2. **--quick 模式跳过集**：golden3、co-sim、m4 全量（保留 qsim 三件套 + timing_p6 +
   verify_m3 + fpga smoke + sram 缩编）；汇总区分 `SKIP(quick)` 与 `SKIP(工具缺失)`。
3. 输出汇总表：组件 / 命令 / 结果 / 耗时；任一 FAIL 则 exit 1。
4. README 组件索引表增一行入口；reproduction.md §0 增「一键验收」。

## 3. 分工

| 项 | 代理 | 写目录 |
|----|------|--------|
| P9 立项预冻结 | Kickoff9 | docs/p9/p9-kickoff.md |
| 全量验收脚本 | AcceptAll | run_all_acceptance.sh、README.md、docs/reproduction.md |

## 4. 验收

- p9-kickoff.md：五项内容齐全，待填参数（含 UART/PL 时钟/上电时序）有测量方法，
  bring-up 含到货验收步、每步有判据。
- run_all_acceptance.sh：实测执行一遍全绿（ASIC 段按工具链在位性自动判定；--quick 模式
  定义与汇总标注正确），README/reproduction 更新后抽验入口。

## 5. 风险与对策

| 风险 | 对策 |
|------|------|
| 全量总时长超长（golden3 ~8 min + co-sim ~30 min + m4 全量小时级） | m4 默认证据校验；--quick 定义明确；默认全量含 golden3/co-sim |
| ASIC 段依赖 Yosys/OpenSTA | 工具链探测 + SKIP(工具缺失) 注记，不伪造 PASS |
| p9-kickoff 与 porting.md 重复 | 只做裁决汇总与执行冻结，引用原文不复制 |
