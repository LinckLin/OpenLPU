# P6+P7 并行执行计划（提案 v1，待评审）

> 状态：M0–M4 ✅（全部经评审一致）。本计划：P6（硬件感知优化，M5）与 P7（QCore RTL，M6）
> 双线并行——软件优化线与硬件线首次同时推进。
> 依据：PLAN.md §4 P6/P7；spec（02 权威 ISA、03 内存、04 执行引擎、05 KV）；D16（KV.LOAD 冻结
> + P7 内部广播总线）；P5 移交（INT8 per-tensor 激活 scale 欠拟合归 P6 量化映射；488 @4K 公式
> 待 P6 验证；P7 立项确认项：RTL 接受 BMM batch_stride_B=0 或 qbin 重生成）。

## 1. 现状与输入

- qsim：功能级（VECTOR/KV 全语义）+ 时序级（timing.py，四引擎模型，GATHER/LOAD 已裁决）。
- qrun/QMetal：全模型端到端（BF16 M4 达成：20/20、8/8@1024、5/5@4K、8K argmax 一致）。
- INT8：2/10（per-tensor 激活 scale 欠拟合，rel 0.33–0.72；M4 已归 P6）。
- qforge：全模型 PF/DC 程序（与 qrun 逐指令一致），qbin 完整。
- roofline 锚点：0.6B decode 4K ≈ 675（HBM 读口径）/ ≈ 488（SRAM 写口径，LOAD 路径）；
  8K ≈ 469。M5 目标 = sustained roofline 的 80%（周期模型口径，非 Python 执行速度）。
- 硬件线输入：spec 六分册（ISA/内存/引擎/KV 全部冻结）；qsim 为逐指令参照。

## 2. 分工与文件布局

| 路 | 代理 | 写目录 | 只读 |
|----|------|--------|------|
| 软件 | OptP6 | qsim/（时序扩展，见基线冻结）、qrun/、docs/p6/ | spec、golden、qforge |
| 硬件 | RtlP7 | rtl/（SystemVerilog）、docs/p7/ | spec 六分册、qsim（参照） |

禁止互写。RtlP7 不依赖 OptP6 产物（spec + qsim 已足够）；OptP6 不依赖 RtlP7。
会师点：P8（三级 golden 验证）依赖两者。

## 3. P6 任务（硬件感知优化，M5）

1. **INT8 量化映射优化**（P5 移交）：
   - 激活量化改 **per-128-group**（对齐权重侧 group 结构），校准数据 = golden 各投影输入；
   - 备选：逐 token 动态校准（decode 每步重算 sx，量化开销计入周期模型）。
   - 验收：INT8 与 BF16 baseline 交叉一致率 ≥8/10（10 token 同 prompt）+ 分歧位置 rel 误差报告；
     达不到则上报评审（per-64-group 或 INT8 动态范围调整）。
2. **调度优化（qsim 时序模型验证周期）**：
   - KV staging 与 dense 权重流重叠（488 @4K 公式假设的验证）；
   - 双缓冲（权重 tile、KV tile）与 DMA PREFETCH 排布；
   - PF/DC 程序在 qsim timing 上重放 4K/8K decode 单 token 全层 + prefill 128 block，
     周期分解表 vs roofline。
   - KV 重读消减（streaming/选择性重读）——若实现，消融单列。
3. **M5 验收（统一写口径，时钟 1 GHz = 1 cyc/ns）**：decode 单 token 全层周期（qsim 时序，
   KV.LOAD 路径、含 KV 重读与 KV staging 写）达到 sustained 写口径 roofline 的 80%：
   **4K：每 token 周期 ≤ 2.56M cycles（= 488×0.8 ≈ 390 token/s）；8K：每 token 周期 ≤
   4.86M cycles（= 257×0.8 ≈ 206 token/s；8K 写口径 = 1e9/(28×131,088+216,087) ≈ 257）**。
   选择写口径的理由：比读口径的 80% 目标（4K 540/8K 375；读口径本身 675/469）更严，
   且 KV staging SRAM 写是当前真实瓶颈。
   消融表（重叠/双缓冲/重读消减 各自贡献）；INT8 一致率 ≥8/10。

> **qsim 基线冻结（P6/P7 并行契约）**：P7 开工时对 qsim 打基线快照（git tag 或文件快照），
> RtlP7 的 co-sim 周期/数值对齐一律以基线为准；OptP6 的时序扩展为**新增可选模型**
> （不改既有指令的周期口径），扩展合入需评审确认后由主会话同步 RtlP7。
4. 交付：docs/p6/opt-report.md（周期分解、消融、INT8 量化方案与误差）。

## 4. P7 任务（QCore RTL，M6）

1. **立项约定（实现级冻结，不改 spec）**：单时钟域 1 GHz（1 cyc = 1 ns）；异步复位低有效、
   同步释放；SRAM 16 bank 2R1W 端口 + 固定优先级仲裁接口；HBM 内存模型接口（64B 突发、
   延迟参数化）；Verilator 版本锁定（写入 rtl-report）。
   - Matrix Engine：128×128 dual-MAC 阵列，weight-stationary；PF 整块 GEMM / DC 16 lane×8 行
     GEMV；dequant 后处理（per-128-group scale、fp32 组间累加、BF16 落盘，04 §1.5）；
     **DC GQA 内部广播总线**（D16：K/V 组内共享；B 侧地址派生规则在此定义并回注 04 §2.2——
     支持 batch_stride_B=0 编码，P5 评审确认项）。
   - Vector Engine：128-lane，VECTOR 18 条（延迟/吞吐对齐 04 §3.2；EXP/RSQRT LUT+NR）。
   - DMA Engine：2D、4 in-flight、64B 突发。
   - Command Processor：128-bit 取指/解码/发射、三引擎队列（Matrix/Vector/DMA；KV 指令
     经 DMA 队列 + 地址生成器执行）、BARRIER/WAIT。
   - **KV 数据通路 = DMA 引擎 + KV 地址生成器**（05 §2 口径，不设独立 KV 引擎）：
     KV.LOAD/STORE_BLOCK/APPEND 由 DMA 传输执行、KV 地址由 05 §1.3 公式派生（C30 pos 耦合、
     BARRIER 后可见性）；**KV.GATHER 实现 ×4 副本写通路**（D16 保留于 ISA，co-sim 覆盖全 33 条）。
   - Scratchpad SRAM：16 bank × 512 KiB（交错寻址 B[7:4]），2R1W + 固定优先级仲裁。
3. **单 layer golden 跑通**：P1 golden 的一层 trace（attn_qkv→…→residual_mlp）在 RTL 上
   数值与 golden 一致。
4. 交付：rtl/（分层模块 + testbench）、docs/p7/rtl-report.md（结构、co-sim 结果、
   广播总线地址派生规则、需评审项）。

### 验收（M6，显式全尺寸）
- **全尺寸 128×128 RTL** co-sim 与 qsim 基线逐指令一致（周期 + 数值，bf16 ≤1 ULP）；
  **M4 口径（与 qsim 基线 test_m2a.py 同构）**：`|y|≥1e-3` 元素 ≤1 ULP，近零消去
  （`|y|<1e-3`，tiny）元素例外——fp32 累加序跨实现差异（numpy/torch），非 RTL 缺陷，
  tiny 占比 ~0.5%；
  缩编实例（如 32×32）仅作开发路径、保持接口一致，**不得作为验收证据**；
- 单 layer golden 跑通；广播总线规则已回注 04 §2.2。

## 5. 里程碑与后续

- M5 = P6 三条验收全过；M6 = P7 验收全过。
- M5/M6 后 → P8（三级 golden 会师：PyTorch=qsim=RTL）+ P9 FPGA 立项。

## 6. 风险与对策
（v2 修订：M5 写口径 4K/8K 数值目标、qsim 基线冻结、KV 数据通路归属 + GATHER 实现、
M6 全尺寸显式、per-128-group 激活 scale 走 qrun 运行时路径（QUANT per-128-group 模式，
ISA/executor 已支持，周期开销计入周期模型，不动 qforge）、RTL 立项约定。）

| 风险 | 对策 |
|------|------|
| RTL 128×128 阵列 Verilator 仿真慢 | 先用缩编实例（如 32×32）验证控制/数据通路，再全尺寸回归；缩编须注明并保持接口一致 |
| 广播总线地址派生与 BMM 语义分歧 | P7 在 04 §2.2 定义后回注，02 §6.1 加 DC 注（P5 评审已预告），经评审确认 |
| INT8 一致率仍不达标 | per-64-group 激活量化 / 混合精度（attention BF16 + MLP INT8）上报评审 |
| 488 公式假设（重叠）在时序模型上不成立 | 消融定位瓶颈（SRAM 写口/DMA in-flight），修正公式并记录 |
| RTL 与 qsim 漂移 | co-sim 断言逐指令；qsim 为唯一数值参照（D12 三级 golden 原则） |

## 7. 需评审关注点

1. M5 的 80% 口径定义（488×0.8≈390 token/s 等价周期）是否可验证；
2. P7 缩编先行策略是否构成验收缩水（M6 要求全尺寸回归还是缩编通过即可——建议全尺寸
   co-sim 为验收、缩编仅作开发路径）；
3. 广播总线地址派生规则归属（04 §2.2 回注）与 batch_stride_B=0 的契约；
4. INT8 一致率 ≥8/10 的备选升级路径是否完整。
