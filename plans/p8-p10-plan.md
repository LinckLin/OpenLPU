# P8+P9+P10 执行计划（提案 v2，并入评审 7 条修订）

> v2：①P10 补 STA（OpenSTA+SDC+多 corner）；②P10 综合用 RTL 快照冻结 + P8 回修触发重 elaborate 门槛；
> ③板卡候选清单前置（ZCU104/Alinx ZU7EV 级首选、KV260 备选注明 SRAM 上限）并立即启动采购并行；
> ④FPGA 宿主闭环（复用 qrun 宿主 + 载体接口 + PF 引导定义，P9 立项冻结）；
> ⑤M7 边界声明内容（援引 D12「RTL/FPGA」+ 传递闭包论证 + 核对 PLAN P8 措辞）；
> ⑥op 类别映射（golden 15 类实例展开，13 合并类显式声明）；⑦P10 SRAM 宏模型来源注明。
> 状态：M0–M6 ✅。P8 与 P10 并行；P9 依赖 P8 + 板卡到位（采购立即启动）。

## 1. 现状与输入

- 三级参照：PyTorch（P1 golden 2540 op 目录 + baseline）、qsim（M4 20/20、M5 达标）、
  RTL（M6 全尺寸 co-sim 4/4 + 单层 14/14）。
- RTL 16-tile PF ≈23 min；全模型 RTL 仿真小时级 → 全模型执行归 P9（FPGA 真实执行）。

## 2. 分工与并行契约

| 节点 | 代理 | 写目录 | 依赖 |
|------|------|--------|------|
| P8 | Golden3 | docs/p8/、rtl/tb/（co-sim 修复权） | 无 |
| P10 | Asic10 | asic/、docs/p10/ | rtl/（只读，启动时打冻结快照 rtl/ref/asicsnap/） |
| P9 | Fpga9 | fpga/、docs/p9/ | P8 + 板卡 |

- **RTL 快照契约**：Asic10 开工即快照 rtl/ 至 rtl/ref/asicsnap/，综合只用快照；
  P8 引发的 rtl/ 源修复须触发 Asic10 重新 elaboration 的门槛（delta 注记更新报告）。

## 3. P8 任务（三级 golden 验证，M7）

1. **逐 op 三级对齐表**：op 实例覆盖 = P1 golden 每层 15 类实例
   （rmsnorm_in/rmsnorm_mlp/attn_qknorm/attn_qkv/attn_rope/attn_score/attn_softmax/attn_ctx/
   attn_o/mlp_gate/mlp_up/mlp_silu/mlp_down/residual_attn/residual_mlp——提案 13 合并类中
   rmsnorm 含 in/mlp 两实例、residual 含 attn/mlp 两实例）；三向 ULP 表（qsim↔PyTorch、
   RTL↔qsim、RTL↔PyTorch）按 15 实例行展开；判据 = M4 口径（|y|≥1e-3 ≤1 ULP，tiny 例外记录）。
2. **逐层三级对齐**：L00（decode cache=1024）层级 hidden 状态三向一致；抽 L00/L13/L27。
3. **端到端边界声明（写入交付条款，内容固定）**：援引 D12 第三级原文「RTL/FPGA」（择一）；
   三级全模型一致由传递闭包成立——M4（PyTorch=qsim 逐 token 一致）+ M8（RTL/FPGA 上板逐 token
   一致）⟹ 三级全模型一致；与 PLAN P8「逐层 + 端到端」措辞核对结论写明。
4. 交付：docs/p8/golden3-report.md。

### 验收（M7）
- 15 实例三向 ULP 表全过（tiny 例外记录）；逐层三向一致；边界声明四项内容完整。

## 4. P10 任务（ASIC 流程，M9）

1. 综合：Yosys + SkyWater 130nm（或工艺库 agnostic）；逻辑部分真综合；
   **SRAM 宏估算注明来源**（OpenRAM 130nm 参考面积/位，或公开密度区间 1.5–4 µm²/bit
   并注明上下界；报告单列 SRAM 占面积/功耗比例与口径）。
2. **STA**：OpenSTA + liberty 多 corner + SDC 约束（asic/ 含 .sdc），报告含时序收敛结论。
3. 产出：面积、功耗（活动因子估算）、频率、token/s 性能评估（结合 M5 周期模型）。
4. 交付：asic/（脚本+约束+.sdc）、docs/p10/asic-report.md。
### 验收（M9）
- elaboration 通过 + STA 结论 + 面积/功耗/频率/性能报告（SRAM 口径注明）。

## 5. P9 任务（FPGA 原型，M8）

1. **板卡候选（立即启动采购，与 P8/P10 并行）**：
   - 首选：ZU7EV 级（ZCU104、Alinx 同级，2–4 GB DDR4 + UltraRAM ≥4 MiB 可容纳缩编 SRAM；
     DDR 容量满足 ≥1.5 GiB 下限）；
   - 备选：Kria KV260（4 GB DDR4，无 UltraRAM、BRAM 5.1 Mb → SRAM 上限 ~0.6 MiB，需更大缩编
     并注明）。
2. 移植：QCore RTL → FPGA（DDR 控制器 IP 适配、时钟域、复位）；**宿主闭环**：复用 qrun 宿主
   （tokenizer/采样/embedding 注入/权重装载），宿主-板卡载体（PCIe/以太网/UART）P9 立项冻结；
   **「PF 引导」定义** = PF 由 qsim 侧执行、KV 导入板卡内存（或板载真跑，二选一在 P9 立项冻结）。
3. 上板跑 0.6B INT8 全模型 decode 生成真实 token。
### 验收（M8）
- 上板生成真实 token，输出与 qsim 逐 token 一致（M4 判据 INT8 口径）；fpga/ 工程可复现。

## 6. 顺序与并行

```
P8（M7）── P9（M8，板卡到位后）
P10（M9）─┘（与 P8 并行；采购与 P8/P10 并行启动）
```

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| 板卡采购周期长 | 候选清单已前置、立即启动；不可得则 P9 挂起不阻塞 P8/P10 |
| P8 回修与 P10 综合竞争 | RTL 快照契约 + 重 elaborate 门槛 |
| 全模型 RTL 仿真小时级 | 已界定：P9 上板为真实执行路径，P8 RTL 侧逐层为界 |
| 三向 ULP 表意外偏差 | 偏差即 bug：定位 op+引擎，回修后复评 |
| 8 MiB SRAM 综合不可行 | 宏估算口径注明来源与上下界 |

## 8. 需评审关注点（第 2 轮）

1. 7 条修订是否全部并入且无新冲突；
2. 若一致，请声明"评审一致，可执行"。
