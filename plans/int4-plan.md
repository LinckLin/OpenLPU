# INT4 全链路计划（提案 v2，回应评审第 1 轮 6 条意见）

> v2 变更：①qsim/ 划入 Q4b（executor W4A16 GEMM 路径为首个增量），Q4c 对「INT4 会师后的
> qsim 基线」只读；②Q4a 判据改 M3 式双轨（同 scale fp64 参考 dequant <1e-6 实现正确性 +
> 量化误差如实报告；≤1 ULP 不用于 golden 对比；token ≥8/10 硬门槛）；③打包位序权威 =
> qsim/executor.py（偶元素低半字节），双向往返锁定测试 + Q4b 合成驱动复用 Q4a 打包器；
> ④W4A8 RTL 移出本期（backlog），65.54 维持 spec 级记录；⑤Q4b 增 INT4 decode 天花板表
> （8B 190.5 sustained / 0.6B ≈2417 短 ctx / ≈938 @4K / ≈582 @8K）；⑥Q4b 删运行时
> QUANT/DEQUANT INT4（W4A16 权重预量化、激活保持 BF16，无消费者）。
> 定位：第一代冻结范围（D5：BF16→INT8→INT4）收尾项。

## 1. 范围决策

- **v0 部署路径 = W4A16**（INT4 权重 + BF16 激活，per-128-group，spec §3.1 裁决 2）。
  W4A8（INT8 激活）为硬件支持的可选路径，**本期不做 RTL 实现（backlog）**；65.54 TMAC/s
  维持 spec 级记录（04 §1.2）。
- 判据口径：实现正确性 = 同 scale fp64 参考 dequant <1e-6（M3 式）；量化误差 = 如实报告；
  token 交叉一致率 ≥8/10（硬门槛，M4 INT8 式）；co-sim ≤1 ULP（同一量化计算的两实现间）。

## 2. 分工（三路并行 + 会师）

| 流 | 代理 | 写目录 | 依赖 |
|----|------|--------|------|
| QuantInt4 | Q4a | qforge/、docs/p4/ | 无 |
| RunInt4 | Q4b | **qsim/（executor W4A16 GEMM 路径，首个增量）**、qrun/、docs/p5/ | Q4a 的 qbin 产物（先合成后接真） |
| RtlInt4 | Q4c | rtl/、docs/p7/ | **对 INT4 会师后的 qsim 基线只读**（快照契约延续） |

- RtlInt4 改 rtl/ 源 → 默认参数全回归（golden3/co-sim/单层）逐周期不变；INT4 为新增通路，
  专用用例验证，不进默认回归集。
- **打包位序契约**：权威 = qsim/executor.py 的 _read_vector/_write_vector（偶元素 → 低半
  字节、奇元素 → 高半字节）。锁定测试：①qforge 打包 → executor 解包 == 原矩阵；②executor
  打包 → RTL 解包 == 原矩阵（Q4c co-sim 内）。Q4b 合成驱动复用 Q4a 的打包器 API，禁止手写。

## 3. 任务

### Q4a（qforge INT4 量化）
1. qforge/quant.py 增 W4A16：权重 INT4 对称 per-128-group（round-to-nearest，v0 不做二阶
   补偿），2b 打包（位序按 executor 权威）；qbin tensors 表 dtype=INT4。
2. lowering/program 适配：GEMM srcB=INT4、srcA=BF16、acc=FP32、dequant=1。
3. 验证（M3 式双轨）：(a) 同 scale fp64 参考 dequant <1e-6（实现正确性）；(b) 6 类投影 ×
   PF/DC 量化误差 vs golden **如实报告**（预期 8-15% rel 量级，以报告为准，不设 ≤1 ULP）；
   与 INT8 同口径对比表。
4. 交付：docs/p4/quant-error-report.md 增 INT4 节 + qbin 产物。

### Q4b（qsim/qrun INT4 全模型）
1. **executor W4A16 GEMM 路径（首个增量）**：_matrix 的 srcB=INT4+srcA=BF16+acc=FP32+
   dequant=1 走 fp32 组内累加（02 §6/04 §1.2 口径；修复现行 int8_group_partials 对 BF16
   激活的 astype(int32) 截断路径）；配套单元测试。
2. qrun --dtype int4：权重装载（2b 解包，复用 Q4a 打包器）+ GEMM dequant per-128-group
   （CD 描述符复用）。**无运行时 QUANT/DEQUANT INT4**（权重预量化、激活保持 BF16）。
3. 全模型 decode：20 token 交叉一致率 vs BF16 baseline ≥8/10 + 分歧位置 logits rel 误差；
   **INT4 decode 天花板表**（HBM-bound 权重流口径：8B 317 peak/190.5 sustained；0.6B
   ≈2417 短 ctx / ≈938 @4K / ≈582 @8K sustained，KV 重读按 INT8 修正注同式）。
4. 交付：docs/p5/m4-report.md 增 INT4 节。

### Q4c（RTL W4A16 数据通路）
1. matrix_engine.sv 仅 W4A16：INT4 权重解包 + BF16 尾数路径（对齐 04 §1.2），dequant 复用
   现有后处理。**W4A8 不做（backlog）**。
2. 专用 co-sim 用例（INT4 GEMM/GEMV vs INT4 会师后的 qsim 基线 ≤1 ULP + trace）+ 位序
   往返锁定测试（executor 打包 → RTL 解包 == 原矩阵）；默认回归不破坏。
3. 交付：docs/p7/rtl-report.md 增 INT4 节（含 04 §1.2 W4A16 同量级口径复核记录）。

## 4. 验收

- Q4a：6 类 × PF/DC INT4 双轨验证报告（fp64 参考 <1e-6 + 量化误差如实）。
- Q4b：INT4 全模型 20 token 交叉一致率 ≥8/10 + rel 误差 + 天花板表。
- Q4c：INT4 专用 co-sim ≤1 ULP；位序锁定测试过；默认全回归逐周期不变。

## 5. 风险与对策

| 风险 | 对策 |
|------|------|
| INT4 量化误差超预算（8-15% rel 可能致 token 一致率 <8/10） | 如实报告；回退方案 = AWQ 式 per-group 搜索（Q4a 增补），不虚报 |
| executor W4A16 通路改动破坏 INT8/BF16 既有路径 | 全量回归（41 测试 + M2a + verify_m3）门槛 |
| 打包位序漂移 | 双向往返锁定测试 + 权威单一（executor） |
| Q4b 等 Q4a 产物阻塞 | 合成权重预跑驱动（复用 Q4a 打包器），Q4a 交付后回归 |
