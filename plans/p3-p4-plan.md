# P3+P4 并行执行计划（提案 v2，回应评审第 1 轮 5 条意见）

> v2 变更：①M3 验收回退 PLAN 口径（qbin 可加载 + 逐层量化误差；全模型 forward 归 P5/M4）；
> ②P3 增补 Vector 引擎时序模型与分解桶；③M2b 偏差判据按桶定义（KV 桶对 roofline §6 口径）；
> ④GATHER vs LOAD 裁决改为两者每层周期直接对比；⑤embedding host 侧边界记为 D14 裁决（spec §3.3）。
> 状态：M0 ✅、M1 ✅、M2a ✅（均已评审通过）。本计划交付 M2b（P3）与 M3（P4）。

## 1. 背景与前提

- P1 交付：golden/qwen3-0.6b/ 全量（2540 op 目录，decode cache 0/512/1024/2048/4096 抽样）、
  docs/p1/roofline.md（prefill 63.5 µs 计算受限【纯矩阵锚点】/ decode 每层权重读 21.85 µs
  HBM 受限、KV 窗口重读 8K 时 0.94 GB/token）。
- P2 交付（评审通过）：qsim/executor.py（功能级：SYS/DMA/GEMM/GEMV/BMM + dequant）、
  compiler/{isa,lowering,qbin}.py、MLIR 21.1.8 工具链。
- 冻结口径：sustained 720/240 GB/s、SRAM 读 512/写 256 B/cyc、16 bank 2R1W、
  T_first=100+align+Q（03 §3.3）、MODE 切换 300 cycle、KV 重读修正（04 §2.4 修正注）。
- **执行边界（本计划冻结）**：executor 的 VECTOR/KV 功能语义、attention/softmax/norm/rope
  的程序段生成、全模型 forward 数值验证**全部归 P5（M4）**。P3 时序模型覆盖 VECTOR 的
  时序行为（按 04 §3.2/§3.4 建模型），不改 executor 功能语义。P4 产出**全模型 GEMM/GEMV
  骨架 qbin**（28 层 × 6 类线性投影【qkv 融合/o/gate/up/down】+ lm_head + DMA 搬移），不含 VECTOR/KV 程序段。

## 2. 分工与文件布局

| 路 | 写目录 | 只读 |
|----|--------|------|
| SimP3 | qsim/（新增 timing 部分）、docs/p3/ | compiler/、golden/、spec |
| ForgeP4 | qforge/、docs/p4/ | qsim/（功能级执行器）、compiler/、golden/、spec |

禁止互写。qsim/executor.py 数值语义冻结（P2 评审），SimP3 只增 timing 层不改语义；
如需改动须列需评审项。ForgeP4 复用 compiler/isa、qbin、lowering（只读导入）。

## 3. P3 计划（qsim 时序模拟器，M2b）

### 任务
1. `qsim/timing.py`：在 executor 之上加 cycle 计数，四引擎模型：
   - **Matrix**：PF 整阵列（周期 ≈ ceil(K/256)×M+256，04 §1.4）；DC 16 lane×8 行 GEMV
     （每批 256 cycle，04 §2.2）；MODE 切换 300 cycle（BARRIER 后）。
   - **Vector**：按 04 §3.2 指令吞吐/延迟表 + §3.4 复合序列（softmax/RMSNorm/RoPE/SwiGLU）
     建模；0.6B 重算（softmax 行数 16 head、RMSNorm len 1024、RoPE 元素 24×128、SwiGLU 3072×2）；
     分解桶含 Vector/norm 项。此为 spec §5.3 第 3 项（Vector cycle 精化）的裁决载体。
   - **SRAM**：16 bank 2R1W、bank=addr[7:4]（16B 粒度 16 路交错）、03 §2.3 固定优先级 stall；读 512/写 256 B/cyc。
   - **HBM**：sustained 读 720/写 240 GB/s；T_first=100+align+Q；64B 突发；KV 全窗口重读
     按 05 §5.2 口径（0.6B：114,688×ctx B/token）。
   - **DMA**：4 in-flight、双缓冲；bank 仲裁与 in-flight 数按 03 §2.3/§4.4 实现即裁决
     （spec §5.3 第 2 项）。
2. **trace 重放**：decode 单 token（cache=1024 与 4096）与 prefill seq=128 的层指令序列
   （按 02 §12 步序列构造，含 VECTOR 复合 op 时序），输出每层 cycle 分解：
   权重流 / KV 重读 / 计算（矩阵）/ Vector / 依赖 stall。
3. **数字核验（按桶）**：
   - 权重流桶 vs roofline §5.2 的 21.85 µs/层（sustained），偏差 <10%；
   - 矩阵计算桶 vs 0.48 µs/层（INT8 峰值）；
   - KV 重读桶 vs roofline §6 / 05 §5.2 HBM 读口径（单副本 = 114,688×ctx / 720 GB/s）；
     **GATHER ×4 的额外 SRAM 写开销是待冻结项裁决证据，不是 bug**；
   - prefill 每层 = 矩阵 63.5 µs + Vector（0.6B 重算 ~30-40K cycle，串行假设）→ 全层 ≈ 95 µs，
     与 63.5 µs 纯矩阵锚点的差异在偏差表中明示（锚点口径声明）。
4. **待冻结项裁决**（spec §5.1，P3 权威产出）：**直接对比 GATHER 与 LOAD 每层周期**——
   两案 HBM 窗口读相同（1×）；差异 = SRAM 写 4× vs 1× 与 tile 上限 512 vs 2048（分块次数）。
   0.6B GQA 2:1 下 broadcast ×4 含 2 份冗余副本（05 §1.5 注）。输出 v0 裁决与量化依据
   （权重读 21.85 µs 仅作参考上下文，不作判定线）。
5. `docs/p3/sim-report.md`：模型描述、trace 重放、按桶偏差表、待冻结项裁决。

### 验收（M2b）
- layer trace 在时序模型下跑通，数值与功能级一致（同一 executor）；
- 按桶偏差判据全部通过（权重流 <10%、计算 <10%、KV 桶与 §6 口径一致）；
- 输出 0.6B decode token/s（含 KV 重读，4K/8K）与 prefill 每层全周期（含 Vector）；
- 待冻结项裁决给出 GATHER vs LOAD 的每层周期直接对比与依据。

## 4. P4 计划（qforge 前端，M3）

### 任务
1. `qforge/cli.py`：`qforge compile Qwen/Qwen3-0.6B --target qcore-v1 --dtype int8
   [-o qwen3-0.6b.qbin]`——config 解析 → 建图 → safetensors 加载 → W8A8 量化
   （对称 per-128-K-group，权重侧；激活 scale 编译期常数）→ 权重重排打包 → tiling
   （N≤128、K 流式）→ 调度 → .qbin（00-container 完整容器）。
2. **MLIR pass 落地（linear 链）**：C++ 实现 qnn→qisa lowering pass
   （`mlir-opt --qnn-to-qisa`），覆盖 qnn.matmul/rmsnorm/linear 三 op 的机制证明；
   全模型 GEMM/GEMV 序列本节点由 Python lowering 生成（复用 compiler/lowering.py），
   attention/rope/swiglu 的 pass 归 P5。
3. **0.6B 全模型 GEMM/GEMV 骨架 qbin**：28 层 × (QKV 4096×1024 / O 1024×2048 /
   gate/up 3072×1024 / down 1024×3072) + lm_head 151936×1024 的 PF 程序（GEMM）与
   DC 程序（GEMV）+ DMA 搬移段。**KV 程序段生成与 GATHER/LOAD 裁决落地归 P5（M4）**：
   本节点骨架不含 KV 段，P3 裁决写入 sim-report 作为 P5 输入。
   **embedding 查表在 host**（D14：host 读 embedding 表 → hidden 经 DMA.LOAD 注入 SRAM；设备永不读 embedding 表）；采样在 host（D9）。
   qbin 输入 = hidden 向量、输出 = logits。
4. **M3 验证（PLAN 口径，不缩水不加戏）**：qbin 由 qsim 功能级执行——
   (a) 加载器完整解析（header/tensors/.pf/.dc/ENDQ + 权重 round-trip，M2a 同款断言）；
   (b) **逐层量化误差**：golden linear 输入（`linear_wq_pf/dc` 扩到 6 类线性投影：
   qkv(融合)/o/gate/up/down/lm_head，PF+DC 各一）→ W8A8 qbin 执行 vs golden y_ref，
   判据沿用 M2a（INT32 逐位 + dequant fp32 <1e-6 于 per-128-group 量化数据）；
   (c) 全模型 greedy/逐 token 一致**不属本节点**，归 M4（P5）。
5. `docs/p4/qforge.md` + 量化误差报告（6 类投影 × PF/DC）。

### 验收（M3）
- `qforge compile Qwen/Qwen3-0.6B --dtype int8` 产出 qbin，qsim 完整加载执行；
- 6 类线性投影 PF+DC 逐层量化误差达标（M2a 判据）；
- qnn→qisa MLIR pass 经 mlir-opt 跑通（linear 链机制证明）。

## 5. 会师与依赖

```
P1 ✅ ─┬─ SimP3（M2b）──┐
P2 ✅ ─┴─ ForgeP4（M3）─┴→ P5 qrun+QMetal（M4：全模型 forward + VECTOR/KV 语义 + greedy 一致）
```
- SimP3 与 ForgeP4 互不依赖；ForgeP4 复用 executor 功能级自验，不等 SimP3。
- 待冻结项：SimP3 出裁决并写入 sim-report；裁决与 KV 程序段生成的落地归 P5（M4），
  本计划不产 KV 程序段（P4/P3 无此耦合，互不依赖成立）。
- 会师判定：M2b = SimP3 报告（含待冻结项裁决）；M3 = qforge qbin + 逐层量化误差达标。

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| 时序模型与 roofline 偏差超限 | 按桶定义判据（权重流/计算 <10%；KV 桶对 §6 口径；Vector 桶对 04 §3.4 重算）；超限即按桶审计，出偏差表 |
| MLIR pass C++ 周期长 | linear 链机制证明优先；全模型程序本节点走 Python lowering，其余 op pass 归 P5 |
| W8A8 量化误差超预算 | per-128-group 为保守选择；超预算上报评审（回退 per-64-group 等） |
| executor 语义漂移 | 功能语义冻结（P2 评审）；P3 只加 timing；VECTOR/KV 功能语义归 P5 一次性实现并评审 |
| KV 待冻结项并行竞争 | 无——KV 程序段归 P5；SimP3 裁决写入 sim-report 作为 P5 输入 |
| GATHER ×4 冗余在 0.6B 2:1 下被误判为 bug | 偏差表显式标注：冗余 2 副本是裁决证据（05 §1.5 注：P6 优化项） |

## 7. 需评审关注点（第 2 轮）

1. 5 条评审意见是否全部落实且无新引入冲突；
2. M3 回退 PLAN 口径后，P4/P5 边界（全模型 forward 归 P5）是否与 PLAN §4 P5 验收一致；
3. D14（embedding host 侧）与 spec 03 §3.2/05 §5.1 措辞的兼容性声明是否足够；
4. 若一致，请声明"评审一致，可执行"。
