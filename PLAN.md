# QCore — LLM 推理个人级开源加速平台 · 工作计划

> **项目定义**：设计并开源一套面向 Transformer/LLM 推理的个人级完整 AI 加速平台。
> 参考 Tenstorrent TT-Forge 的软硬件分层思想，编译器、ISA、Runtime、RTL 全部独立设计，
> 实现 Hugging Face 的 Qwen 等模型"一键编译并部署"到自己的加速器。
>
> **追求**：全链路完整、真正可运行、完全开源、个人可复现。不以绝对性能对标商业芯片，
> 以**体系完整度**对标 Tenstorrent。要证明的不是"我会设计 GEMM 加速器"，而是：
> **"下载真实 Qwen → 自己设计的编译器 → 自己设计的机器指令 → 自己写的 SystemVerilog
> 处理器真正执行 → 生成正确的自然语言。"**
> 发布形式：完整开源项目 + 系列知乎文章。

```bash
qforge compile Qwen/Qwen3-0.6B --target qcore-v1 --dtype int8   # → qwen3-0.6b.qbin
qrun qwen3-0.6b.qbin                                            # → 逐 token 生成
```

```text
Hugging Face / PyTorch → Model Frontend → Q-MLIR → QNN/LLM IR
→ Hardware-aware Compiler → Q-ISA → qsim (ISA Simulator)
→ QMetal Runtime → QCore RTL → FPGA Prototype → ASIC Synthesis
```

## 0. 命名对照（正式，旧名废弃）

| 名称 | 含义 | 旧名（废弃） |
|------|------|--------------|
| QCore | 芯片/平台（单核 Matrix/Vector/DMA/SRAM/CP + KV/DDR/HBM 访存） | newlpu |
| Q-ISA | Tensor Command ISA（33 条：GEMM/GEMV/BMM/DMA/Vector/Reduce/KV/Barrier） | — |
| QNN | 算子层 IR（matmul/attention/rmsnorm/rope/swiglu） | myllm dialect |
| Q-MLIR | MLIR 编译器层（fusion/tiling/layout/memory planning/scheduling） | accel dialect |
| qforge | 编译 CLI（HF → .qbin） | myllm-compile |
| qrun | 运行 CLI（.qbin → token） | myllm-run |
| QMetal | Runtime（Memory/Kernel/Command Queue/Device Control） | — |
| qsim | ISA 模拟器（功能级 + 时序级） | isa_ref / P3 模拟器 |

## 1. 冻结决策（P0 产出，全项目约束）

| # | 决策 |
|---|------|
| D1 | 编译路线：HF/PyTorch → MLIR → 加速器 IR → Q-ISA → ASIC；不做 LLVM backend |
| D2 | Q-ISA = Tensor Command ISA（33 条），非 scalar CPU ISA |
| D3 | QCore = Command Processor + Matrix/Vector/DMA/KV 引擎 + 8 MiB Scratchpad SRAM + 16 GiB HBM；**FPGA 原型以 DDR 替代 HBM**（带宽/突发按板卡重标定，ISA/协议不变，DDR 口径 P9 立项时冻结） |
| D4 | 双模阵列：128×128 阵列，PF 整块 GEMM / DC 16 lane×8 行 GEMV |
| D5 | 验证模型：**Qwen3-0.6B 先行 → Qwen3-4B/8B 放量**；架构按 8B 尺寸设计（0.6B 是其子集）；batch=1；4K→8K ctx；BF16→INT8→INT4 |
| D6 | 特性集：GQA、RoPE、RMSNorm（含 QK-norm）、SwiGLU、KV Cache、Causal Attention |
| D7 | KV cache 一级公民：KV.APPEND / KV.STORE_BLOCK / KV.LOAD / KV.GATHER |
| D8 | 验证顺序 0.6B→4B→8B，只换尺寸不换架构；单 QCore → 后续 QCore×N + NoC |
| D9 | host 采样：argmax/temperature 由 host 完成，logits 经 DMA.STORE 回传（ISA 无 argmax） |
| D10 | roofline 口径：验收锚定 sustained 720 GB/s（读 900×80%） |
| D11 | 平台身份：完全开源、个人可复现；体系完整度对标 Tenstorrent |
| D12 | **三级 Golden Reference**（总验证原则）：PyTorch → qsim → RTL/FPGA 三者输出在规定精度内一致 |
| D13 | 范围禁止：训练、多卡/分布式推理、70B、MoE、多模态 |

规格冻结文档：`docs/spec.md` + `docs/spec-src/00–05`（P0 ✅ 完成，2026-08-13）。

## 2. 十大目标 ↔ 节点映射

| 目标 | 节点 | 状态 |
|------|------|------|
| G1 Qwen Reference（dump 中间 tensor/KV/logits） | P1 | 🔄 执行中 |
| G2 Q-ISA | P0 | ✅ 完成 |
| G3 ISA Simulator（仅靠 Q-ISA 完整执行 Qwen） | P3 | 待启动 |
| G4 Q-Compiler（HF/PyTorch → MLIR → Q-ISA） | P2+P4 | 🔄 P2 执行中 |
| G5 Hardware-aware Optimization（fusion/tiling/memory planning/PF-DC scheduling） | P6 | 待启动 |
| G6 QCore RTL（SystemVerilog） | P7 | 待启动 |
| G7 RTL 验证（与 qsim/PyTorch 数值对齐） | P8 | 待启动 |
| G8 FPGA（真实模型上板生成 token） | P9 | 待启动 |
| G9 ASIC Flow（综合/STA/面积/功耗/频率） | P10 | 待启动 |
| G10 完全开源（compiler/ISA spec/asm/sim/runtime/RTL/FPGA/ASIC scripts） | 贯穿全部节点 | 持续 |

## 3. 节点总览

| 节点 | 交付物 | 验收标准 | 依赖 |
|------|--------|----------|------|
| P0 规格冻结 | spec.md + 六分册 | ✅ 已完成（15 条审计全裁决） | — |
| P1 Qwen Reference | ref/ + golden/ + roofline | M1：greedy 与 HF 一致 + 逐层瓶颈明确 | P0 |
| P2 Q-Compiler 骨架 | qnn/qisa dialect + 编码器 + qsim 核心 | M2a：单 linear 全链路数值正确 | P0 |
| P3 qsim 模拟器 | 功能级+时序级 | M2b：跑通 layer trace，周期数/利用率 | P0/P2 |
| P4 qforge 前端 | HF → .qbin + 量化 | M3：qforge 产出可加载 qbin | P2 |
| P5 qrun + QMetal | 端到端运行时 | M4：8K ctx 输出与 HF 逐 token 一致 | P1+P3+P4 |
| P6 硬件感知优化 | tiling/scheduling/双缓冲 | M5：decode 达 sustained roofline 80%+ | P5 |
| P7 QCore RTL | SystemVerilog 三引擎+CP+SRAM | M6：RTL co-sim 与 qsim 一致 | P0（ISA 冻结后） |
| P8 RTL 验证 | 三级 golden 数值对齐 | M7：PyTorch=qsim=RTL 精度内一致 | P5+P7 |
| P9 FPGA 原型 | 板级工程 + 上板生成 token | M8：FPGA 上真实 token 生成 | P8 |
| P10 ASIC 流程 | 综合/STA/面积/功耗/频率报告 | M9：可综合 + 评估报告 | P7 |

## 4. 节点细化（P0–P6 为现行计划，P7–P10 为概览）

### P0 ✅（完成）
ISA v0（33 条、128-bit 编码）、SRAM/HBM/DMA、双模阵列、KV 协议、qbin 容器；
一致性审计 15 条冲突全部裁决。见 docs/spec.md §3。

### P1 Qwen Reference（执行中）
- **目标模型 Qwen3-0.6B**（config 核验：hidden 1024 / 28 层 / 16 Q head / 8 KV head /
  head_dim 128 / intermediate 3072 / vocab 151936 / rope_theta 1e6 / QK-norm；
  tie_word_embeddings 待核验——0.6B 通常 tied=true）。
- 手写全路径参考 + golden trace（J1 schema）+ roofline（0.6B 口径重算；spec §4 为 8B 设计上限）。
- 验收（M1）：greedy ≥20 token 与 HF 完全一致；逐层瓶颈表。

### P2 Q-Compiler 骨架（执行中）
- QNN（qnn dialect）→ Q-MLIR（qisa dialect）→ Q-ISA 编码器 → qbin writer → qsim 功能核心。
- 验收（M2a）：单 linear 层（0.6B Wq **2048×1024**）{PF,DC}×{BF16,INT8} 四例数值正确。

### P3 qsim 模拟器
- qsim 功能核心上加时序：四引擎流水、SRAM/HBM 带宽、bank 冲突、双缓冲。
- 验收（M2b）：跑通 P1 的 layer trace，输出周期数/利用率/带宽分解。

### P4 qforge 前端
- `qforge compile Qwen/Qwen3-0.6B --dtype int8`：config 解析 → 建图 → safetensors →
  量化（W8A8 默认 / W4A16 可选）→ 打包 → tiling → 调度 → .qbin。
- 验收（M3）：qbin 可被 qsim 加载执行；逐层量化误差达标（INT8 先，INT4 达标才默认）。

### P5 qrun + QMetal
- `qrun qwen3-0.6b.qbin`：QMetal（Memory/Kernel/Command Queue/Device Control）+
  tokenizer + KV 生命周期 + host 采样（D9）。
- 验收（M4）：BF16 下 4K/8K ctx 输出与 HF 逐 token 一致；固定 benchmark perplexity 无偏差。

### P6 硬件感知优化
- Tiling 自动搜索、SRAM 规划、双缓冲、DMA overlap、PF/DC 调度、KV 预取、量化映射、
  KV 重读消减（streaming/选择性重读）。
- 验收（M5）：decode 达 sustained roofline（720 GB/s）的 80%+，口径随 context（含 KV 全窗口重读）：
  0.6B 权重流天花板 1208 token/s；含 KV 重读 4K ≈ 675 / 8K ≈ 469 token/s，80% 按对应 context 计；
  8B 口径 ≈ 76（短上下文），消融齐全。
  附 488 @4K 口径（KV staging SRAM 写计入关键路径）：1e9/(28×65,536+216,087)≈488，假设 dense 权重流与 KV staging 重叠、lm_head 权重读（216,087 cyc）不重叠。

### P7 QCore RTL
- SystemVerilog：Matrix（128×128 dual-MAC，PF/DC 双模）、Vector（128-lane）、
  DMA、Command Processor、Scratchpad SRAM（16 bank）、KV 地址生成器。
- 验收（M6）：RTL co-sim 与 qsim 逐指令一致；单 layer 跑通 golden。

### P8 RTL 验证（三级 golden 会师）
- PyTorch 参考 ↔ qsim ↔ RTL 三级数值对齐（规定精度内），逐层 + 端到端。
- 验收（M7）：三级输出一致报告。

### P9 FPGA 原型
- 板卡选型（含 DDR 的 FPGA 开发板，预算友好优先）；QCore 上板；0.6B INT8 权重驻留 DDR；
  上板生成真实 token。
- **板卡资源下限**：DDR ≥ 1.5 GiB（权重 0.6 GB + 8K KV 0.875 GiB + 运行时缓冲；或明示 4K 限制 ≈1.1 GiB）；
  阵列/SRAM 允许缩编实例（如 64×64 阵列、4 MiB SRAM），性能口径以 qsim 时序模型为参照重标定。
- 验收（M8）：FPGA 输出与 qsim 一致；发布 FPGA 工程。

### P10 ASIC 流程
- 综合、STA、面积、功耗、频率与性能评估；开源 scripts。
- 验收（M9）：可综合 + 评估报告（面积/功耗/频率/token/s）。

## 5. 里程碑

| 里程碑 | 判定条件 | 节点 |
|--------|----------|------|
| M0 | ✅ spec 冻结（2026-08-13） | P0 |
| M1 | 0.6B golden 与 HF 一致 + roofline | P1 |
| M2 | M2a 单层 MLIR→Q-ISA→qsim 数值正确；M2b qsim 时序跑通 layer trace | P2+P3 |
| M3 | qforge 产出可加载 .qbin | P4 |
| M4 | qrun 输出与 HF 逐 token 一致 | P5 |
| M5 | decode 达 sustained roofline 80%+ | P6 |
| M6 | RTL co-sim 与 qsim 一致 | P7 |
| M7 | 三级 golden（PyTorch=qsim=RTL）一致 | P8 |
| M8 | FPGA 上板生成真实 token | P9 |
| M9 | ASIC 评估报告 | P10 |

## 6. 并行与依赖

```text
P0(✅) ─┬─ P1 ────────────┬─ P5 ─ P6 ─┬─ P8 ─ P9 ─ P10
        ├─ P2 ─┬─ P4 ──────┘           │
        └─ P3 ─┘                       └─ P7 ───────────┘
            （M2 会师）                    （ISA 冻结后与软件线并行）
```

- 关键路径（软件线）：P0→P2→P4→P5→P6→P8→P9→P10。
- 硬件线（P7）在 ISA 冻结后与 P4–P6 并行，不影响关键路径。

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| 模型范围失控（MoE/GDN/multimodal） | D5/D13 冻结；D13 禁止训练/多卡/70B/MoE/多模态 |
| INT4 精度不达标 | 逐层误差检查 + BF16/INT8 回退路径 |
| 性能不及预期 | 一切性能决策以 qsim roofline 为依据（D10 口径） |
| KV cache 事后补课 | D7：ISA 从 P0 起含 KV 指令 |
| 0.6B 与 8B 口径混淆 | spec §4 标注"8B 设计上限"；逐模型重算流量 |
| FPGA 板卡/资源不足 | P9 板卡下限 = DDR ≥ 1.5 GiB（含 8K KV 0.875 GiB）或缩编阵列/SRAM 实例（qsim 参照） |
| ASIC PDK 不可得 | P10 用开源 PDK（如 SkyWater）或工艺库 agnostic 的综合评估 |

## 8. 推进规则

1. 每个节点开工前交付物与验收标准已写明——验收不过就留在该节点。
2. 每周至少一个可运行增量（编译产物 / 模拟器输出 / RTL 波形），不允许"先搭一个月框架"。
3. 新想法先记 backlog，不得挤占当前节点验收。
4. **每个新计划须经 subagent 评审，双方意见一致后方可执行**（目标工作流）。
5. **一切验收最终落到三级 golden：PyTorch → qsim → RTL 数值一致**（D12）。
