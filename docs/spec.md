# QCore 平台架构规格 v0（Q-ISA v0）

- **版本**: 0.1（2026-08-13）
- **状态**: ✅ 冻结（P0 完成）。后续节点（P1–P10）不得引入架构级新决策；新想法一律进 §6 backlog。
- **目标模型**: Qwen3-0.6B dense（验证优先）→ Qwen3-4B/8B（放量）；架构按 8B 尺寸设计（0.6B 是其子集）；
  decoder-only，batch=1，4K/8K context，BF16→INT8→INT4。
- **分册**: 本文件是主入口与裁决记录；细节分册在 `docs/spec-src/`。

## 1. 冻结决策（D1–D16）

| # | 决策 |
|---|------|
| D1 | 编译路线：HF/PyTorch → MLIR → 加速器 IR → Tensor ISA → ASIC；不做 LLVM backend |
| D2 | ISA = Tensor Command ISA（33 条），非 scalar CPU ISA |
| D3 | Command Processor + Matrix/Vector/DMA/KV 引擎 + 8 MiB Scratchpad SRAM + 16 GiB HBM；**FPGA 原型以 DDR 替代 HBM**（带宽/突发按板卡重标定，ISA/协议不变，DDR 口径 P9 立项时冻结） |
| D4 | 双模阵列：128×128 阵列，PF 整块 GEMM / DC 16 lane×8 行 GEMV |
| D5 | 验证模型：Qwen3-0.6B 先行 → 4B/8B 放量；架构按 8B 尺寸设计（§4 数值为 8B 口径）；batch=1，4K→8K ctx，BF16→INT8→INT4 |
| D6 | 特性集：GQA、RoPE、RMSNorm（含 QK-norm）、SwiGLU、KV Cache、Causal Attention |
| D7 | KV cache 一级公民：KV.APPEND / KV.STORE_BLOCK / KV.LOAD / KV.GATHER |
| D8 | 验证顺序 0.6B→4B→8B，只换尺寸不换架构；单 QCore → 后续 QCore×N + NoC |
| D9 | host 采样：argmax/temperature 由 host 完成，logits 经 DMA.STORE 回传（ISA 无 argmax；见 §3.3） |
| D10 | roofline 口径：性能验收锚定 sustained 720 GB/s（读 900×80%；见 §3.3） |
| D11 | 平台身份：完全开源、个人可复现；体系完整度对标 Tenstorrent；命名见 §8 |
| D12 | 三级 Golden Reference（总验证原则）：PyTorch → qsim → RTL/FPGA 三者输出在规定精度内一致 |
| D13 | 范围禁止：训练、多卡/分布式推理、70B、MoE、多模态 |
| D14 | embedding 查表在 host（host 读 embedding 表 → hidden 经 DMA.LOAD 注入 SRAM，设备永不读 embedding 表）；lm_head 在设备算 logits |
| D15 | SRAM 交错寻址修正：bank = B[7:4]、bank 内字 = B[22:8]（修正 P0 公式 bank=addr[18:15] 与「16B 交错」的矛盾，2026-08-13 评审裁决） |
| D16 | KV.LOAD 冻结（decode datapath，见 §5.1） |

## 2. 分册索引与权威边界

| 分册 | 内容 | 权威域 |
|------|------|--------|
| 00-container | qbin 容器布局 + 命令流执行模型 | 容器格式 |
| 01-target-model | 目标模型卡（**8B 设计上限**，官方 config/index 逐字节核验） | 数值事实 |
| 01b-target-model-0.6b | 验证模型卡（0.6B 官方 config 核验：GQA 2:1、q_proj 2048×1024、tie=true） | 数值事实 |
| 02-isa | 33 条指令、128-bit 编码、opcode 表 | **指令语义** |
| 03-memory | SRAM bank/仲裁、HBM 带宽/延迟、DMA 2D | **内存行为** |
| 04-execution-engines | 阵列/向量微架构、双模推导、cycle 估算 | **执行单元行为** |
| 05-kv-cache | KV slab 地址公式、指令语义、prefill/decode 流程 | **KV 协议** |

冲突时以本文件 §3 的裁决为准。

## 3. 整合裁决记录（2026-08-13 一致性审计，15 条全裁决）

### 3.1 Blocker 裁决（3 条）

1. **BF16 峰值 = 8.19 TMAC/s**（非 16.38）。推导：BF16 尾数拆 2 子字 → 2×2=4 部分积，每 PE 2 乘法器 → 1 BF16 MAC 需 2 cycle → INT8/4。02-isa §11 已改，04 §1.2 为权威。
2. **W4A16 数据通路**：srcA=BF16/FP16、srcB=INT4、acc=FP32，走 BF16 尾数路径，吞吐与 BF16 同量级（8.19 TMAC/s）。INT4 打包双倍（65.54 TMAC/s）**仅限 W4A8**（INT8 激活）。02 §6 已改。v0 部署默认 W8A8；INT4 部署走 W4A16（与 HF 量化生态一致），W4A8 为硬件已支持的可选路径。
3. **SLAB_SHIFT 参数化**：KV slab 步长 = 2^SLAB_SHIFT，SLAB_SHIFT∈{20,21,22} 为 load-time 参数（C31）。默认 21（8K slab、pos 13b）；20（4K slab、pos 12b）供 **BF16 参考模式** 使用——BF16 权重 15.26 GiB + 8K KV 1.125 GiB 超出 16 GiB HBM，BF16 模式限定 4K context（15.82 GiB ✅）。02 §8、05 §1.3 已参数化，03 §7.3 为权威。

### 3.2 Minor 裁决（12 条，全部已修）

| # | 位置 | 裁决 |
|---|------|------|
| C4 | 03 §5.3 | DC 权重流 = 每 lane 4096×128×1B = 512 KiB，16 lane 全批 8 MiB，经 bsrc=1 直连 HBM 不占 SRAM 双缓冲 |
| C5 | 03 §3.2/§8.2 | decode 每 token 权重读 = 7.57 GB（dense 6.95G + lm_head 622M，不含 embedding） |
| C6 | 04 §2.4 | token/s 天花板双口径：峰值 1.2 TB/s 与 sustained 720 GB/s 并列；P6 验收锚定 sustained（见 D10） |
| C7 | 02 §12 | CONFIG 写 C 寄存器必须 class=0（示例已改） |
| C8 | 02 §8.1 | KV.APPEND 频次 = 每层每 KV head 一条（36×8=288 条/token） |
| C9 | 02 §8.4/§12 | GATHER count 字段 ≤8192，但 broadcast=1 时 v0 footprint ≤512（4 副本），编译器按 512 分 tile（仅 GATHER 路径） |
| C10 | 00 §2 | qkv_proj 示例 scale_bytes = 393,216（与 group=128 自洽） |
| C11 | 00 §1 | 发射队列 = 四引擎（Matrix/Vector/DMA/KV），WAIT 掩码 4 位 |
| C12 | 05 §5.1/§5.2 | 流程表标注为 attention/KV 步序，MLP 与 LM head 见 02 §12 第 7–8 步 |
| C13 | 00 §2 | 示例张量名 qkv_proj（QKV 融合变体），与官方 q_proj 区分 |
| C14 | 04 §2.4 | lm_head 已计入每 token 流量表（untied），注脚不再双重计数 |
| C15 | 00 §2 | flags bits[1:0] = 默认 dtype 编码（0=BF16，2=INT8，3=INT4） |

### 3.3 主会话新增决策

- **D9 host 采样**：argmax/temperature/top-k 采样由 host 完成——logits 经 DMA.STORE 回传 HBM 后由 host 读；ISA 不设 argmax 指令（VREDUCE_MAX 仅作数值校验）。02 §12 已改。
- **D10 roofline 口径**：理论上限用峰值 1.2 TB/s；**性能验收锚定 sustained 720 GB/s**（读 900×80%，03 §3.4）。P6 目标 = decode ≥ 76 token/s（**8B 口径**：INT8 sustained 天花板 95 token/s 的 80%；0.6B 口径按 §4 重算 ≈ 960 token/s）。**以上为短上下文口径；长上下文按 §4 修正注随 context 下调**（8B@4K≈88/@8K≈82、0.6B@4K≈675/@8K≈469，80% 按对应 context 计）。

- **D14 embedding host 侧边界**（2026-08-13，P3/P4 计划评审裁决）：token embedding 查表由 host 完成，
  hidden 向量经 DMA.LOAD 注入 SRAM，设备永不读 embedding 表；lm_head 在设备（每 token 全读其权重算 logits）。
  与 03 §3.2「仅首 token 读单行 8 KB」及 05 §5.1 步 1「首个 block 读 embedding」的措辞差异声明：
  该 8 KB 读发生在 **host 侧**（设备侧 embedding 读流量 = 0）；05 §5.1 的 DMA.LOAD 载入对象不变
  （hidden block），embedding 计算上移到 host。P5 qrun 契约据此。

- **D15 SRAM 交错寻址修正**（2026-08-13 评审裁决）：bank = B[7:4]、bank 内字 = B[22:8]
  （修正 P0 公式 bank=addr[18:15] 与「16B 交错」的矛盾）。

- **D16 KV.LOAD 冻结**（decode datapath，见 §5.1）。

## 4. 关键数值速查

| 量 | 值 |
|----|-----|
| 阵列峰值 | INT8 32.77 / INT4 65.54 / BF16 8.19 TMAC/s @1 GHz |
| SRAM | 8 MiB = 16 bank × 512 KiB；读 512 B/cyc、写 256 B/cyc；16B 粒度，bank=addr[7:4]（16B 粒度 16 路交错） |
| HBM | 16 GiB；读 900 / 写 300 GB/s；sustained 720 / 240 GB/s；64B 突发 |
| decode 瓶颈 | DC 权重流需求 32.77 TB/s vs HBM 1.2 TB/s = **27.3× 短缺**，阵列利用率 3.66%，HBM-bound |
| decode token/s 天花板 | BF16 79 / INT8 159 / INT4 317（峰值）；47.7 / 95.1 / 190.5（sustained） |
| decode 每 token 权重读 | BF16 15.1 / INT8 7.57 / INT4 3.78 GB（不含 embedding） |
| KV cache | 147,456 B/token；8K = 1.125 GiB（HBM 的 7.03%） |
| 权重驻留 | BF16 15.26 GiB（8K 溢出 → 限 4K）/ INT8 7.63 / INT4 3.81 GiB |
| 模式切换 | PF↔DC ≈ 300 cycle（一次请求仅一次切换） |
| 模型 | 8.19B 参数，36 层，hidden 4096，32 Q/8 KV head，head_dim 128，vocab 151936 |
| 0.6B 速查 | 751.6M 参数（tie=true 物理双份，去重 596M）；KV 114,688 B/token，4K 448 / 8K 896 MiB；decode 权重读 INT8 0.596 GB/token；GQA 2:1；q_proj 2048×1024 |
| KV 窗口重读 | decode 每 token 重读全窗口：147,456×ctx B（8B）/ 114,688×ctx B（0.6B）；8K 时 0.6B 重读 0.94 GB = 权重读的 158%（P1 roofline §6 核验，见 04 §2.4 修正注） |

> **口径说明**：上表全部数值为 **8B 设计上限口径**（模型 8.19B 参数）。0.6B 验证模型的
> 逐层流量按其形状（hidden 1024、28 层、intermediate 3072）重算；DC 模式 27.3× 带宽短缺、
> 阵列利用率 3.66% 等结论与模型尺寸无关，直接适用。
> **BF16 限 4K 的结论仅适用 8B**：0.6B BF16 权重 1.4 GiB + 8K KV 0.875 GiB ≈ 2.3 GiB，16 GiB HBM 无压力。

> **decode token/s 天花板修正**（含 KV 全窗口重读，05 §5.2 口径）：sustained（INT8）
> 8B @4K ≈ 88 / @8K ≈ 82 token/s；0.6B @4K ≈ 675 / @8K ≈ 469 token/s（详见 04 §2.4 修正注）。
> P6 验收口径随 context 而定；KV 重读消减列为 P6 优化项。

## 5. 待 P3 验证冻结项（非架构决策，协议已同时保留两案）

1. **已冻结（2026-08-13，P3 裁决）：decode = KV.LOAD 单副本（tile≤2048）；P7 阵列加内部广播总线（K/V 组内共享，ISA 不变）；GATHER 保留于 ISA（8B 4:1 GQA 或多副本场景，P7 可再评估）。依据：LOAD 65,736 vs GATHER 262,944 cyc/层@4K（4.00×，0.6B 2:1 GQA 下 4 副本中 2 份冗余）。**
2. bank 仲裁优先级与 DMA in-flight（2 vs 4）细节精化。
3. Vector 引擎 cycle 估算（04 §3.4 为 v0 近似，P3 精化）。

## 6. Backlog（P0 明确排除，不阻塞当前节点）

分页 KV、KV INT8/INT4 量化、HBM 扩容（24/32 GiB）、MoE/GDN/multimodal 模型、batch>1、32K+ context、FP8、RoPE strided 变体、DMA 乱序/QoS、标量访存、多轮对话二次 prefill、训练（D13）、多卡/分布式推理（D13）、QCore×N + NoC。

## 7. P0 验收判定

- ✅ spec 冻结：33 指令全集（SYS 5 + DMA 3 + MATRIX 3 + VECTOR 18 + KV 4）、五类引擎 128-bit 布局、KV 协议、容器格式均已定义；一致性审计（契约 10 条 + 跨文件 4 轴）15 条发现全部裁决修复。
- ✅ 后续节点可从本文件 + 分册直接引用实现，无需再开会定接口。

## 8. 命名对照（正式；旧名废弃）

| 名称 | 含义 | 旧名（废弃） |
|------|------|--------------|
| QCore | 芯片/平台（Matrix/Vector/DMA/CP + SRAM + HBM） | newlpu |
| Q-ISA | 本规格定义的 Tensor Command ISA（33 条） | — |
| QNN | 算子层 IR（matmul/attention/rmsnorm/rope/swiglu） | myllm dialect |
| Q-MLIR | MLIR 编译器层（fusion/tiling/layout/memory planning/scheduling） | accel dialect |
| qforge | 编译 CLI（HF → .qbin） | myllm-compile |
| qrun | 运行 CLI（.qbin → token） | myllm-run |
| QMetal | Runtime（Memory/Kernel/Command Queue/Device Control） | — |
| qsim | ISA 模拟器（功能级 + 时序级） | isa_ref / P3 模拟器 |
