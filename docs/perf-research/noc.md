# R-NoC：AI 加速器 NoC 调研与 QCore×N 多核方向（首代建议）

> 口径：全部推导引用冻结口径（spec §4；p1-roofline；p3-sim-report；p6-opt-report 的逐层周期模型；p10 面积/Fmax）。1 GHz=1 cyc/ns。TP=张量并行，LP=层流水，BP=批量并行。结论性数字均可由 §2 公式复算。

## 1. 文献调研：NoC 形态 × 划分模式

| 系统 | NoC 形态（已核验事实） | 对 QCore×N 的启示 |
|---|---|---|
| Simba（MICRO'19/CACM'21） | 36-chiplet MCM 推理原型；chiplet 内带宽远高于封装级链路，划分策略须最小化片间流量 | 小 N（4-8）片上 NoC 带宽充足，瓶颈是划分策略本身 |
| DaDianNao（MICRO'14） | 多芯片 ML 超算：聚合片上存储装下整个模型，高内部带宽/低外部通信；64 芯片系统 450.65× 加速、150.31× 能效（vs GPU）；28 nm 节点 P&R + 工业级片间互连 | 把权重留在 HBM 的 QCore 不可照搬"全片上"路线，NoC 只需搬运激活/部分和 |
| Tesla Dojo（HotChips'22） | 训练 tile=25×D1（354 核/die，2D mesh）；on-tile bisection 10 TB/s、off-tile 聚合 36 TB/s、9 PFLOPS | mesh 到 25 die 仍可行；bisection 量级远超 LLM 推理单流需求 |
| Cerebras WSE-2 | 850K 核、40 GB SRAM、200 Pb/s 片上 fabric、46,225 mm² | 权重全片上→无 HBM 墙；等价 QCore 需 ~950 个核的 SRAM 聚合，不可达→QCore×N 仍以 HBM 为权重源 |
| TPU v4（ISCA'23） | 4096 芯片；4×4×4 3D torus 积木，16 ICI 链路/面、96 光链路/块；OCS 毫秒级重配；twisted torus 使 all-to-all +1.31-1.63×；按并行模式选拓扑（pipeline 用 cigar 形 4×4×32，embedding 用 8³） | 拓扑随划分模式重配是超算级玩法；N≤8 时无必要，固定拓扑即可 |

**划分模式流量特征（LLM 推理 batch=1）**：

| 模式 | NoC 流量 | 权重流 | KV 流 | batch=1 decode 吞吐收益 |
|---|---|---|---|---|
| LP 层流水 | 单向点对点激活 2 KB/token/边界（0.6B） | 每核 1/N 层（本地） | **全本地** | **0**（单 token 串行跨核、无批内重叠；仅 prefill 有收益） |
| TP 张量并行（GQA 对齐） | 每层部分和 all-reduce，**延迟敏感** | 每核 1/N 列 | **全本地**（Q head 组与 KV head 同核） | **N×** ✓ |
| BP 批量并行 | 零（推理权重只读、无梯度同步） | 全复制 | 每核全窗口 | 需 batch>1（D13 禁） |
| 混合 TP×LP | 组内 all-reduce + 组间激活 | 1/(TP·LP) | 本地 | 继承 TP |

关键结论：**batch=1 decode 下 TP 是唯一带来吞吐收益的划分**；LP 的流水级重叠要求多个 token 在飞（prefill 的 128-token 块恰好是 mini-batch，故 LP 仅助 prefill）。

## 2. QCore×N 第一性流量预算

**内存方案裁决**：token/s ∝ 聚合 HBM 带宽（decode 恒 HBM/SRAM-bound，27.3× 权重流短缺是硬件常数）。
- **私有 HBM 切片（每核 sustained 720 GB/s，聚合 N×720）**：唯一可扩展方案。每核容量需求极小：0.6B N=4 权重 149 MB + KV@4K 112 MiB；8B N=4 权重 1.9 GB + KV@8K 288 MiB→4 GiB 切片即够；成本 = N× HBM stack 带宽。
- **共享 HBM（单 720 GB/s）**：decode 天花板不变（1208/488/257）；prefill 每阵列权重供给 = 32.77T/128 = 256 GB/s，共享 720 GB/s 至多喂 2.8 阵列→N≤2 且 decode 零收益。**否决**。

**逐层每核周期模型（TP，P6 口径扩展）**：
```
t_w = W_层/N/720；t_kv = K_层·ctx/N/720；t_s = K_层·ctx/N/256（SRAM 写口）
t_层 = max(t_w + t_kv, t_s)   // SRAM 写 ∥ HBM 读（P6 已证重叠成立）
t_token = Σ t_层 + lm_head/(N·720) + t_noc
```
W_层 = 15.73 MB（0.6B）/192.9 MB（8B）；K_层 = 4,096 B/token（两模型同为 8 KV head×128×2×2B）；lm_head = 155.6 MB/622.3 MB。

**token/s（理想 TP，无 NoC 开销）**：

| ctx | 0.6B N=1/2/4/8 | 8B N=1/2/4/8 |
|---|---|---|
| 短 | 1208 / 2416 / 4832 / 9664 | 95 / 190 / 381 / 761 |
| 4K | 488 / 975 / **1950** / 3900 | 88 / 176 / 352 / 705 |
| 8K | 257 / 515 / 1029 / 2059 | 82 / 164 / 328 / 656 |

全口径近似线性（每 token 成本 ∝ 1/N：权重流、KV 重读、**SRAM 写墙 1,835,008/N**、lm_head 216,087/N）。0.6B@4K N=4 = 1950 tok/s，**超过单核 KV 重读消减的假设上界 1208**（p6 §4）；N=1 列与冻结锚点 488/257/1208、88/82/95 逐位一致 ✓。

**NoC 带宽/延迟需求（TP，0.6B 推导）**：
- decode all-reduce：每层 o=16×128=2048 元素 + o_proj/down 输出 2048 元素 = **4096 元素×4B（fp32 累加）
  = 16 KB/层（bf16 落盘 8 KB）**（q/k/v 本地无跨核流量）；ring 每核发送 2(N−1)/N×16 KB ≈ 24 KB/层（N=4）
  → 28 层 672 KB/token，@1950 tok/s = **≈1.3 GB/s** → 带宽充裕，延迟主导。
- decode all-reduce 关键路径：N=4 ring 3 hops×~60 ns ≈ 0.2 µs/层 → 5.6 µs/token = 1.1%（@4K）；短 ctx N=8 ≈ 14 µs vs 103 µs = **14%** → 高 N 短上下文由 NoC 延迟定界（4704/8512 tok/s）。
- prefill 部分和：每层 4096 元素 × 128 token（block）× 4B = **2 MB fp32/层**（冻结形状：
  o=2048 + o_proj/down 2048）；每层预算 29.7 µs → **N=4 需 ≈101 GB/s（fp32）/ ≈50 GB/s
  （bf16 部分和）；N=8 需 ≈235 GB/s（fp32）**。首代建议：N=4 用 bf16 部分和 +
  512-bit（64 GB/s）链路；N=8 需 bf16 + 1024-bit（128 GB/s）链路（或 512-bit×2）。
- **KV 广播/收集 = 零**：GQA 对齐 TP 下每核独占其 KV head 的全部层（05 §1.3 slab 按 (layer, kv_head) 已天然可切）；P7 广播总线语义逐核不变。
- LP 对照：decode 零收益（487 tok/s ≈ 单核 488）；prefill 交接 256 KB/层 vs 63.5 µs 计算 → 每边界 ≈4 GB/s；LP prefill N=4 = 832,055 cyc/128-token 块 → 153.8k tok/s（4.0×，流水被 128 token 填满）。

## 3. 拓扑对比（N=4，1 GHz，512-bit 链路 = 64 GB/s/向，sky130 量级）

| 拓扑 | 匹配划分 | all-reduce 延迟 | 面积估算 | 评注 |
|---|---|---|---|---|
| ring-LP | LP | 无（单向流） | 4 链路≈4 mm² | decode 零收益，仅作 gen-3 组间级联 |
| ring-TP | TP | (N−1)×~60 ns+串行化 ≈0.2 µs/层 | 4 链路≈4 mm²+4 端口 0.5 mm² | N=4 可用；N=8 退化为 7 hops |
| **2×2 mesh-TP** | TP | 2-3 hops ≈0.12-0.2 µs/层 | 8 链路≈8 mm²+4 路由≈1 mm² | **首推**：Simba 验证过、bisection 2×ring、可扩 N=16 |
| 4×4 crossbar | TP | 1 hop ≈15 ns | 交叉开关≈1-2 mm²+4 NI | N≤8 延迟最优；N≥16 面积 N² 不可扩 |
| 层次化 TP×LP | 混合 | 组内 1 hop+组间 ring | — | gen-3（N=16：4 TP 组×4 LP 段） |

链路面积口径：512 线×4 mm×0.5 µm/线（多层金属）≈1 mm²/链路。参照：QCore 阵列 138 mm² + SRAM 100-268 mm²（p10 实测）→ **NoC 仅占 ~1-3% 面积**。

## 4. 首代建议（与 D8 衔接）

1. **形态**：QCore×4 单芯片集成（每核私有 HBM 切片、每核 sustained 720 GB/s，聚合 2.88 TB/s）+ 2×2 mesh、512-bit/向 NoC（64 B flit、~30 ns/hop、2-stage 路由）；GQA 对齐 **张量并行**；qforge 增加 TP tiling 与部分和 reduce 调度；qsim 时序模型按 §2 公式加 N 参数即可扩展。
2. **ISA 增量（最小）**：+2 条 NOC.SEND/NOC.RECV（128-bit 编码含 dst/addr/len）+ 跨核 BARRIER（WAIT 掩码扩展）；KV 指令、KV 地址公式、P7 广播总线全部不动。
3. **预期收益**：0.6B INT8：488→1950（4K）、257→1029（8K）、1208→~4700（短 ctx，含 NoC 延迟）；8B INT8：82→328（8K）、95→381（短）；prefill ≈4.0×（每核权重供给 256/4=64 GB/s ≪ 720）。成本：4× 逻辑面积 + 4× HBM 带宽（~1-3% 附加 NoC 面积）。
4. **与 D8 衔接**：v0 维持单 QCore（D13 禁多卡，D8 只换尺寸不换架构）；**QCore×N + NoC = v0 之后的 gen-2 第一步**，本报告即其预研。TP 是编译期划分+2 条指令，不破坏"换尺寸不换架构"；三级 golden（D12）与 M5/M6 验收体系逐核复用。
5. **风险**：① GQA 对齐上限 N≤8（0.6B 16Q/2=8 组；8B 8 KV head）——N>8 需跨核 attention KV 交换，另立课题；② lm_head 串行尾随 216,087/N cyc 不缩放（高 N 短 ctx 占比上升）；③ 短 ctx 高 N 的 NoC 延迟开销 ~14%；④ 28 层不可被 8 整除 → LP 路线 N 受限（TP 无此限）；⑤ batch>1（D13 解禁后）应重开 LP/BP 与混合划分评估。

## 5. 参考文献

1. Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer…", ISCA'23, arXiv:2304.01433（3D torus/OCS/twisted torus 数字出处）。
2. Zhu et al., "Theseus: Exploring Efficient Wafer-Scale Chip Design for LLMs", arXiv:2407.02079（Dojo 25×D1、10 TB/s on-tile bisection、36 TB/s off-tile；WSE-2 850K 核/200 Pb/s 数字出处）。
3. Shao et al., "Simba: Scaling Deep-Learning Inference with Multi-Chip-Module-Based Architecture", CACM'21, cacm.acm.org/research/simba（36-chiplet MCM；chiplet 内外带宽比结论）。
4. Chen et al., "DaDianNao: A Machine-Learning Supercomputer", MICRO'14, doi:10.1109/MICRO.2014.58（64 芯片 450.65×/150.31×，多芯片聚合片上存储路线）；ICT 项目页 novel.ict.ac.cn/diannao。
5. 内部冻结口径：docs/spec.md §4/§6、docs/p1/roofline.md、docs/p3/sim-report.md §5、docs/p6/opt-report.md §3-4、docs/p10/asic-report.md、PLAN.md D8/D13。
