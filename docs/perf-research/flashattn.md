# R-FlashAttn — FlashAttention 文献与 QCore 适用性

> 口径：QCore 冻结常数（spec §4、p1-roofline、p3-sim-report、p6-opt-report）；0.6B INT8（KV BF16，114,688 B/token）。
> 文献：FA-1 arXiv:2205.14135、FA-2 2307.08691、FA-3 2407.08608、Flash-Decoding（Princeton NLP, 2023-10）、
> PagedAttention 2309.06180、StreamingLLM 2309.17453、H2O 2306.14048。

## 1. 核心思想（各一句话）

| 文献 | 核心思想 |
|---|---|
| FA-1 | IO-aware：S=QKᵀ 分块 + online softmax（running max/sum 跨块递推），S 的 O(N²) HBM/SRAM 驻留降为 O(N)；反向用重算换显存 |
| FA-2 | 纯调度：单头跨 thread-block 并行、减非-matmul FLOP；A100 达 50–73% 理论 FLOPs |
| FA-3 | Hopper 专用：warp-specialization/TMA 异步重叠、GEMM-softmax 交织、FP8 块量化+incoherent processing |
| Flash-Decoding | decode 沿 KV 维 split-K：每 split 并行算注意力并输出 1 个 log-sum-exp 标量，末次 reduce 合并 |
| PagedAttention | KV 按页管理（类虚拟内存）：消除碎片、跨请求共享；vLLM 吞吐 2–4× |
| StreamingLLM | attention sink（首 4 token）+ 滚动窗口 W：无微调扩展到无限长流式 |
| H2O | 动态保留 20% heavy-hitter（历史累积 score 最大）+ 最近 token：近全精度、延迟 −1.9× |

## 2. 适用性矩阵（逐条对照 QCore 已有实现）

| 技术 | QCore 现状 | 判定 |
|---|---|---|
| 分块 score（N≤128 tiling） | PF 128-token block、DC 16-lane per-head GEMV、K/V tile ≤2048（05 §1.5/§5.1-5.2，04 §2.2） | **已内化**（构造性：S 从不整体物化） |
| 跨 tile online softmax | 04 §3.4：L>128 跨块 max/sum 全局规约（L=8192 ≈360 cyc），VREDUCE_MAX/SUM 6 指令/行 | **已内化** |
| 单副本 KV 载入 + GQA 共享 | KV.LOAD 单副本已冻结（D16）+ P7 内部广播总线（batch_stride_B=0，04 §2.2） | **已内化**（优于 GPU 侧 kernel 技巧） |
| FA-1 反向重算 | QCore 纯推理（32.77 TMAC/s 无反向通路） | **不适用** |
| FA-2 work partitioning | GPU warp/occupancy 概念无对应物；QCore 等价物 = MODE PF/DC 切换 + 16-lane 分配，已做 | 不适用/已内化 |
| FA-3 warp-spec/TMA 异步 | 思想层已内化：DMA 4 in-flight + PREFETCH 双缓冲 + KV staging ⟂ 权重流重叠（P6 验证：4K 重叠 −38%） | **部分内化**；寄存器级 producer-consumer 不适用 |
| FA-3 GEMM-softmax 交织 | Vector/Matrix 异引擎按依赖图串行；decode Vector 513 cyc 已隐藏于 HBM 读下（P3 §3.3） | 不适用（收益 ≈0） |
| FA-3 FP8 / incoherent | 对应 QCore INT8 KV 量化（05 §8 backlog）+ per-128-group 量化映射（P6） | 缺失→backlog |
| Flash-Decoding split-KV | 单核已含串行 split-K：2048-token 分 tile + 跨块 online softmax；并行 split 需多核 | **单核不适用**；QCore×N 衔接点（见 §5） |
| PagedAttention | v0 batch=1 单序列、固定 slab（2 MiB，零碎片）；batch>1/32K 时必需 | **v0 不适用**；backlog（05 §8） |
| StreamingLLM（sinks+窗口） | 无；KV.LOAD 连续区间语义天然支持（pos_start=pos−W） | **缺失、纯软件可落地** |
| H2O（选择性重读） | 无；块粒度=多段连续 LOAD 纯软件；元素级需索引式 LOAD | **缺失**；分级落地 |
| block-sparse attention | 由 StreamingLLM/H2O 固定预算式覆盖；真稀疏掩码需硬件 | 暂不做 |

## 3. 收益量化（全部基于 P6 重叠模型推导）

decode 每 token 周期 = 28 × max(21,849 + 4096·q·R/720, 4096·q·R/256) + 216,087，
其中 R = 每层实际重读 KV token 数（r·ctx 或窗口 W），q = KV 字节系数（BF16=1、INT8=0.5）；
21,849 = 每层权重读、216,087 = lm_head（P3 §3.1）。绑定资源：SRAM 写口（staging 16·q·R cyc/层）
或 HBM 读（权重+KV）。锚点自检：r=1 时 4K=2,051,095 cyc→**487.5**、8K→**257.3**，与 P6 逐位一致；
R=0 上界 827,846→**1208**（=P6 假设上界 827,859，差 0.002% 为取整）。

| 方案（R） | 4K | 8K | 32K* | 硬件改动 |
|---|---|---|---|---|
| 当前 P6 调度（r=1 全窗口） | 481 | 255 | 67 | — |
| H2O r=0.5（0.5·ctx） | 866 | 488 | 140 | 无（块粒度） |
| H2O r=0.25（0.25·ctx） | 1009 | 866 | 257 | 无（块粒度） |
| H2O r=0.1（0.1·ctx） | 1120 | 1043 | 594 | 无（块粒度） |
| Streaming W=2048（+4 sinks，+0.2% 可忽略） | 866 | 866 | 866 | **无** |
| Streaming W=1024 | 1009 | 1009 | 1009 | **无** |
| Streaming W=512 | 1100 | 1100 | 1100 | **无** |
| 上界 R=0（窗口全驻留） | 1208 | 1208 | 1208 | 需 16/32 MiB SRAM，不可行 |
| INT8 KV × r=1 | 866 | 488 | 132 | KV 量化（backlog） |
| INT8 KV × r=0.25 | 1100 | 1009 | 488 | KV 量化（backlog） |

*32K 需 slab 扩容/paged KV 前置（v0 slab 上限 8K，05 §1.2/§8）。

提升倍率：**8K 255→866/1009 = 3.4×/4.0×；4K 481→1009 = 2.1×；32K 67→866 = 12.9×**。
质量注：StreamingLLM 丢长程依赖（须保 sinks、窗口 ≥1024）；H2O r=0.25 论文口径近全精度。
开销注：KV 指令 ≈(⌈R/2048⌉×8+8)×28 ≈1.1K/token，占 DC 程序 641K 指令的 0.2%，忽略。

prefill（flash 式分块的 SRAM 驻留节省——**已内化，收益为已实现事实**）：

| 项 | 4K prompt | 8K prompt |
|---|---|---|
| S 物化驻留 L²×4B | 64 MiB | 256 MiB |
| QCore tiled 工作集（K/V staging 1 MiB + 分数块 128×128 阵列内） | ~2 MiB | ~2 MiB |
| 节省 | **32×** | **128×** |

> 这正是 prefill 长上下文不溢出 8 MiB SRAM 的原因；flash 系对 prefill 的剩余空间：
> ① causal skip（跳上三角，attention MAC 减半）= 4K −17.7% / 8K −26.0% prefill 时间
> （attention 占 prefill 总 MAC 35%/52%），需阵列三角执行（硬件改）；② 末块 KV staging
> 65,536 cyc ≈ 矩阵 63,488 cyc，需与前块计算重叠（PF 双缓冲，R-Audit 杠杆）。prefill 本身
> 计算受限 94–99%，flash 技巧不减少计算量，收益远小于 decode 侧。

## 4. 推荐优先级（收益 × 成本）

| 优先级 | 方案 | 收益 | 成本 | 理由 |
|---|---|---|---|---|
| **P0** | StreamingLLM：sinks(4) + 滚动窗口 W=1024/2048，纯程序改造（每层 KV.LOAD 改 pos_start 区间，两段 LOAD） | 8K 3.4–4.0×、4K 2.1×、32K 12.9× | ISA/RTL 零改动 | 唯一零硬件成本的即时长上下文杠杆；先 qsim 建模（重读桶=4096·W）→ QMetal 端到端 golden 验证质量 |
| **P1** | 块粒度 H2O 选择性重读（历史累积 score 选 top-k 块 + 最近块，多段连续 LOAD） | 与 r 对应，长程质量优于 P0 | 纯程序；score 累积维护 ~256 cyc/层（Vector 隐藏） | 兼顾语义长程依赖；qsim 消融后再定 r（0.1–0.25） |
| **P2** | INT8 KV 量化 + 索引式 KV.LOAD（元素级 H2O） | 与 r 相乘叠加（8K r=0.25 再 +17%） | 量化映射（backlog 已有）+ KV 地址生成器读 SRAM 索引表（小硬件改） | 字节减半同时作用于 staging 写口与 HBM 读两道墙 |
| **P3** | prefill causal skip 三角执行 | prefill −18%（4K）/ −26%（8K） | 阵列三角/变长窗口模式（硬件改） | prefill 计算受限下唯一大杠杆；P0–P2 落地后再排期 |
| **P4** | SRAM 写口 256→512 B/cyc | r=1 时 8K 255→488 | 存储宏带宽改动（R-Audit 范围） | 与 P0 正交；P0 落地后其紧迫性下降 |

## 5. 衔接

- **R-NoC（QCore×N）**：Flash-Decoding 的并行 split-KV 在多核才成立——单核 decode 已 per-head 16-lane
  并行 + 串行 split（2048 tile + 跨块 online softmax 即其串行实例，04 §3.4）；N 核并行 split 的 KV 分块
  广播/收集与 log-sum-exp 归约流量直接决定 NoC 预算（KV 0.6B 114,688 B/token、8B 147,456 B/token）。
- **R-Audit**：本报告的 P0/P1 即其「KV 重读消减」杠杆的量化展开（各档收益 + 上界 1208）；
  SRAM 写口、KV 量化、分页 KV 为其杠杆子集，数字可互引。
