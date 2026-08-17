# 投机解码硬件友好变体（R-Spec）——batch=1 固定功能加速器可行性 + 权重墙增益 + ISA 影响

> 口径：与 audit.md 冻结常数一致——HBM 读 720 B/cyc、SRAM 写 256 B/cyc、1 GHz=1 cyc/ns；
> KV INT8+W=2048 档 decode = 28×33,500+216,087 = 1,154,087 cyc/token = **866 tok/s**（audit §3.1）；
> 33,500/层 = 权重 21,849 + KV 窗口 11,651（HBM 读绑定）；216,087 = lm_head 155.6 MB 权重流尾部串行（qsim/timing_p6.py:111）。
> 文献数字全部引自各论文 HTML 全文原文（本次抓取）。
> 任务单 ID 修正：**REST 正确 ID = 2311.08252**（任务单 2404.08639 是 COCONut，与投机解码无关）。

## 0. 结论摘要

1. **免训练草案可行**：Lookup/PLD、Lookahead（2402.02057）、REST（2311.08252）、LLMA（2304.04487）四路全部可迁移
   （host 侧字符串/检索匹配出草案，零模型成本、零训练、贪心验证 lossless）。**经典 draft-模型（Leviathan）对小目标关闭**：
   Qwen3 最小即 0.6B（=目标本身），无更小同词表模型；host CPU 跑 0.5B 级 draft 仅百 tok/s 量级，远低于所需 γ×866 tok/s。
   Medusa（2401.10774）/EAGLE（2401.15077）需训练 → **D13 否决**，但记录其 2.2–3.6× / 2.7–3.5× 为收益天花板参照。
2. **增益（第一性，§3）**：γ=4 链式验证，贪心接受率 α∈[0.35,0.55] → **2.4–3.2×**（866→2.1–2.8K tok/s）；
   γ=1 高 α ≈1.7×（与 audit「2-token ≈1.5–1.8×」一致）。批验证摊薄仅 ≈+1.6%@n=5：权重流与 KV 窗口每次 verify 只流一次，
   计算桶 n×~1,250 cyc/层被绑定 33,500 隐藏至 n≈26。文献 2–3× 声称在本架构上是**保守下界**。
3. **ISA 最小集**：单序列 n 候选链式验证（v0 无需树掩码：链=因果平移）；KV 窗口 staging 提升到候选循环外；
   lm_head 权重流跨候选共享；每候选窗口指针 (start,end)+sinks。树型多分支（REST Trie 式）留 gen-2。
4. **立项门槛**：0.6B 的贪心 n-gram α 无任何文献锚点（文献全部 ≥7B）。进 RTL 前须 qrun 实测
   （host 侧 n-gram + 现有 20/20 交叉一致口径，半天工作量）：α≥0.4 立项；α∈[0.25,0.4) 降 gen-2；α<0.25 关闭。

## 1. 文献关键数据表

| 论文 | 草案来源 | 训练? | 关键数字（原文） | 小模型证据 |
|---|---|---|---|---|
| Leviathan SpS 2302.01318 | 小 draft 模型 | 否（需现成同词表小模型） | "leads to a 2–2.5× speedup"（§1）；HumanEval "almost 2.5×"；XSum nucleus 最优 K=3；K 增大增益平台/回退 | 无 <7B 数据；7B draft 于 16 TPU "actually increases the latency"（§6） |
| Medusa 2401.10774 | 多头解码头 | **是**（Medusa-1 冻主干训头 / Medusa-2 联合 / 自蒸馏） | Medusa-1 7B **2.18×**、13B **2.33×**；Medusa-2 13B **2.83×**，总 **2.3–2.8×**；类目峰值 coding 3.29× / extraction 3.62× | batch=1 设定与我们同（§1）；训练冲突 D13 |
| EAGLE 2401.15077 | 特征级自回归（1 层+LM head） | **是**（可训 0.24–0.99B，RTX 3090 1–2 天） | LLaMA2-Chat 70B **2.7–3.5×**；Vicuna-7B 贪心 **3.33×（τ=4.29）**；比 Lookahead 快 1.70–2.08× | **经典 spec 实测：7B 无可用 draft、13B 无增益、33B 1.12×、70B 1.88×** → 小目标关闭的直接证据 |
| Lookahead 2402.02057 | Jacobi 迭代 + n-gram 池（生成轨迹） | 否 | **1.5–2.3×**（code 2.3×）；MT-Bench **1.8×**；贪心 lossless | **"smaller models also exhibit a higher speedup"**；需 FLOP 盈余，"compute-bound（大 batch）may cause slowdowns"；步数线性下降需 FLOP 指数上升 |
| REST 2311.08252 | 数据存储检索（suffix array+Trie，贪心验证） | 否 | HumanEval **2.16–2.36×**；MT-Bench Vicuna **1.62–1.77×**；检索 <1 ms、开销 <6%；贪心>核采样 | datastore 27 GB（The Stack）；无小模型数据 |
| LLMA 2304.04487 | 参考文本 n-gram 拷贝验证 | 否 | "over 2×"；RAG 2.48/2.22/2.49、cache 2.22/2.19/3.06（7B/13B/30B）；n=1、k=14–18；贪心 lossless | 依赖输出与参考重叠（RAG/缓存/多轮场景） |
| PLD（Saxena 2023，HF blog） | prompt+已生成文本 n-gram 匹配 | 否 | Lookahead §5.4 消融：prompt-as-reference（Yang 2023/Saxena 2023）"can further boost Lookahead Decoding" | 机制同 LLMA 拷贝式，host 零成本 |

## 2. 与 QCore 事实的适用性判定

| 候选方向 | 判定 | 理由（对照 QCore 事实） |
|---|---|---|
| Leviathan 经典 draft 模型 | **不适用** | 目标 0.6B 已是最小档，无更小同词表模型；host draft 吞吐不够（verify 步长 1.15 ms，draft 需 ≥γ×866 tok/s）。EAGLE 实测背书：spec-sampling ≤13B 无增益 |
| Lookup/PLD / Lookahead / REST / LLMA | **适用（首选）** | 免训练免模型；草案 = host 字符串匹配（µs 级 ≪ 1.15 ms 步长；REST 实测检索 <1 ms）；贪心验证 lossless → 输出分布不变 → 与 20/20 交叉一致门槛同构 |
| 自草拟（self-draft 层跳过） | **不适用** | 草案在设备上跑仍流权重：每草案 token ≈(28−k)/28 层 + lm_head 尾部 ≈0.25–0.5 步；γ=2,k=14 → (1+1.4α)/2 ≤1.2×；γ=4,k=21 ≤1.9× 且浅层草案 α 无保证。收益不如零成本 lookup 草案 |
| Medusa / EAGLE | **否决（D13 训练）** | Medusa 冻主干仍须训头；EAGLE 训 0.24B+ 自回归头。且 EAGLE 式草案需目标隐藏态+LM head（设备上每草案 token ≈216K cyc=0.19 步）——即便放行训练，设备端草案成本也吃掉大半收益 |

关键事实对接：
- **权重墙 B1（27.3×）**：投机 verify = 单序列内部的 n 候选批处理——不违反 batch=1 对外契约（D5/D13 精神），
  却复用 batch>1 才有的权重流共享。audit 把 batch>1 排 backlog 因其架构级改动；链式 n 候选是静态窄版，成本低一个量级。
- **计算余量**：decode 计算桶 ~1,121 cyc/层（matrix 608 + vector 513，audit B 注）vs 绑定 33,500 → γ 上限 ≈24 前
  verify 步长不增（ε≈0）。GPU 文献的批摊薄（FlashAttention/tree attention 开销）在固定功能流水线上天然更小。
- **窗口化 KV（O1 W=2048）**：n 候选窗口互差 j 位 → union = R+γ 条，KV HBM 读仅 +γ/R ≈ +0.24%；每候选窗口指针即"树掩码"。

## 3. 增益推导（第一性，冻结常数代入）

T_step = 28×33,500 + 216,087 = 1,154,087 cyc/token（866 tok/s，audit §3.1）。链式 verify（n=γ+1 候选，权重流+KV 窗口流各一次）：
- 每层：绑定 33,500 不变；计算 n×~1,250 cyc 隐藏（n<26）；窗口 union +γ 条。
- lm_head：155.6 MB 权重流一次（216,087 主体）；MAC 增量 n×155.6M/32,770 = n×4,748 cyc。
- ⇒ T_verify(n) ≈ 1,154,087 + (n−1)×4,748。每步产出 (1+αγ) token（贪心：接受至首个失配）：

| γ (n) | α=0.35 | α=0.45 | α=0.55 | α=0.65 |
|---|---|---|---|---|
| 1 (2) | 1.34× | 1.44× | 1.54× | 1.64× |
| 2 (3) | 1.68× | 1.87× | 2.08× | 2.28× |
| 4 (5) | 2.36× | 2.76× | 3.15× | 3.54× |
| 4 (5) +0.5%/候选余量 | 2.30× | 2.69× | 3.07× | 3.45× |

- γ=1、α=0.75 → 1.74×，与 audit「2-token 1.5–1.8×」自洽；理论上限 (1+αγ)=5，摊薄 ε≈1.6%@n=5。
- 对照文献：Lookahead 贪心 1.5–2.3×（GPU 摊薄大）、REST 2.16–2.36×、EAGLE 3.33×（训练 + τ=4.29）。
  **文献 2–3× 声称对应本架构 α∈[0.25,0.5] 即可达成——摊薄不是瓶颈，0.6B 的 α 本身才是**（0.6B 分布更平坦，α 可能低于 7B+ 文献，须实测）。

## 4. ISA 影响（v0 最小集）

1. **多候选链验证指令**：decode 步带 candidate_count=n 与每候选窗口指针 (start,end)+sinks；权重流与 KV 窗口 staging
   提升到候选循环外（编译器负责——量化/调度是编译期问题，天然友好）。
2. **KV 提交语义**：投机索引 vs 已提交索引分离；verify 后按 accept_count 提交（写 n 条、指针回滚 n−k 条，零数据搬移）。
3. **lm_head 跨候选共享**：权重流一次 + n×MAC；输出 n×vocab argmax（贪心验证在设备端比对 draft token 即可）。
4. **host 接口**：每步 host→设备 n−1 个草案 token（~10 B）；设备→host accept_count；lookup 草案 host 侧 µs 级，全流水隐藏。
5. **树型多分支（REST Trie / Medusa 稀疏树）留 gen-2**：需祖先指针+真树掩码；v0 链式已覆盖 PLD/Lookahead 主收益。
6. **B4 指令流**：程序按步计而非按 token 计 → inst/token ÷(1+αγ)，短 ctx 87% 占比同步缓解。

## 5. 方向建议

**支持（有条件立项）：lookup 草案 + 链式 n 候选验证（γ=4）作为攻击 B1 的 v1。** 理由：①四路免训练方案
（Lookahead/REST/LLMA/PLD）机制可迁移，lossless 贪心验证与现有质量门槛同构；②本架构摊薄 ε≈1.6% 远低于 GPU 文献，
2.4–3.2×（α∈[0.35,0.55]）有第一性依据；③ISA 增量小（候选维度 + KV 提交语义），不触 D13；④是唯一免训练攻击 B1 的路径。
**反对盲目进 RTL**：0.6B 贪心 α 无文献锚点。门槛 = qrun 实测 Qwen3-0.6B 目标语料上的 PLD/n-gram α（host 纯软件）：
α≥0.4 → ISA 立项（预期 2.4–3.2×）；α∈[0.25,0.4) → gen-2；α<0.25 → 关闭，B1 改靠 INT4 精度修复（R-Rot 路）或 batch>1。
优先级保持 audit §3 排序（O1/O2、Fmax 之后），与「KV 重读消减」同列，高于「SRAM 写口翻倍」。

## 引用

- 2302.01318 Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"（ICML 2023）
- 2401.10774 Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
- 2401.15077 Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
- 2402.02057 Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"（ICML 2024）
- 2311.08252 He et al., "REST: Retrieval-Based Speculative Decoding"
- 2304.04487 Yang et al., "Inference with Reference: Lossless Acceleration of Large Language Models"（LLMA）
- PLD: Saxena, "Prompt Lookup Decoding"（HF blog 2023；本报告经 Lookahead §5.4 引用间接取证，blog 原文因网络限制未直接抓取）
