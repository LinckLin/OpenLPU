# R-Rot：旋转量化修复小模型 W4A16 —— 文献调研与 QCore 适用性判定

> 2026-08-16 决策文档。输入事实（p5 §2.5 / audit B8）：Qwen3-0.6B W4A16
> per-128-group 对称 RTN 与 AWQ 搜索均 3/20（门槛 ≥8/10）；W8A8 已 10/10；BF16 20/20。
> QK-norm 离群（k_norm 通道比 2784×–180859×，o2-kv-int8.md §3）属激活/KV 侧事实，与权重侧量化无关。
> **结论先行：反对将旋转量化列为下一最优推进方向；支持将其作为 W4A16 修复的低成本最后验证项
> （纯编译期融合、零硬件改动），以受限实验裁决。**

## 1. 文献关键数据（权重 4-bit + 旋转，全部来自原文表格）

| 论文(ID) | 模型/规模 | 配置 | 数据 | 要点 |
|---|---|---|---|---|
| QuaRot 2404.00456 A.4 | Llama-2 7B/13B/70B | W4A16 per-col | GPTQ 8.25/5.65/3.87 → **+旋转 5.60/5.00/3.41**（FP 5.47/4.88/3.32）；RTN+旋转 6.76/5.48/3.66 | 7B+ 旋转+GPTQ 使 W4 逼近 FP |
| QuaRot 2404.00456 Tab.3 | 7B vs 70B | GPTQ−RTN 差 | 7B 差 2.27 → 70B 差 0.34 PPL | **小模型更依赖误差补偿** |
| SpinQuant 2405.16406 Tab.4 | Llama-2-7B | W4A16 RTN | 随机 Hadamard wiki 6.9±0.45；**Cayley 学习旋转 5.5=FP**（0-shot 64.6 vs FP 66.9） | 学习旋转 ≫ 随机旋转；7B W4A16 近无损 |
| SpinQuant 2405.16406 Tab.1 | **Llama-3.2-1B** | W4A8KV16 | GPTQ wiki 17.3(+29%) → nohad 15.3(+14%) → had 14.4(+7.5%)；0-shot 55.0→56.0/56.5（FP 56.9） | 1B 收窄过半仍有残差 |
| SpinQuant 2405.16406 Tab.1 | Llama-3.2-3B | W4A8KV16 | GPTQ wiki 25.2 → 11.6/11.5（FP 10.7） | 3B 残差 +0.8 PPL |
| BBT-spectral 2605.25203 Tab.1 | SmolLM-135M/360M、Qwen2.5-0.5B/1.5B | **W2A16** | wiki2 81.03→45.11 / 43.21→36.55 / 119.31→50.22 / 36.93→28.16（−15%~−58%） | **W4 消融：收益落入 ±0.5 PPL 噪声**；Qwen3-0.6B 需 PCA 替换 q_norm/k_norm（136.76→88.99 仍差）；Qwen2.5-1.5B 需 RoPE 对易 SO(2) 旋转 |
| AffineQuant 2403.12544 | Llama-2-7B/30B | W4A4 | C4 15.76 vs OmniQuant 18.02；30B 0-shot 58.61 | 仿射（非正交、学习），W4A4 聚焦，无 <1B 数据 |
| ParoQuant 2511.10645 | 推理 LLM | 权重-only INT4 | 较 AWQ +2.4% avg，运行时 <10% | 权重侧成对 Givens 旋转可超 AWQ |
| 2409.11055 (IJCAI'25) | 1B–405B | W4 GPTQ/AWQ | **小模型 4-bit 显著掉点、70B 稳定**；AWQ>GPTQ；FP8 最稳 | 规模是主因 |
| SLMQuant 2511.13023 | SLM 基准 | W4 | SLM 与 LLM 量化敏感度有本质差异，直接移植 LLM 方法欠佳 | 同上 |

注：SpinQuant 无 W4A16 列，4-8-16 最接近（A8 近似无损）。QuIP#(2402.04396) 以随机 Hadamard
不相关化 + E8 码本在 **2-bit 权重-only** 达 SOTA（7B/70B）——旋转对权重侧有效的最强证据，
但仅限 ≤2-bit 与大模型。

## 2. 关键发现
1. **旋转在 7B+ 实质修复 W4A16**（QuaRot A.4、SpinQuant Tab.4）。机制 = 计算不变性下的
   联合基变换（权重与激活同旋），权重侧离群被摊平后 4-bit 重建误差大降。
2. **随机 Hadamard 方差大**：SpinQuant 100 次随机旋转 0-shot 差 13 点（随机 Hadamard 6 点）；
   学习旋转（Cayley SGD，参数量 ~0.26%，权重冻结 = PTQ 类，同 AWQ 尺度搜索性质，不与 D13 冲突）
   稳定且更优。
3. **小模型例外困难**：1B 纯 GPTQ W4 已 +29% wiki PPL；旋转只收窄不消除（残差 +7.5%）；
   <1B 无 W4 成功案例，BBT 明示 W4 收益 ≈ 噪声；小模型更依赖 GPTQ 补偿（QuaRot Tab.3）。
4. **QK-norm/RoPE 阻碍 q/k 旋转**：QuaRot Stage1d 需在线 head-wise 旋转 Q/K；BBT 对 Qwen3-0.6B
   须替换 q_norm/k_norm 才可旋转。gate/up/down/o/v 可安全吸收式旋转。

## 3. 固定功能加速器落地成本
- **编译期（零硬件）**：SpinQuant-nohad 的 R1（残差基）/R2（V–O head 对）全部吸收进权重
  （W'←R·W 等），qbin 格式、dequant 通路、ISA 全不变；激活仍 BF16，无需任何变换算子。
  旋转 = 编译器内一次矩阵乘（qforge 侧），在线开销 = 0（W4A16 不做激活量化，无需在线 FWHT）。
- **在线（仅 W4A4/KV4 路径需要）**：QuaRot 每层 1.5 个 FWHT（O(d log d)，GPU 上并入 RMSNorm）；
  SpinQuant 在线 Hadamard 实测 ~8% 端到端延迟（M1 Pro，58.88→63.90 ms/token）。**W4A16 不触发**。
- **q/k 排除**：Qwen3 QK-norm（每通道）+ RoPE 使 q_proj/k_proj 不可吸收式旋转；方案 = 只旋转
  gate/up/down/o/v（0.6B 上 q+k ≈ 88M 参数 ≈ 总去重参数 15% 保持原样）。

## 4. 对 QCore 3/20 失败的适用性判定
- 我方失败面：权重侧对称 per-128-group，**无 GPTQ 误差补偿**；AWQ 仅 per-group scale 搜索。
  文献成功案例全部 = 旋转 + GPTQ（或 Cayley + GPTQ）。3/20 未必是「离群问题」——权重 7% rel
  误差均匀分布在 q/lm_head/gate（p5 §2.5），而旋转针对的是极值浓度。
- 外推期望：按 1B 数据，旋转+GPTQ 可把 wiki 劣化从 ~+29% 压到 ~+7–14%；token 一致率 3/20→?
  无文献数据（无人报告贪婪一致率），仅能推定「明显改善、未必达标」；按 BBT W4 消融，
  0.6B 量级旋转收益可能落在噪声内。
- 判定：**旋转不保证把 3/20 修到 ≥8/10**。它是「可能必要、不充分」的候选成分（应叠加 GPTQ
  补偿 + 更细分组 g64），且 <1B 证据薄弱。

## 5. 建议
- **反对**作为下一最优推进方向：W4 收益上限（权重流 317/190 tok/s 天花板，短 ctx 2×）受
  正确性风险支配，文献对 <1B 无正面证据；同等投入优先 R-KV（866→1009）与 R-Spec。
- **支持**一个受限低成本验证（条件：不占主线、≤2 周）：qforge 实现 SpinQuant-nohad 式
  R1/R2 吸收（q/k 除外）+ GPTQ g=128/64 + 对称 INT4，重跑 P1_PROMPT 20-token 一致率与 ΔPPL。
  **判据**：≥8/10 或 wiki ΔPPL ≤2% 才进路线图，否则归档 W4A16=backlog（与 audit.md 优先级
  ④「先解决 3/20 精度，再谈带宽」一致）。
- 若验证失败：W4A16 正式降级；权重侧带宽问题转由 batch>1（gen-2）与指令发射优化承接；
  本结论回注 roadmap.md。

## 参考
- QuaRot 2404.00456（Tab.1/3/7/8，§4 Stage1a–1d，A.4/A.5）；SpinQuant 2405.16406（Tab.1/2/4/6，§3.1–3.2/4.5）
- AffineQuant 2403.12544；QuIP# 2402.04396；GPTQ 2210.17323；AWQ 2306.00978
- BBT-spectral 2605.25203（Tab.1、W2/W4 消融）；2409.11055（IJCAI 2025）；SLMQuant 2511.13023；ParoQuant 2511.10645
- 任务清单中的「Slope」经 arXiv/Crossref/OpenAlex/S2 多拼写检索（2026-08-16）无法定位到可核验论文，未采用其数据。
