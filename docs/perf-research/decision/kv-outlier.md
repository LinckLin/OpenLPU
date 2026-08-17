# 低比特 KV 离群值处理调研 — KIVI/KVQuant/GEAR/attention-aware vs QCore QK-norm 事实

> 结论：文献四派的一致处方 = **K per-channel、V per-token**（与 O2 实测互证）。对我们 k_norm
> 2784×–180859× 离群的迁移关键不在"更聪明地处理离群元数据"，而在**把静态 k_norm 从被量化张量中折出**
> （= KVQuant pre-RoPE 静态 per-channel scale 的免校准特例）。INT4 KV 若兑现（以 on-read RoPE 专用旋转器为条件）：**866→≈1096 tok/s
> （1.27×，距权重墙 1208 仅 9.2%）**；INT8 全量（需折叠）866→1009；INT3 再往上仅 +2.3%，不值。
> 反对：GEAR 在线 SVD（每 20 token）、QAQ 逐 token 位宽自适应+CPU 回拷（固定功能单核不可行）；
> KIVI FP16 残差窗照搬（与 O1 窗口化重叠，且 passkey 0.70–0.76 弱于 KVQuant 4-bit 的 1.0）。
> 文献全部证据在 7B+ 模型，0.6B 无先例——INT4 必须重跑 O2 数值门槛。

## 1. QCore 事实锚点（docs/perf-research/o2-kv-int8.md，0.6B 实测）

| 事实 | 数值 |
|---|---|
| K post-RoPE 通道幅值 | max 范围 [1.95, 458]，通道 RMS 比 **~296×**；pre-RoPE 更甚 ~2472× |
| k_norm 静态权重通道比 | **2784×–180859×**（layer 0–7；q_norm 5027×）——离群源是**静态权重**，非随机激活 |
| V 通道比 | ~2.4×（干净） |
| per-head 对称 INT8 K+V | ΔPPL 4.543%/5.196% ❌ 否决；仅 V INT8 0.641%/1.756% ✅（866→930.8） |
| per-channel K（动态 scale/token） | ΔPPL 0.701%/1.776% ✅ 但 scale 元数据 2048 B/token = 数据 2×，零净收益 |
| per-channel K 静态 scale（post-RoPE 128 token 校准） | ΔPPL 66.9% ❌（RoPE 使通道幅值 token 相关，静态 scale 不泛化） |

## 2. 文献关键数据表（全部 7B+；无 <1B 证据）

| 方案 | 离群机制（元数据形态） | 质量数字（论文原值） | 系统/硬件成本 | 对 QCore 的可迁移部件 |
|---|---|---|---|---|
| **KIVI** 2402.02750 | K per-channel、V per-token；token 向 G=32 分组共享 scale（asym：scale+zp）；最近 R=128 token 存 FP16 残差窗 | 2-bit：CoQA 63.88→63.05、GSM8K 13.50→12.74（Llama-2-7B）；LongBench 均值 44.52→44.27；峰值内存 2.6×↓、批 4×、吞吐 2.35–3.47×（A100） | G=32 分组使 K scale 摊薄到 ~128 B/token（asym）；残差窗 = 滑动 FP16 读写 | K/V 不对称处方（与 V 干净/K 脏互证）；分组 scale = 动态 scale 的可压缩形式（折叠前的备选）；残差窗勿照搬（见 §3.5） |
| **KVQuant** 2401.18079 | **pre-RoPE** K per-channel（离线 16×2K 样本校准的**静态** scale）；V per-token 在线；NuQ 非均匀 LUT；per-vector 1% 离群存 CSR 稀疏；首 token FP16（attention-sink-aware） | **4-bit ΔPPL <0.02、3-bit <0.1、2-bit <0.5**（WikiText-2，LLaMA/Llama-2/Llama-3/Mistral 全家族）；内存省 3.7×/4.8×/6.9×；passkey 4-bit-1% = 1.0@2K–32K；RULER 3-bit 53.65 vs fp16 56.40 | batch=1 kernel：K/V matvec 4-bit 比 fp16 快 1.3–1.7×（A6000，l=16K：Key 219.4→126.3μs）；静态 K scale 2 KB/layer 共享全序列；NuQ=16 项 LUT 查表 dequant | **pre-RoPE + 静态 per-channel scale = 我们的 QK-norm 折叠**（§3.3）；V 在线 per-token 量化 = KV.APPEND 一次 max-abs（微硬件）；sink-FP16 与 S=4 sink 结构零成本同构 |
| **GEAR** 2403.05527（context 给的 2403.01927 是错误 ID，该号是肿瘤基因选择论文） | 量化骨干（KCVT/KIVI）+ 低秩 r=4 补偿量化误差 + s=2% 稀疏离群；流式缓冲每 n_b=20 token 做一次 SVD（幂迭代） | 4-bit 近无损（KV 剩 31.0%）；2-bit GSM8k-CoT LLaMA3-8B 47.83 vs FP16 48.69（KIVI 仅 28.82）；峰值内存 2.39×↓、吞吐 2.1–5.07×（V100，批 3→18） | **在线 SVD/稀疏提取/双路径低秩前向**——固定功能单核不可行；低秩+稀疏只在 2-bit/硬任务必需 | 证据价值：4-bit 近无损在 7B+ 可达；机制不可迁移 |
| **Attention-aware**：KVQuant §3.5；QAQ 2403.04643 | QAQ：K/V 灵敏度理论推导（K 误差→softmax 方差 ∝ Q 范数；V 误差 ∝ 1/注意力分数）→ 逐 token 自适应位宽 + 1% 离群 FP16 + attention window(5) 防重要性突变 | LLaMA-2-7B：HellaSwag/PIQA/MathQA 压缩 7.5–8.0×（<1% acc 降）；13B 达 9.0×；比 SOTA（Scissorhands/H2O ~3–5×）高 1.6–1.8×；无离群处理 acc 降 12–26%，1% 离群只多 4% 开销 | 需 CPU 保存未量化 KV + 运行时位宽自适应重量化（不可逆）——单核不可行 | 证据价值：1% 离群元数据即够；attention sink FP16 保留与 O1 天然同构 |

## 3. 与 QK-norm 离群事实的对照判定

1. **per-channel 处方直接有效**：我们 per-channel-K 动态 0.701%/1.776% 过门槛 == KIVI/KVQuant 核心结论（K 固定通道大值必须 per-channel 隔离，KIVI Table 1：2-bit K per-channel+V per-token 是唯一可用的 2-bit 配置）。卡点在元数据压缩，不在数值。
2. **静态化失败机理与 KVQuant 论断一致**：我们 post-RoPE 静态 per-channel 66.9% ↔ KVQuant §3.2 "RoPE 使配对通道按不同位置混合、通道幅值随 token 变化"→ 静态 scale 必须配 **pre-RoPE 量化点**，二者互为独立验证。
3. **k_norm 折叠 = KVQuant pre-RoPE 配方的免校准特例**：Qwen3 K = k_norm ⊙ rmsnorm(k_raw)（单位 RMS 向量）。把 k_norm 折入 dequant scale 后，量化对象 = 单位范数向量（O2 §7 估幅值有界 ≈[0,11]，无通道离群）→ per-head INT4 都可行；scale_c = k_norm[c]×s_q 是**模型参数**（带符号 BF16 scale；实测 0.6B 3.4% 负通道 min −3.86，dequant 按带符号 scale 乘；4B/8B 放量时重检），无需 KVQuant 的 16×2K 校准集。KIVI G=32 分组 scale（64–128 B/token）是折叠落地前的备选压缩形式。
4. **V 侧无争议**：三篇一致 V per-token；我们 V ~2.4× 干净、INT8 已近乎无损（重建误差 0.0088）→ INT4 V 风险低（KVQuant 4-bit 全家族 ΔPPL<0.02）。
5. **反对直接引入**：GEAR 在线 SVD（每 20 token 幂迭代）、QAQ 运行时位宽自适应（需未量化 KV 回拷，且量化不可逆需 CPU 副本）——与单核固定功能、batch=1、编译期量化哲学冲突；KIVI 残差窗与我们 W=2048 窗口重叠（残差窗只保最近 128 token，KVQuant Table 2 显示其 passkey 0.70–0.76 显著弱于 nuq4-1% 的 1.0，全上下文检索弱）。

## 4. 增益量化：866→?（qsim 冻结模型，W=2048 R=2052，overlap 口径；先逐位复现 O2 六行 866.0/1008.8/930.8/257.3 再外推）

| 配置 | KV B/token/layer | tok/s | 相对 866 |
|---|---|---|---|
| windowed BF16（O1 基线） | 4096 | 866.0 | 1.000× |
| V-only INT8（已过门槛） | 3088 | 930.8 | 1.075× |
| INT8 K+V（需折叠） | 2048 (+32 scale) | 1008.8 (1006.2) | 1.165× |
| **INT4 K+V + per-head scale 32B** | 1024+32 | **1096.3** | **1.266×** |
| INT4 + K 静态 scale 折入 k_norm（SRAM 常驻 2 KB/layer，共 56 KB） | 1024+16 | 1097.9 | 1.268× |
| INT4 + KIVI 式 per-channel 分组 scale（G=32） | 1024+80 | 1091.7 | 1.260× |
| INT4 + KIVI 残差窗 R=128（FP16） | 1290.6 均摊 | 1074.3 | 1.240× |
| INT3 + per-head scale（数据口径，需 3-bit 解包器） | 768+32 | 1121.5 | 1.295× |
| **权重墙（KV=0 渐近，B1）** | 0 | **1207.9** | 1.395× |

- **INT4 KV（含元数据）866→1096**（**以 on-read RoPE 专用旋转器为条件；无旋转器口径 ≈4× 旋转开销吞噬收益，须同列**）；INT8→INT4 增量 +8.7%；距权重墙仅 **9.2%**——B1 是下一道墙，KV 再压缩收益归零。
- INT3 相对 INT4 仅 +2.3%，还需 3-bit 解包器/非均匀 LUT，**不值**；INT2 在 7B+ 已需低秩+稀疏兜底（GEAR），单核不可行。
- B6 指令发射墙 ≈1386 tok/s（721,895 inst/token）在 INT4 档（1096）仍不绑定。
- 前提：以上数字按"质量门槛通过"计；INT4 质量在本项目尚未重验（文献 4-bit 近无损证据全在 7B+，见 §2）。

## 5. 方向建议

- **支持 ①（终态）INT4 KV + QK-norm 折叠**：量化点移到 k_norm 相乘前（单位范数 K），k_norm 作为静态 per-channel dequant scale（8×128×2B=2 KB/layer，SRAM 常驻；若每 token 读 ≈56 KB/token×866 tok/s≈48 MB/s，可忽略）；V 保持 per-head INT4（APPEND 时在线 max-abs，微硬件）。先过数值门槛（ΔPPL≤2% + 量化 vs windowed-BF16 隔离交叉一致），再动 RTL。天花板 866→**1097（≈1.27×）**。
- **支持 ②（近端，零风险）V-only INT8**：已过门槛（0.64%/1.76%），866→930.8，RTL 最小（V 地址步长+scale），INT4 立项前的确定收益。
- **支持 ③（零成本附加）sink-FP16**：KVQuant attention-sink-aware 与我们 S=4 sink 结构同构——sink token 的 KV 保持 FP16，sink 段本来就每步重读，无额外带宽，且 2-bit 档文献显示收益最大。
- **反对**：GEAR 在线 SVD；QAQ 位宽自适应+CPU 回拷；KIVI 残差窗（与 O1 重叠、全上下文检索弱）；INT3（+2.3% 不值）；INT2（需低秩/稀疏在线机制）。
- **记录**：context 所列 GEAR ID 2403.01927 有误，正确为 **2403.05527**；"Attention-aware KV 量化"未给 ID，本报告以 KVQuant §3.5 + QAQ(2403.04643) 覆盖。
