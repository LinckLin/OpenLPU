# C-Limited：W4A16 GPTQ + SpinQuant-nohad 受限验证 —— 判据判定（归档）

> 2026-08-16 决策文档。输入：DECISION.md v2 §3.3「C 受限验证：GPTQ g64/128 + 编译期吸收；
> 判据 ≥8/10 或 ΔPPL≤2%；不过 → 归档（记录含 W4A8KV16 配置注与 ctx 限定）」。
> 口径：HF 原生 full-attention（`eager`），BF16 权重，fp32 log-softmax；golden 基线 =
> `docs/perf-research/quality-baseline/quality_baseline.json`（PPL@4K **32.20** / @8K **27.43** pooled，
> HellaSwag acc_norm **55%**）。
> **结论先行：全部配置双门槛否决，W4A16 归档 backlog。** 最优档（GPTQ g64 无旋转）ΔPPL
> 仍 **20.5%@4K / 19.1%@8K**（门槛 ≤2%），20-token 交叉一致 **3/20**（门槛 ≥16/20）；SpinQuant-nohad
> 旋转**不仅不修复、且恶化**（+rot 档 46–56%）。根因与文献一致：<1B 模型 W4 无成功先例，
> 且 Qwen3-0.6B 的 RMSNorm 权重离群使「折叠吸收」制造新离群（§3）。

## 0. 判据判定

| 判据 | 门槛 | 最优实测 | 判定 |
|---|---|---|---|
| 20-token 交叉一致（vs BF16 基线） | ≥ 8/10（16/20） | **3/20**（gptq64） | ❌ |
| ΔPPL（vs BF16 full，per-token pooled） | ≤ 2% | **19.1%**（gptq64 @8K） | ❌ |
| **总体** | 任一条达标即升级 | 双双不达标 | **归档 W4A16 → backlog** |

## 1. 方法与口径

- **量化对象**：全模型 0.6B 的 197 个线性投影（q/k/v/o/gate/up/down ×28 层 + lm_head；
  权重 [out,in]，embed 与 q_norm/k_norm 保持 BF16），对称 INT4（[-7,7]，`scale=max|w_group|/7`），
  group 轴 = K（输入维），scale 布局 `[N, K/g]` BF16 —— 与 qforge/qbin INT4 契约（
  `qforge.quant.quantize_weight_int4` / executor `_gemm_dequant`）逐字段同构，ISA/qbin 格式不变。
- **GPTQ**（`qforge/gptq.py`，2210.17323 Algorithm 1）：列优先（沿 K）逐列量化 + 逆 Hessian
  二阶误差补偿，blocksize=128，damp λ=0.01；Hessian 来自 **golden 校准集 = PG19 validation**
  （与 quality-baseline 同源，32 seq × 512 token，确定性 span `[512,1024)`）。
- **SpinQuant-nohad 编译期吸收**（`qforge/rot_quant.py`，2405.16406）：
  - **R1 残差流旋转（全局 R，1024×1024 随机 Hadamard）**：为跨 RMSNorm 保持精确等价，把
    RMSNorm 尺度参数折入紧随其后的权重（SpinQuant footnote 2），norm 退化为参数自由的单位范数
    算子（与旋转对易）。吸收式：`q/k/v←W·diag(w_in)·R`、`gate/up←W·diag(w_post)·R`、
    `o/down←Rᵀ·W`、`embed←W·R`、`lm_head←W·diag(w_final)·R`，输入/后/最终 norm 权重置 1。
  - **R2 V–O head 对旋转（每层每 KV head，128×128）**：`v←R_h·v`、`o 输入块←o·R_hᵀ`，精确无损
    （`A(V R_hᵀ)=(A V)R_hᵀ`，`c R_hᵀ(R_h O)=c O`）。
  - 旋转为**随机 Hadamard**（Sylvester + 随机 ±1 符号翻转，seed 控制），**未做 Cayley 学习**（
    SpinQuant-nohad 的学习旋转是训练型 PTQ，本验证保持免训练；记入 §4 局限）。
- **可复现性**：`qforge/rot_quant.py` 旋转 fp32 逐位无损（单层消融 max diff 6.7e-4）；BF16 原模型
  贪心解码 20/20 复现基线（harness 正确性锁定）；旋转未量化模型的 ΔPPL 仅 **0.38%/0.52%**（BF16
  再参数化噪声，远低于门槛）。

## 2. 结果（pooled PPL / ΔPPL / 20-token 交叉一致）

| 配置 | 交叉一致 | ΔPPL@4K | ΔPPL@8K | PPL@4K | PPL@8K |
|---|---|---|---|---|---|
| BF16 基线（参照） | 20/20 | — | — | 32.20 | 27.43 |
| rtn128（对称 RTN，控制组） | 9/20 | **44.21%** | **46.51%** | 46.43 | 40.18 |
| **gptq128**（无旋转） | 7/20 | 26.11% | 24.32% | 40.60 | 34.10 |
| **gptq64**（无旋转，**最优档**） | **3/20** | **20.51%** | **19.11%** | 38.80 | 32.67 |
| gptq128+rot（R1+R2） | 1/20 | 55.45% | 55.56% | 50.05 | 42.66 |
| gptq64+rot（R1+R2） | 9/20 | 46.13% | 45.28% | 47.05 | 39.85 |

- ΔPPL = `exp(mean per-token ΔNLL)−1`（O1 门槛定义，量化 vs BF16 full 逐 token）；cross-consistency
  = 量化模型贪心 20 token vs BF16 基线 token（P1_PROMPT，与 p5 `int4-results.json` 同 prompt 同 20 token）。
- **门槛判定：全部 ❌。** 最优 ΔPPL 19.1%（gptq64）仍 10× 于 2% 门槛；交叉一致最高 9/20（rtn128 /
  gptq64+rot），远低于 16/20。

## 3. 关键发现（逐条可审计）

1. **GPTQ 有效但远不足**：44%→26%（g128）→20.5%（g64），二阶补偿 + 更细分组方向正确（与
   QuaRot「小模型更依赖误差补偿」一致），但 0.6B 的 W4 本质掉点无法靠 GPTQ 抹平。对照文献：
   SpinQuant 1B GPTQ W4A8KV16 wiki +29%，本 0.6B GPTQ g64（W4A16）+20.5% 量级自洽（<1B 更差）。
2. **旋转不仅不修复、且恶化（44→55% / 20→46%）**，根因 = **RMSNorm 折叠制造新离群**：
   - Qwen3-0.6B 的 RMSNorm 权重含离群（`model.norm.weight` max **15.31**、layer-7 input norm
     max 2.13、post norm max 1.49）。R1 折叠把 `w_norm` 逐列乘入权重 → **lm_head max 0.318→2.0**、
     全模型权重 max 均值 0.407→0.625（峰值 1.23→6.06），对称 INT4 的组 scale 被离群主导，其余值被压碎。
   - 这解释为何 SpinQuant 在 **7B Llama**（norm 权重 ≈1，折叠不引入离群）有效、在 **0.6B Qwen3**
     （norm 权重有离群）反而恶化——文献「<1B 无 W4 成功案例」的机制之一。
   - **R2-only 消融**（V–O head 对，无折叠，精确无损 20/20）：gptq64+R2-only ΔPPL@4K(4 段子集)
     **25.5%**，与无旋转 gptq64（20.5%）同阶不改善——旋转本身（即使无损、无折叠）对 0.6B W4 无增益。
3. **交叉一致是混沌敏感指标，ΔPPL 是稳定判据**：gptq64 交叉一致仅 3/20 却是最优 ΔPPL；gptq64+rot
   9/20 却 ΔPPL 46%。贪心 argmax 对 BF16/量化扰动混沌敏感（O2 报告同述），故判据以 ΔPPL 为主、
   交叉一致为辅，二者在此均不达标。
4. **旋转未量化的 BF16 再参数化噪声本身很小（ΔPPL 0.4%）**：证明「旋转 fp32 精确等价」成立，
   恶化不是 BF16 舍入，而是折叠制造的结构性离群（§3.2）。
5. **与 p5 3/20 的差异（如实记录）**：本报告在 HF eager + fp32 softmax 参照上测对称 RTN=9/20、
   ΔPPL 44%；p5 在 qsim 执行器（额外 BF16 逐 op 舍入）测同配置 3/20。二者一致方向（W4A16 严重掉点），
   差异来自执行器数值路径；本报告隔离「纯量化误差」，对判据更干净。

## 4. 判定与去向

**W4A16 归档 backlog（如实，D5 INT4 数据通路已达成、部署质量如实记录不变）。** 依据：
- 双门槛（≥16/20 或 ΔPPL≤2%）**全部配置双双否决**，最优 ΔPPL 19.1% ≫ 2%，量级不在「再调参可救」区间；
- 旋转（SpinQuant-nohad，含免训练随机 Hadamard）**无益且恶化**，与 rot-quant.md §4 预期一致
  （「3/20 未必是离群问题、0.6B 旋转收益可能落在噪声内」）；
- **W4A8KV16 配置注**：SpinQuant 1B 的 W4「+29%→+7.5%」出自 **W4A8KV16**（激活 8-bit + KV 16-bit），
  **非 W4A16**；本验证为纯 W4A16（激活/KV 保持 BF16），未测 W4A8KV16——该配置需激活量化通路，
  超出本「权重-only、编译期零硬件」验证范围，若未来重启须单列。
- **短 ctx 限定**：rot-quant.md 的 W4 增益「短 ctx 2×（2417 tok/s）」仅对短上下文成立；长 ctx
  （4K/8K）被 KV 重读稀释（INT4 权重天花板 4K 938 / 8K 582，相对窗口化基线 866 仅 ×1.08/×0.67）。
  本报告质量否决在先，性能口径不变：**W4A16 的带宽收益本就不成立（长 ctx），质量又不过关，双杀归档**。

## 5. 局限（诚实记录）

- **Cayley 学习旋转未跑**：SpinQuant-nohad 正式版 R1/R2 是 Cayley SGD 学习（~0.26% 参数、权重冻结），
  本验证用随机 Hadamard（免训练）。文献「随机 Hadamard 方差大、学习旋转稳定更优」——但 §3.2 的折叠
  离群是**结构性**问题（与旋转矩阵值无关），学习旋转同样要折叠 norm 权重，预期同样制造离群，不改变判定。
- **校准集规模**：32×512=16K token（>5× 最大 K=3072，Hessian 满秩充分）；GPTQ 文献常用 128×2048，
  若校准不足可能低估 gptq 潜力，但 19% vs 2% 的量级差不受此影响。
- **仅 0.6B 实测**（D10 双尺寸义务）：8B 的 ΔPPL 未测；8B 亦含 QK-norm 与 RMSNorm，折叠离群问题预期同向。

## 6. 复现

```bash
cd /home/lzl/project/newlpu
# 全量（5 配置 + PPL + 交叉一致；输出 /tmp/w4a16-gptq-results.json）
.venv/bin/python3 -m qforge.w4a16_gptq --out /tmp/w4a16-gptq-results.json
# 快速（跳过 PPL）
.venv/bin/python3 -m qforge.w4a16_gptq --skip-ppl --n-calib-seqs 8
```

前置：HF 模型缓存 `Qwen/Qwen3-0.6B` + PG19（`emozilla/pg19`，自动缓存）+ golden 基线 JSON。
数值锚点（gptq64 19.11% / gptq128 24.32% / rtn128 46.51% @8K；交叉一致 3/7/9；旋转恶化 45–56%）见
`/tmp/w4a16-gptq-results.json`。实现：`qforge/gptq.py`、`qforge/rot_quant.py`、`qforge/w4a16_gptq.py`。

## 参考

- GPTQ 2210.17323；SpinQuant 2405.16406（§3.1 旋转参数化、footnote 2 RMSNorm 折叠）；QuaRot 2404.00456
- 决策上下文：`docs/perf-research/decision/{DECISION,rot-quant}.md`；基线：`docs/perf-research/quality-baseline/`
