# P5 执行计划（提案 v2，回应评审第 1 轮 9 条意见）

> v2 变更：①前置补生成 8K golden（wave-0 代理 Golden8K）；②M4 验收对齐 PLAN（1024 ctx 20 token
> + 4K 5 token + perplexity 等价覆盖 + 8K 单 token，8K cache 用引导方案）；③reg_class 上游同步
> 指派 FullProgGen 扩权 compiler/；④C_KV_POS 生命周期归波 2 运行时；⑤8K prefill 预算与引导兜底；
> ⑥scores N-tiling + online softmax 入 FullProgGen；⑦logits 比较口径明确；⑧INT8 参照措辞修正；
> ⑨stride=0 措辞与 P7 契约确认项。
> 状态：M0–M3 ✅。P5 = 软件线会师点（M4）。

## 1. 前提与已冻结契约

- qsim/executor.py 现实现 SYS/DMA/GEMM/GEMV/BMM；VECTOR 18 条与 KV 4 条数值语义归本节点。
- decode KV 路径 = KV.LOAD 单副本（tile≤2048，D16）；GQA 组内共享以 BMM batch_stride_B=0 模拟
  P7 内部广播总线——**ISA 未禁止 stride=0，语义公式 B[b]=ARb+b×batch_stride_B 在 stride=0 下
  有定义（02 §6.1），qsim executor 已按此实现**。P7 立项确认项：RTL 须接受 batch_stride_B=0
  （或 qbin 重生成）。
- embedding 在 host（D14）、采样在 host（D9）；qbin 输入 hidden、输出 logits。
- **8K 参照数据前置项**：P1 golden 无 cache=8192 采样点 → **wave-0 代理 Golden8K** 复用
  ref/gen_golden.py 补生成 decode cache=8192 单点 golden（含 lm_head logits；DECODE_CACHE_POINTS
  增 8192，只跑该点）。波 2 开始前必须完成。
- 性能预期（Python 参考执行器实测口径）：DC 全 GEMM 3.3 s/token；全量 PF 单 block（29,522 inst）
  ≈16 min。**8K prefill（64 blocks）不可行** → 8K 验收采用 KV cache 引导方案（见 §4）。
- logits 比较口径（全节点统一，numpy-vs-CUDA fp32 累加序差异下可达）：比较对象 = 参照 logits
  （bf16 量化，dtype_code=0；golden「logits」数组或 HF 现场同口径量化）；判据 = **逐元素 max abs
  ≤ 1 ULP（bf16 离散网格，~0.1% 元素可差 1 ULP 属正常）+ argmax 与参照一致 + 比较 span 的 NLL
  相对偏差 <1e-3（vs HF 现场参照）**。禁用「fp32 域 <1e-5」口径（不可达）。
- **INT8 参照**：与 P1 baseline（BF16 greedy）交叉一致率 ≥8/10 + 分歧位置 logits 相对误差报告
  （INT8 权重与 BF16 baseline 可合理分歧，不写「与 HF 逐 token 一致」）。

## 2. 分工（wave-0 + 两波）

| 波 | 代理 | 职责 | 依赖 |
|----|------|------|------|
| 0 | Golden8K | 补生成 decode cache=8192 golden（含 logits） | 无 |
| 1 | ExecVecKv | qsim VECTOR/KV 数值语义 + 字段/数值单元测试 | 无 |
| 1 | FullProgGen | qforge 全模型 PF+DC 程序生成 + **reg_class 上游同步（扩权 compiler/）** | 无 |
| 2 | EndToEndRuntime | qrun/QMetal 运行时 + M4 端到端验证 | 0+1 全部 |

## 3. 波 0/1 任务

### Golden8K（前置）
复用 ref/gen_golden.py：**seq/KV cache 构建长度提到 ≥8193 token**（cache 建到 8192 后仍有
1 token 可 decode，gen_golden.py 的 build_seq 长度与 cache 构建同步改），DECODE_CACHE_POINTS 增
8192 并只跑该点（全 op 目录 + lm_head logits），J1 schema 不变。验收：golden/qwen3-0.6b/ 出现
cache=8192 采样点且 meta.json 合法。

### ExecVecKv
1. qsim/executor.py 增 VECTOR 18 + KV 4 条数值语义（02-isa §7/§8 字段与 dtype/acc 约束）：
   RMSNORM（normal/per-head，eps=1e-6）、ROPE（θ=1e6）、VMASK、softmax 族、VSILU、VMAX/VMOV/VSCALE、
   QUANT/DEQUANT、KV.APPEND/STORE_BLOCK/LOAD（SLAB_SHIFT 公式）/GATHER（非默认保留）。
   数值：内部 fp32（acc=FP32），BF16 输入升格，输出按 srcA dtype 落盘；VEXP 用 numpy exp（功能级）。
2. qsim/test_vector_kv.py：VECTOR/KV 字段级断言 + 数值单元测试（与 P1 golden 单 op 对比，
   内部 fp32、落盘 bf16，判据 = §1 的 ULP 口径）。
3. 回归：test_isa_fields.py 11/11 保持绿。

### FullProgGen（含 reg_class 同步，扩权 compiler/ + qsim CONFIG 访问器）
1. **reg_class 上游同步（P4 评审遗留）**：compiler/mlir/qisa.td 的 config.class → reg_class；
   compiler/isa/isa.py 字段名同步；**qsim/executor.py 的 CONFIG 解析访问器（cls = d["class"]）
   同步改 reg_class（唯一一处，ExecVecKv 不动该路径，跨代理契约）**；qforge/mlir 副本回归一致；
   test_isa_fields.py 与 test_m2a.py 回归绿。
2. qforge/lowering.py 扩展全模型程序段（0.6B 28 层）：
   - decode（DC）每层：RMSNorm → QKV GEMV → QK-norm → RoPE → KV.APPEND → 每 KV head 组
     [KV.LOAD sel=both tile≤2048 → QK^T BMM batch=2 batch_stride_B=0 → **scores N-tiling
     （N≤128，8K 窗口 = 64 tile）+ 跨 tile online softmax（VREDUCE_MAX/SUM running 用法）**
     → AV BMM batch=2 batch_stride_B=0] → O GEMV → 残差 → RMSNorm → gate/up GEMV → VSILU/VMUL
     → down GEMV → 残差；末层 lm_head GEMV → DMA.STORE logits。
   - prefill（PF）同构 GEMM 版（block=128），KV.STORE_BLOCK 块写。
   - CONFIG 段：AR_KV_BASE/C_KV_POS/C_SLAB_SHIFT/scale 指针。
3. 验收（波 1 内）：qbin 全解码 + 前 2 层 DC 程序 smoke（含 N-tiling/online softmax 结构检查）；
   不跑全量数值（归波 2）。

## 4. 波 2 任务：EndToEndRuntime（qrun + QMetal，M4）

1. qrun/ 包：`python -m qrun <qbin> --prompt ... [--ctx 8192] [--max-new 20]`。
   - QMetal：HBM slab 分配、tensor 装载、命令队列（PF 1 次 → DC per token）、设备控制（qsim 后端）。
   - **C_KV_POS 生命周期**：prefill 完成 = prefill_len；每 token runtime 更新 DC 程序首条
     CONFIG 的 IMM（pos），token 末 +1（05 §5.2 步序）。
   - **8K cache 引导**：8K 验收用 P1 参考模型直接写 K/V 进 qrun HBM slab（跳过 8K PF 全量；
     引导脚本随 qrun 交付）；PF 数值路径以 prefill seq=128 block 与 golden 单独验证。
2. M4 验收（对齐 PLAN，不缩水；**prefill 预算：Python 执行器全量 PF ≈16 min/block → 多块 PF
   不可行，qrun 验收中 >128 token 的 prefill 一律用 KV 引导（P1 参考模型写 K/V 入 HBM slab +
   hidden 序列经 qrun 的 host embedding 路径写入），PF 计算路径以 prefill seq=128 block 与
   golden 单独验证；引导仅写输入数据，decode 计算仍为 qsim 全链路**）：
   - **BF16 短 ctx（真实 PF 路径）**：P1 baseline prompt（14 token，1 block PF 真跑）+
     ≥20 token 逐 token 与 P1 baseline 一致；
   - **BF16 中 ctx**：1024-token prompt（KV 引导）+ ≥8 token 逐 token 与 HF 现场一致 +
     span NLL 相对偏差 <1e-3 + logits 判据按 §1 ULP 口径——perplexity 等价覆盖；
   - **BF16 长 ctx**：4K（KV 引导）+ ≥5 token 逐 token 与 HF 现场一致；8K（KV 引导）+ 单 token
     logits 与 Golden8K 参照按 §1 口径（≤1 ULP + argmax 一致）；
   - **INT8**：≥10 token 与 P1 baseline（BF16）交叉一致率 ≥8/10 + 分歧位置 logits rel 误差报告。
   - 三级 golden 第 1-2 级达成报告入 docs/p5/。
3. 交付：qrun/ + docs/p5/m4-report.md（token 一致表、logits 误差表、执行时间、引导方案记录）。

## 5. 会师判定与后续

- M4 = 上述四条全过。M4 后：P6（硬件感知优化：KV staging 重叠、双缓冲、488 @4K 验证）
  与 P7（RTL，ISA 冻结已久）并行启动。

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| 8K prefill 小时级不可行 | KV cache 引导方案（P1 参考模型写 slab）；PF 数值以 128-block golden 验证 |
| 4K 5 token 超预算 | 参照 HF 现场；单 token 预算 2.5-5 min，超时降为 3 token 并报评审 |
| VECTOR 数值超容差 | 逐 op 单元测试先行；判据按 §1 ULP 口径；VEXP 用 numpy exp |
| 全模型程序错误难定位 | 分层 smoke：单层 DC → 3 层 → 全模型，logits 对比定位首个分歧层 |
| tokenizer 漂移 | 与 P1 同版本（4.51.0） |
| reg_class 同步破坏 P4 已交付 | 改后跑 qforge 回归（verify_m3.py 需保持绿） |

## 7. 需评审关注点（第 2 轮）

1. 9 条意见是否全部落实且无新冲突；
2. M4 四条判定是否等于 PLAN 判定（4K/8K ctx + perplexity 等价覆盖）且可执行；
3. 8K 引导方案的边界（引导 = 参考模型写 KV，验证对象仍为 qsim 计算路径）是否合理；
4. 若一致，请声明"评审一致，可执行"。
