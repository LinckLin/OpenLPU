# P1+P2 并行执行计划（提案 v3，已获批执行；v3 响应项目定义更新）

> v3 变更（响应 2026-08-13 项目定义更新）：①目标模型改为 Qwen3-0.6B（验证优先，放量 4B/8B）；
> ②命名正式化：qnn dialect（QNN）/ qisa dialect（Q-ISA）/ qsim（原 isa_ref，P3 种子）/ QCore 平台；
> ③spec §4 数值标注为 8B 设计上限口径，0.6B 流量按形状重算。
> v3.1（评审复评修复）：Wq 形状修正 2048×1024；J1 示例 q 形状 2048；GQA 2:1；DDR/P9/P6 口径同步。
> v2 变更（评审第 1 轮 8 条）：M2 拆 M2a/M2b；J1 完整 schema；M2a 矩阵 {PF,DC}×{BF16,INT8}；
> P1.2 补全模型路径；容差按 dtype；dequant 路径；MLIR fallback；BF16 落盘自洽。
> 目标：P1（Qwen Reference）与 P2（Q-Compiler 骨架）并行推进，本计划交付 M2a；M2b 归 P3（qsim 时序）。

## 1. 背景与约束

- P0 已冻结：`docs/spec.md` + `docs/spec-src/00–05`。所有数字口径、Q-ISA 编码、KV 协议、容器格式以 spec 为准。
- P1/P2 并行前提：无文件/依赖交叠；共同下游是 P3（qsim，M2b）。
- 冻结口径：INT8 32.77 TMAC/s、HBM sustained 读 720 GB/s、SRAM 写 256 B/cyc、tile M/N ≤ 128、K 流式。
  **注意**：spec §4 的 decode 7.57 GB/token、KV 147,456 B/token 等是 **8B 设计上限口径**；
  0.6B 的逐层流量按 0.6B 形状重算（DC 模式 27.3× 带宽短缺与阵列利用率结论与模型尺寸无关，仍成立）。

## 2. 共享接口契约

### J1 golden trace 格式（完整 schema，冻结）

- 目录：`golden/qwen3-0.6b/<op_name>/`。op_name 命名：`L{layer:02d}_{op}`（层内 op），
  全模型级为 `embed` / `final_norm` / `lm_head`；M2a 专用：`linear_wq_pf`、`linear_wq_dc`。
- 每目录文件清单：
  - `inputs.npz`：该 op 输入（激活一律 BF16；`embed` 例外——输入为 token id，int32，
    dtype 以 meta.json 的 `dtype_np`/`dtype_code` 为准）；
  - `outputs.npz`：该 op 输出激活（BF16）+ 期望结果 `y_ref`（FP32，供容差比较）；
  - `weights.npz`（仅 `linear_wq_pf` / `linear_wq_dc`；embed/lm_head 权重按总原则不落盘，
    `weights_ref` 记名，P4 直接读 safetensors）：该 op 权重切片。
- 权重总原则：**全部层权重不入 golden**（避免重复落盘，P4 管权重）；仅 M2a 会师用的
  linear 层权重以 `weights.npz` 单独保存。
- `meta.json` schema（字段名与类型冻结）：
```json
{
  "op": "attn_qkv",                 // 见 op 枚举表
  "layer": 3,                       // 全模型级 op 为 null
  "mode": "PF",                     // PF=prefill | DC=decode
  "inputs":  {"x":  {"shape": [128, 1024], "dtype_np": "bfloat16", "dtype_code": 0}},
  "outputs": {"q":  {"shape": [128, 2048], "dtype_np": "bfloat16", "dtype_code": 0},   // 0.6B: q=16h×128=2048, k/v=1024
  "params":  {"eps": 1e-6, "theta": 1000000.0, "pos": 0, "seq": 128, "attn_scale": 0.08838834764831845},
  "weights_ref": ["wq", "wk", "wv"] // 引用 P1 已加载权重的名字，不落盘
}
```
- dtype 映射表（冻结，与 02-isa §2.2 一致）：
  `bfloat16↔0, float16↔1, int8↔2, int4↔3, int32↔4, int16↔5, float32 仅作 y_ref 比较用，不编码`。
- BF16 落盘规则：numpy≥2.1（原生 bfloat16）或安装 ml_dtypes；torch bfloat16 → 原生转存；
- op 枚举（P1 必须全覆盖）：`attn_qkv, attn_qknorm, attn_rope, attn_score(QK^T), attn_softmax,
  attn_ctx(AV), attn_o, residual_attn, rmsnorm_in, rmsnorm_mlp(pre-MLP), mlp_gate, mlp_up,
  mlp_silu, mlp_down, residual_mlp`（前两者合并时记 `attn_qkv` 一个 op）+ `embed, final_norm, lm_head`。

### J2 qbin 格式

直接按 00-container §2 完整实现（magic/header JSON/tensors 表/.pf_program/.dc_program），
P2 产出最小合法 qbin（单 linear 层，PF 与 DC 两段程序各一）。P3/P4 共用同一解析器，不做中间格式。

### J3 qsim 功能核心（P3 种子）

`qsim/executor.py`（Python）：数值精确执行最小子集
`CONFIG/BARRIER/WAIT/MODE/DMA.LOAD/DMA.STORE/GEMM/GEMV/BMM`，GEMM 族支持：
`acc_init、transpose_A、transpose_B、bsrc、dequant=1 + CD per-128-group scale 描述符
（INT32 组内累加 → fp32 scale → fp 域组间累加，与 spec 02 §6 / 04 §1.5 一致）`。
P3 在此执行器上加时序模型（M2b），不改数值语义。

## 3. P1 计划（2 周）— Qwen Reference（Qwen3-0.6B）

### 任务分解
1. **P1.1 环境与基线**：torch + transformers==4.51.0（与 01-target-model 核验版本一致）；
   numpy≥2.1 或 ml_dtypes；经 hf-mirror 下载 Qwen/Qwen3-0.6B（BF16 约 1.2 GB）；
   **先 read config.json 核验**：预期 hidden=1024、layers=28、q_heads=16、kv_heads=8、
   head_dim=128、intermediate=3072、vocab=151936、rope_theta=1e6、rms_eps=1e-6、QK-norm 有；
   核实 tie_word_embeddings（0.6B 通常 tied=true）；跑通 HF 官方 greedy decode 基线。
2. **P1.2 全模型手写参考**：`ref/model.py` 覆盖完整前向路径——token embedding 查表、
   28 层 decoder layer（输入 RMSNorm(eps=1e-6) → QKV 投影(4096×1024) → per-head QK-norm(128)
   → RoPE(theta=1e6) → causal attention(scale=128^-0.5) → O 投影 → 残差 → pre-MLP RMSNorm →
   gate/up(3072×1024) → SiLU⊙up → down(1024×3072) → 残差）、末层 final RMSNorm、lm_head
   （tied 视 config 核验：tied=true 则与 embedding 共享权重）、host 侧 greedy argmax 采样器
   （D9：采样在 host，ISA 无 argmax）。与 HF 逐 op 对比：每 op max abs diff < 1e-3（BF16 量级）。
3. **P1.3 golden trace**：prefill（seq=128，一个 block）与 decode（seq=1，cache 0→4K 抽样
   若干点）各一套，覆盖全部 op 枚举；按 J1 schema 落盘。**优先**生成 `linear_wq_pf`（seq=128）
   与 `linear_wq_dc`（seq=1）两套（0.6B Wq **2048×1024**，含 weights.npz + y_ref），完成后立即
   经 hub 通知 CompilerP2。
4. **P1.4 roofline**：按 spec §4 方法对 0.6B 重算每 op FLOPs/Bytes → `docs/p1/roofline.md`；
   decode 每 token 权重读 = 28 层 dense + lm_head（视 tied，tied 时 lm_head 即 embedding 转置
   共享，仍须全读）按 INT8 1B/参数计；与 8B 设计上限（7.57 GB/token）的关系注明；
   prefill/decode 逐层瓶颈表。

### 验收（M1）
- greedy decode ≥20 token（固定 prompt，与基线同一 prompt）与 HF 逐 token 完全一致；
- 逐 op 对比误差表（max abs diff < 1e-3）写入 docs/p1/；
- roofline 表逐层给出瓶颈结论（decode = HBM 权重流主导，方法与 spec 04 §2.4 一致，数值按 0.6B）。

## 4. P2 计划（3 周）— Q-Compiler 骨架（交付 M2a）

### 任务分解
1. **P2.1 MLIR 工程骨架**：方言 C++（TableGen + passes），Python 绑定驱动。工具链路线：
   ① 优先 LLVM 官方 snapshot wheel（GitHub llvm-project releases 的 snapshot-build 含
   mlir_core 绑定）；**拿到后第一步验证 Python 绑定存在**；② 绑定缺失则源码编译（仅 MLIR
   子项目：`LLVM_ENABLE_PROJECTS=mlir -DMLIR_ENABLE_BINDINGS_PYTHON=ON -j64`）；
   版本一经选定写死 `compiler/README.md`。
2. **P2.2 qnn dialect（QNN 层）**：`qnn.matmul / attention / rmsnorm / rope / swiglu`
   （语义对齐 spec 02 §12）。
3. **P2.3 qisa dialect（Q-ISA 层）**：与 02-isa 33 条指令一一对应（GEMM/GEMV/BMM/DMA/Vector/KV/SYS）。
4. **P2.4 Q-ISA 编码器**：128-bit 字段布局（02-isa §2/§4–§8）assembler + disassembler +
   qbin writer（00-container §2，含 header JSON）。
5. **P2.5 lowering**：qnn → qisa（GEMM 序列；tiling 遵守 N≤128、K 流式、M=seq≤128）
   → ISA asm；linear 层在 PF 出 GEMM 程序、DC 出 GEMV 程序（MODE 指令包裹）。
6. **P2.6 qsim 功能核心**：按 J3，目录 `qsim/`。
7. **P2.7 M2a 验证**：验证矩阵 {PF, DC} × {BF16, INT8} 共 4 例，输入/权重/期望取
   golden `linear_wq_pf`/`linear_wq_dc`（0.6B Wq **2048×1024**，16 个 N=128 tile）：
   - INT8 例权重：测试脚本对 Wq 做一次对称 per-128-group 量化生成（**仅验证数据，非 P4 产物**）；
   - 容差（比较层级写明）：INT8 → INT32 累加与参考**逐位一致**，dequant 后 fp32 输出 <1e-6；
     BF16 → 在 FP32 累加结果上比较 <1e-5；BF16 舍入落盘后 ≤1 ulp。

### 验收（M2a，本计划交付）
- {PF,DC}×{BF16,INT8} 四例数值通过；qsim 语义与 spec 字段级断言通过；
- qbin 容器可被 qsim 加载执行。**M2b（P3 时序模拟器跑通 layer trace）不在本计划范围**。

## 5. 并行契约与文件布局

- P1 写 `ref/`、`golden/`、`docs/p1/`；P2 写 `compiler/`、`qsim/`、`docs/p2/`。
- 共同只读：`docs/spec.md` + `docs/spec-src/`、`PLAN.md`、`plans/p1-p2-plan.md`（J1–J3 契约）。
- 会师 M2a：P2 的 qsim 执行 P1 的 `linear_wq_pf`/`linear_wq_dc` golden → 数值一致 = M2a 达成。

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| MLIR Python 绑定缺失 | §4 P2.1 双路线：snapshot wheel → 验证绑定 → 缺失则 MLIR-only 源码编译 |
| 权重下载 + golden 磁盘 | 0.6B 仅 ~1.2 GB；先 `df -h`；golden 只存激活（J1 权重不落盘原则）+ linear 例外 |
| 网络下载失败 | hf-mirror（P0 已验证）；断点续传 |
| numpy 无原生 bfloat16 | numpy≥2.1 / ml_dtypes；降级转 fp32 + meta 标注（J1 规则） |
| P1/P2 对 spec 理解漂移 | 任务提示词要求口径引用 spec 原句；M2a 会师天然校验 |
| P4 量化管线混淆 | INT8 验证数据明确标注"测试侧临时量化，非 P4 产物" |
| 0.6B 与 8B 口径混淆 | §1 已标注；P1.4 注明与设计上限的关系 |

## 7. 变更记录

| 轮次 | 变更 |
|------|------|
| v1→v2 | 评审第 1 轮 8 条：M2a/M2b 拆分、J1 schema、PF+DC 双套、全模型路径、dtype 容差、dequant 路径、MLIR fallback、BF16 落盘 |
| v2→v3 | 项目定义更新：目标模型 Qwen3-0.6B（验证优先）、命名 QNN/Q-ISA/qsim/QCore、spec 8B 数值标注为设计上限 |

## 8. 评审状态

- 第 1 轮：DISAGREE（2 major + 6 minor）→ v2 全修。
- 第 2 轮：1 minor（embed 输入 dtype 措辞）→ 已修。
- 第 3 轮：**AGREE，评审一致，可执行**（2026-08-13）。
- v3 变更（响应项目定义更新）已随重定向消息同步执行中两代理；本文件为同步后的文字固化，待复评。
