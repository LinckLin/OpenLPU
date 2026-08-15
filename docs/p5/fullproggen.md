# P5 波 1 — FullProgGen 交付说明（reg_class 同步 + 全模型程序生成）

> 节点：FullProgGen（P5 波 1）　计划依据：`plans/p5-plan.md` §3 FullProgGen（评审 3 轮达成一致）。
> 数值语义（VECTOR/KV）与全量数值验证分别归 ExecVecKv（波 1 并行）与 EndToEndRuntime（波 2）。

## 1. 交付物清单

| # | 交付物 | 路径 | 说明 |
|---|--------|------|------|
| 1 | reg_class 上游同步 | `compiler/mlir/qisa.td` | `config.class` → `reg_class`（C++ 关键字冲突，与 P4 已交付的 `qforge/mlir/include/QNN/QisaOps.td` 对齐） |
| 2 | 编码器字段同步 | `compiler/isa/isa.py` | `CONFIG` 字段名 `class` → `reg_class`（OPSPEC + 汇编/反汇编访问器） |
| 3 | 执行器访问器同步 | `qsim/executor.py` | `_exec_config` 的 `d["class"]` → `d["reg_class"]`（唯一一处，ExecVecKv 不触碰该路径） |
| 4 | 全模型程序生成 | `qforge/program.py`（新增） | `lower_transformer(mode, layouts)` 产出 0.6B 28 层完整 PF/DC 程序 |
| 5 | qbin 组装 | `qforge/build.py` | `build_model_qbin` 改用全模型 lowering；删除过时的 `build_programs` |
| 6 | M3 回归适配 | `qforge/verify_m3.py` | `full_model_checks` 改为结构检查（decode + 指令序列存在性） |

## 2. 全模型程序结构

`qforge compile` 产出 qbin：**141 张量不变**（28×5 投影 + lm_head，W8A8 INT8），
程序替换为全模型：

- **DC 程序**（686,530 条）：每层 `RMSNorm → QKV GEMV → QK-norm(per-head) → ROPE
  → KV.APPEND(8) → 每 KV head [KV.LOAD sel=both tile=2048 → BMM QK^T(batch=2,
  batch_stride_B=0) → VSCALE(×128^-0.5) → scores N-tiling(N≤128, 8K=64 tile)
  + 跨 tile online softmax(VREDUCE_MAX/SUM running，单组；per-head 标量广播的
  VSUB/VMUL 用 cv=C_BROADCAST，即 cv≠0 → ARb[0] 标量广播) → BMM AV(batch=2,
  batch_stride_B=0)] → O GEMV → 残差 → RMSNorm → gate/up GEMV → VSILU/VMUL →
  down GEMV → 残差`；每 _st tile 迭代开头重发 `CONFIG AR_KSTAGE/AR_VSTAGE`
- **PF 程序**（910,940 条）：同构 GEMM 版（M=128，KV.STORE_BLOCK 块写，单 block 窗口 128）；
  QK^T 前发 causal `VMASK` tile（row/col base=0，128×128）+ `VSCALE`+`VADD` 到 scores。
  **per-row softmax**：128 个逐行 max/sum 标量先按 16B 步距存（每行一条 VREDUCE，`cv=C_MASK`
  即 ngroups=1），再经 `VMUL` 广播（cv=C_BROADCAST）展开为 16384 元素缓冲，VSUB/VMUL 按
  cv=0 连续读（逐行值，不用 b[0] 广播顶替）；softmax 后的 AV GEMM 与 ctx 写出不变。
- **CONFIG 段**：`AR_KV_BASE`(AR63) / `C_KV_POS`(C30) / `C_SLAB_SHIFT`(C31) / eps / theta /
  activation-scale 描述符。

结构烟测（verify_m3 `_check_dc_structure`，前 2 层）：KV.LOAD sel=both 64、BMM batch=2
2048、N-tiling max=128、VREDUCE_MAX/SUM 各 2048、VEXP 4064、VMAX 2016。

## 3. 数值口径约定（波 2 细化）

- 权重 INT8 W8A8 + per-128-group BF16 scale（141 张量不变）。
- 激活 BF16；线性 = QUANT(BF16→INT8) → GEMM/GEMV(dequant=1) → VMOV（BF16 同型拷贝；
  dequant 输出已由 MATRIX 后处理落 BF16，`C_CC row_stride = N_TILE×2` 字节）。
- attention BMM srcA=srcB=BF16、acc=FP32（BF16 落盘，按「输出按 srcA dtype 落盘」统一口径）。
- 向量 op srcA=BF16、acc=FP32（内部 fp32、输出 srcA dtype）。
- 二元 op（VADD/VSUB/VMUL/VDIV/VMAX）按 **cv 字段**分发 ARb 语义：`cv=0` → ARb 连续
  len 元素；`cv≠0`（程序用 `C_BROADCAST=11`）→ ARb[0] 标量广播到 len（02-isa §7.2 补注）。

## 4. 需评审项

1. **RMSNorm gamma 缺失**：`input_layernorm / post_attention_layernorm / q_norm / k_norm`
   权重（约 85 个小 BF16 向量）不在 141 投影张量内；程序 ARb 指向全 1 向量占位。波 2 须
   将其补入 qbin 或由 qrun 注入（与「141 张量不变」约束冲突，需评审定案）。
2. **VMOV dtype 转换语义**：✅ 已裁决——MATRIX dequant 后处理直接落 BF16（04 §1.5），
   VMOV 是 BF16 同型拷贝（无 dtype 转换）；`C_CC row_stride = N_TILE×2` 字节。
3. **QUANT per-tensor scale 描述符**：镜像 CD 布局（`[20]=0` per-tensor、`[19]=0` BF16、
   `[18:0]` SRAM 字地址）；激活 scale sx=1.0（编译期常数）。
4. **ROPE pos 即时数**：decode 每 token 需更新，运行时除 C_KV_POS CONFIG 外还须补丁
   ROPE 的 `imm[15:0]`（或改由 C30 派生），波 2 定案。
5. **VSUB/VMUL 标量广播**：✅ 已裁决——二元 op 按 **cv 字段**分发（`cv=0` 连续 / `cv≠0`
   标量广播 ARb[0]）；DC/PF softmax 的标量广播 op 用 `C_BROADCAST`，连续 op 一律 cv=0。
6. **HBM 地址占位**：`INPUT_HBM / LOGITS_HBM / AR_KV_BASE` 为 v0 占位；qrun 波 2 重排
   （权重止于 ~579 MiB，logits/KV 区置于其后）。

## 5. 验收证据

| 断言 | 结果 |
|------|------|
| `qsim/test_isa_fields.py` | 11/11 通过 |
| `qsim/test_m2a.py` | 4/4 通过 |
| `qsim/test_vector_kv.py` | 30/30 通过（含 cv 字段级标量广播用例） |
| `python3 qforge/verify_m3.py`（qbin 重建后） | 退出码 0；141 张量 round-trip ✅；PF/DC 程序全解码 ✅；DC/PF 结构检查 ✅；6 类 × PF/DC 12 例全 PASS |
| `qforge compile` | 141 张量、PF 910,940 条、DC 686,530 条、容器合法 |
| `python3 qsim/timing.py` | 退出码 0；executor numeric consistency PF/DC `matches=True` |

## 6. wave-2 移交清单（EndToEndRuntime 必读）

波 2 运行时（qrun/QMetal）在提交 DC 程序前必须处理的补丁项：

1. **KV.LOAD 窗口长度补丁（关键，P5 wave-1 评审 P2-8）**：DC 程序保持固定窗口结构，
   每 KV head 每层发射 `4 × KV.LOAD(count=2048, pos_start=0/2048/4096/6144)`，静态覆盖
   `0..8191`。decode 阶段 cache 长度 `C < 8192` 时，运行时**每 token 按当前 `C_KV_POS`
   派生 C 并补丁**这 4 条 `KV.LOAD` 的 `count`（或等价地 `pos_start`）：第 `kt` 个 tile 的
   `count = min(2048, C - kt*2048)`，超出 C 的 tile `count=0`。否则零填充 HBM 的非法
   key（score=0）会进入 softmax 分母，稀释 ctx（严重性见 p5-plan §4 波 2 运行时）。
   程序本身不读运行时窗口长度寄存器（ISA v0 无该寄存器）。
2. **C_KV_POS / ROPE imm**：prefill 完成 = prefill_len；每 token 更新 DC 首条 CONFIG 的
   C_KV_POS，token 末 +1；ROPE 的 `imm[15:0]` 需同步补丁（§4 第 4 项）。
3. **RMSNorm gamma 注入**：input/post/q/k norm 的 gamma 权重需由 qrun 注入或补入 qbin
   （§4 第 1 项）；HBM 地址重排见 §4 第 6 项。
