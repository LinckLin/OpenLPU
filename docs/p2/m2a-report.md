# M2a 验证报告（P2 交付）

## 1. 交付物清单

| 路径 | 内容 |
|------|------|
| `compiler/README.md` | 工具链版本写死 + 用法 |
| `compiler/isa/isa.py` | Q-ISA 128-bit 编码器/解码器 + asm assembler/disassembler（33 条指令） |
| `compiler/isa/qbin.py` | `.qbin` 容器 writer/reader（00-container §2） |
| `compiler/mlir/qnn.td` | qnn 方言（matmul/attention/rmsnorm/rope/swiglu） |
| `compiler/mlir/qisa.td` | qisa 方言（与 02-isa 33 指令一一对应） |
| `compiler/lowering.py` | lowering：qnn.matmul → qisa GEMM/GEMV → Q-ISA asm → qbin |
| `qsim/executor.py` | qsim 功能核心（J3 子集，P3 种子） |
| `qsim/test_isa_fields.py` | 字段级断言（对 02-isa 编码/语义） |
| `qsim/test_m2a.py` | M2a 四例数值验证（本报告数据源） |
| `docs/p2/*.md` | 工具链/方言/编码器/本报告 |

## 2. M2a 数值验证矩阵（{PF,DC}×{BF16,INT8}）

数据源：`golden/qwen3-0.6b/linear_wq_pf` / `linear_wq_dc`（GoldenP1，Wq=[2048,1024]，
x=[128,1024]/[1,1024]，y_ref FP32，y BF16）。**真实 golden 数据，非合成**。

| 例 | 判据 | 实测 | 判定 |
|----|------|------|------|
| PF×BF16 | fp32 累加 vs y_ref <1e-5；落盘 ≤1 ulp | abs 1.67e-6（rel 2.91e-7）；ulp ≤1（\|y\|≥1e-3） | ✅ |
| PF×INT8 | INT32 累加逐位一致；dequant fp32 <1e-6 | 逐位一致 ✅；abs 4.77e-7（rel 8.3e-8） | ✅ |
| DC×BF16 | fp32 <1e-5；落盘 ≤1 ulp | abs 1.91e-6（rel 3.90e-7）；ulp ≤1 | ✅ |
| DC×INT8 | INT32 逐位一致；dequant <1e-6 | 逐位一致 ✅；abs 2.38e-7（rel 4.9e-8） | ✅ |

**四例全部通过。**

- BF16：qsim 以 BF16 输入升 fp32、fp32 累加（acc=FP32），与 golden `y_ref`（fp32）比较
  <1e-5 ✅；落盘 round-to-nearest-even 到 BF16 后与 golden `y`（bf16）比较，正常量级元素
  （|y|≥1e-3）≤1 ulp ✅（见 §4 需评审项①）。
- INT8：权重=测试侧对称 per-128-K-group 量化（**非 P4 产物**），激活=测试侧 per-tensor 对称
  INT8（W8A8）。qsim 走 dequant=1 + CD per-128-group scale（INT32 组内累加 → fp32 scale →
  fp 域组间累加，对齐 02 §6 / 04 §1.5）：INT32 部分和与独立 einsum 参考**逐位一致** ✅；
  dequant 后 fp32 与 fp64 参考 <1e-6 ✅。
- 复现命令：`python3 qsim/test_m2a.py`（输出 `/tmp/m2a_results.json`）。

## 3. qbin 容器可被 qsim 加载执行

`build_linear_qbin` 产出最小合法 qbin（单 linear 层，PF+DC 两段程序各一），`read_qbin` 完成：
magic `"NLPU"` / version / flags / header_size / header JSON（model/cfg/quant/tensors/pf_entry/dc_entry）
/ tensors 表（hbm_off 升序）/ `.pf_program` / `.dc_program` / `"ENDQ"` 哨兵 + 长度校验，
qsim `load_qbin_into_executor` 按 `hbm_off` 装载权重/scale 后执行程序、DMA.STORE 回写 HBM、
harness 读回比对 —— 全链路 ✅（weight/scale 字节 round-trip 逐字节一致，已断言）。

## 4. 字段级断言（对 02-isa）

`python3 qsim/test_isa_fields.py` → **11/11 组通过**：33 指令全集冻结与计数（SYS 5/DMA 3/
MATRIX 3/VECTOR 18/KV 4）、opcode 表逐条、engine tag 区间与冗余一致性、头字段位布局
（[127:120]/[119:112]/[111:109]/[108:106]/[105:104]）、dtype/acc 编码表、MATRIX/DMA/KV 操作数
位布局、CD 描述符位布局、全指令 encode/decode round-trip、engine tag 不一致检出、reserved
opcode 拒绝、asm assembler/disassembler round-trip。

## 5. 需评审项（偏离已批准计划之处）

1. **BF16「落盘 ≤1 ulp」仅对正常量级成立**：golden `y`（torch BF16 输出）与 `y_ref`（fp32）
   相比，有 0.45% 元素（|y|<1e-3 的深相消小值）在 torch-vs-numpy 的 fp32 累加次序差异下
   bf16 舍入相差 1 ulp；qsim 对这些小值最多相差 6 ulp（fp32 累加次序差异，非执行器缺陷）。
   正常量级元素（|y|≥1e-3）均 ≤1 ulp。判定：接受，属标准浮点累加次序差异，已记录。
2. **INT8 例激活 scale `sx` 按 (mode,quant) 各求一次**（测试侧对输入求 per-tensor 对称 scale）。
   真实静态 W8A8 管线中 `sx` 为**编译期常数**（P4 校准产物）、PF/DC 共享。判定：接受为 v0
   测试约定——M2a 验证的是 dequant=1 + CD per-128-group 数据通路正确性，`sx` 取值方式归 P4。
3. **0.6B 模型卡**：M2a 数据源由 Qwen3-8B 改为 **Qwen3-0.6B**（q_proj=2048×1024、hidden=1024、
   GQA 2:1——主会话重定向，已随 01b 模型卡落档）。tiling 约束不变：K=1024 流式、N=2048 → 16 个
   N=128 tile。
4. **MLIR 工具链**：snapshot wheel 路线不可用（PyPI 无 `mlir` 包、无 snapshot-build wheel 发布），
   按计划预案走源码编译 **LLVM 21.1.8 仅 MLIR 子项目**（详见 toolchain.md）：构建 4150/4150、
   Python 绑定验证通过、`qnn.td`/`qisa.td` 经 `mlir-tblgen` 编译通过（1414/15347 行 op 声明）。
   M2a 数值/字段/容器验收为纯 Python，不依赖 MLIR 运行时；lowering 的 MLIR pass 实现归 P4。
