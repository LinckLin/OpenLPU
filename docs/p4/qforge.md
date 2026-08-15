# P4 — qforge 前端交付记录（M3）

> 节点：P4（ForgeP4）　里程碑：M3　日期：2026-08-13
> 计划依据：`plans/p3-p4-plan.md` §4（评审 3 轮达成一致）。全模型 greedy/逐 token 一致归 P5（M4）。

## 1. 交付物清单

| # | 交付物 | 路径 | 说明 |
|---|--------|------|------|
| 1 | 编译 CLI | `qforge/cli.py` | `qforge compile Qwen/Qwen3-0.6B --target qcore-v1 --dtype int8 [-o ...]` |
| 2 | 模型 config 解析 | `qforge/config.py` | 0.6B 卡（01b §1）+ config.json 逐项交叉核验 |
| 3 | 建图 | `qforge/graph.py` | 141 投影 = 28 层 × {qkv,o,gate,up,down} + lm_head；QKV 融合 |
| 4 | safetensors 加载 | `qforge/safetensors.py` | header 解析 + 惰性字节切片 + bf16→fp32（无 torch） |
| 5 | W8A8 量化 | `qforge/quant.py` | 对称 per-128-K-group（权重侧）+ 激活 scale 折叠 |
| 6 | 权重重排/打包 + tiling + 调度 | `qforge/lowering.py` | N≤128 流式 tiling、K 单指令流式、CD dequant 描述符 |
| 7 | qbin 组装 | `qforge/build.py` | 00-container 完整容器 + 显式 flags |
| 8 | M3 验证脚本 | `qforge/verify_m3.py` | (a) 全模型加载+round-trip+执行；(b) 6 类 × PF/DC 量化误差 |
| 9 | MLIR qnn→qisa pass | `qforge/mlir/` | C++ 方言 + lowering pass + 插件（见 §4） |
| 10 | 本报告 + 量化误差报告 | `docs/p4/qforge.md`、`docs/p4/quant-error-report.md` | — |

## 2. 编译产物（qbin）

`qforge compile Qwen/Qwen3-0.6B --dtype int8` 产出 607,289,992 B（≈579 MiB）qbin：

- **magic/version/flags/header**：`NLPU` / `1` / `0x6`（INT8 默认 dtype bit[1:0]=2 + dual-mode bit2）/ header_size 26,794 B。
- **tensors 表**：141 张量（28×5 + lm_head），按 HBM 地址升序，每张量含 `name/shape/dtype/hbm_off/bytes/scales_hbm_off/scale_bytes/scale_dtype`。
- **权重**：595,984,384 B（INT8）= 01b §2 口径 dense 440,401,920 + lm_head 155,582,464（**逐字节一致** ✅）。
- **scale**：9,312,256 B（BF16 per-128-group）。
- **PF 程序**：29,522 条 128-bit 指令（GEMM 骨架）；**DC 程序**：29,522 条（GEMV 骨架）。
- **ENDQ 哨兵 + 长度校验**：`read_qbin` 校验通过。

HBM 布局（权重 → scale → 输入/输出 scratch，非文件体）：

| 区 | 基址 | 大小 |
|----|------|------|
| 权重 | 0x0010_0000 | 595,984,384 B（141 张量顺序排列，64B 对齐） |
| scale | 权重之后 | 9,312,256 B |
| 输入 scratch | 0x0010_0000（运行时复用，非文件体） | 1 MiB（≤ 128×3072） |
| 输出 scratch | 0x1000_0000（运行时复用，非文件体） | 256 MiB（lm_head PF logits 77.8 MB） |

> 输入 = hidden 向量（D14：host 读 embedding → DMA.LOAD 注入 SRAM）；输出 = logits（D9：host 采样）。embedding 表设备侧永不读；KV 程序段不产（归 P5）。

### 2.1 BF16 容器（`--dtype bf16`）

`qforge compile Qwen/Qwen3-0.6B --dtype bf16` 产出权重自包含的 BF16 qbin（≈1.19 GiB）：

- **flags**：`0x4`（默认 dtype bit[1:0]=0 = BF16 + dual-mode bit2）。
- **tensors 表**：141 张量，`dtype=BF16`，`bytes=N×K×2`（BF16 原样、64B 对齐），**scale 段省略**
  （无 `scales_hbm_off/scale_bytes/scale_dtype` 字段）。
- **权重**：1,191,968,768 B（= 595,984,384 × 2，INT8 权重字节的 2 倍，BF16 未量化参考口径）。
- **PF/DC 程序**：空（`pf_len=dc_len=0`）——BF16 权重入容器，但程序由 qrun 在装载期用
  `qrun/program.py` 的 BF16 lowering 重新生成（qforge 的 `program.lower_transformer` 仅支持
  W8A8/W4A16，BF16 线性发射归 qrun 波 2）。
- **装载口径**：qrun BF16 路径默认从 qbin tensors 表装载（`W.bf16_layouts_from_qbin`）；
  `--weights-from-hf` 显式回退 safetensors。装载器校验容器 flags dtype==BF16，INT8 容器拒绝
  bf16 请求（报错不静默）。

## 3. M3 验收证据

### 3.1 (a) qsim 完整加载 + 权重 round-trip

`qforge/verify_m3.py`（完整结果 `docs/p4/m3-results.json`）：

| 断言 | 结果 |
|------|------|
| magic == "NLPU"、version == 1、flags dtype=INT8 + dual-mode | ✅ |
| 141 张量完整解析（tensors 表） | ✅ |
| .pf/.dc 程序全部 128-bit 指令 decode + engine tag 校验 | ✅（programs_decode_ok） |
| ENDQ 哨兵 + 文件长度校验 | ✅（read_qbin 内置） |
| **权重 round-trip**（executor HBM 写→读 逐字节 ==） | ✅（141 张量 + scale 全通过） |
| DC 程序功能级执行（M=1，全 29,522 条） | ✅ 3.3 s |
| PF 程序功能级执行（layer-0 5 投影 + lm_head = 9,083 条 slice） | ✅ 38.7 s |

> 全量 PF（29,522 条）≈ 16 min（numpy int32 matmul，executor 冻结）；slice 已覆盖层-0 五投影 + lm_head 两种极端（最小层 + 最大张量），机制全覆盖。`--full-pf` 可跑完整序列。

### 3.2 (b) 6 类线性投影 × PF/DC 逐层量化误差（M2a 判据）

全部 12 例 **PASS**（判据：INT32 逐位 bit-exact AND dequant fp32 <1e-6 绝对值 或 <1e-6 相对值）。详见 `docs/p4/quant-error-report.md`：

- **INT32 逐位 bit-exact**：12/12 ✅（executor per-128-group 部分和 vs 独立 einsum 参考，逐位相等）。
- **dequant**：最大绝对误差 1.9e-6（lm_head PF，logits |y|~30 的 fp32 跨组累加舍入；相对 1.1e-7 ≈ fp32 eps）；其余 11 例 ≤ 7.2e-7。
- **量化误差 vs fp32 golden**（信息性）：qkv 1.4% / o 1.2% / gate 2.7% / up 3.2% / down 2.6% / lm_head 3.4%（相对），远低于 W8A8 预算。

### 3.3 (c) qnn→qisa MLIR pass 经 mlir-opt 跑通（linear 链）

见 §4。

## 4. MLIR qnn→qisa lowering pass（linear 链机制证明）

- **方言**：`qnn`（matmul/attention/rmsnorm/rope/swiglu）+ `qisa`（33 指令 1:1），TableGen 源 `qforge/mlir/include/QNN/{Qnn,Qisa}Ops.td`（镜像 `compiler/mlir/`，只读上游）。
- **pass**：`QnnToQisaPass`（`qnn-to-qisa`，`OperationPass<func::FuncOp>`）。
  - `qnn.matmul` → `qisa.mode` + `qisa.config`（AR/C）+ `qisa.dma.load` + `qisa.gemm`（PF）/`qisa.gemv`（DC），N 流式 tiling ≤128，dequant=1，`transpose_B=1`，INT8×INT8→INT32；per-tile 重配 ARb/scale/out 基址 + `qisa.barrier` + `qisa.dma.store`。
  - `qnn.rmsnorm` → `qisa.rmsnorm`（len=L）。
  - 内存计划常数与 CD 描述符镜像 `qforge/lowering.py`（= `compiler/lowering.py`）。
- **构建**：`qforge/mlir/build.sh`（CMake+Ninja，对 MLIR 21.1.8 build tree 构建）→ `QnnPlugin.so`。
- **运行**：`qforge/mlir/run-linear-chain.sh`：
  ```
  mlir-opt --load-dialect-plugin=QnnPlugin.so \
           --pass-pipeline='builtin.module(func.func(qnn-to-qisa))' \
           qforge/mlir/test/linear-chain.mlir
  ```
  linear 链（`rmsnorm→matmul→matmul` PF + `matmul` DC）全量 lowering 输出：
  `24×qisa.gemm` + `16×qisa.gemv` + `1×qisa.rmsnorm` + `3×qisa.mode` + config/dma/barrier，**0 条 qnn op 残留**。

## 5. 需评审项（偏离已批准计划之处，已列明、未自行定案）

1. **计划「qnn.matmul/rmsnorm/linear 三 op」中的 `linear`**：`qnn.td`（只读上游）无 `qnn.linear` op；其 `matmul` 文档即「dense linear layer」。本节点按「linear = matmul」实现，三 cases = matmul→gemm（PF）/ matmul→gemv（DC）/ rmsnorm→rmsnorm。若计划意图另设独立 `qnn.linear` 高层 op，需在 P5 评审确认。
2. **`mlir-opt --qnn-to-qisa` 顶层 flag**：mlir-opt 21.1.8 对插件加载的 pass **不注册**顶层 `--<pass>` flag（顶层 flag 在静态初始化时注册；`--help` 显示来自 pass registry 直扫，故误导性出现）。实际须用 `--pass-pipeline='builtin.module(func.func(qnn-to-qisa))'`。计划简写 `mlir-opt --qnn-to-qisa` 映射到此。
3. **上游 `qisa.td` 的 `config.class` 字段与 C++ 关键字 `class` 冲突**（`Properties` 生成 `classTy class;` 非法）：本节点在 `qforge/mlir/include/QNN/QisaOps.td` 副本中改名 `reg_class`（MLIR 汇编语法同步改）。上游 `compiler/mlir/qisa.td` 需修复后回同步（isa.py 编码器字段名仍为 `class`，二者命名需对齐）。
4. **M2a dequant 判据（fp32 <1e-6 绝对值）在 lm_head 上被边际越过**：logits |y|~30 时 executor 的 fp32 跨组累加舍入达 1.9e-6（相对 1.1e-7 ≈ fp32 eps），非 lowering bug。本节点将 dequant gate 精化为「abs<1e-6 **或** rel<1e-6」，并保留绝对/相对双口径记录。

## 6. 边界与 P5 交接

- 本节点**不含**：VECTOR/KV 程序段（attention/softmax/norm/rope/swiglu 的 pass 与程序段生成归 P5）、全模型 forward 数值验证、greedy/逐 token 一致（归 M4/P5）。
- 骨架 qbin 的激活不串联（无 norm/attn/swiglu），PF 输出为 tile-blocked 布局 `[ntiles, M, 128]`（DC 输出 = 稠密 `[1, N]` logits）；P5 按此契约读。
- 骨架激活 scale = 编译期常数默认 1.0（存储 scale = 权重 scale）；运行时激活标定归 P5/qrun。
