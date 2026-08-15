# compiler — QCore 编译器骨架（P2 交付，M2a）

目录结构：

```
compiler/
  README.md        本文件（工具链版本 + 用法）
  isa/             Q-ISA 编码器（128-bit assembler/disassembler + qbin 容器）
    isa.py         33 条指令字段布局、encode/decode、asm 文本 assembler/disassembler
    qbin.py        .qbin 容器 writer/reader（00-container §2）
  mlir/            MLIR 方言定义（TableGen）
    qnn.td         qnn 层方言：matmul / attention / rmsnorm / rope / swiglu
    qisa.td        qisa 方言：与 02-isa 33 条指令一一对应（SYS 5/DMA 3/MATRIX 3/VECTOR 18/KV 4）
  lowering.py      lowering：qnn.matmul(linear) → qisa(GEMM/GEMV) → Q-ISA asm → qbin
```

配套目录：`qsim/`（qsim 功能核心执行器 + 测试）、`docs/p2/`（工具链记录/方言/编码器用法/M2a 报告）。

## 工具链（P2.1，版本写死）

| 项 | 选择 | 版本 | 说明 |
|----|------|------|------|
| MLIR 方言前端 | LLVM/MLIR | **llvmorg-21.1.8**（写死） | 源码编译，仅 MLIR 子项目 + Python 绑定 |
| 构建 | CMake + Ninja | cmake 3.22 / ninja 1.11 | 本机 256 核，`-j128` |
| Python 绑定 | MLIR Python bindings | nanobind 2.14 + pybind11 3.1 | `MLIR_ENABLE_BINDINGS_PYTHON=ON` |
| 数值执行器 | Python + numpy + ml_dtypes | numpy 1.26 / ml_dtypes 0.5.4 | qsim 功能核心 |

### P2.1 路线裁决

计划 §4 双路线：**① 优先 LLVM 官方 snapshot wheel → 验证 Python 绑定 → ② 缺失则源码编译**。

实测（2026-08-13）：
- PyPI 无 `mlir` 包（`pip index versions mlir` → No matching distribution）。
- `llvm-project` GitHub releases 无 `snapshot-build` 含 `mlir_core` wheel 的发布；最新 stable
  release 资产为完整 LLVM 二进制 tarball（无 Python 绑定）。
- **判定：snapshot wheel 路线不可用 → 走路线 ② 源码编译**（计划 §6 风险表已预案）。

源码编译配置（仅 MLIR，非全 LLVM）：

```bash
cmake -G Ninja -S llvm-project-21.1.8.src/llvm -B build \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -Dpybind11_DIR=<pybind11 cmake dir> -Dnanobind_DIR=<nanobind cmake dir>
```

> LLVM 21+ 的 MLIR Python 绑定改用 **nanobind**（不再只有 pybind11），需 `pip install nanobind`。
> 构建目标：`MLIRPythonModules mlir-tblgen mlir-opt`。构建/安装路径：
> 源码 `~/.cache/llvm-mlir/llvm-project-21.1.8.src`，构建 `~/.cache/llvm-mlir/build`，
> venv `~/.local/llvm-mlir-venv`。验证：`venv/bin/python -c "import mlir.ir"`。

## 用法（最小 linear 层 → qbin → qsim 执行）

```python
from compiler.lowering import build_linear_qbin
from compiler.isa.qbin import read_qbin
from qsim.executor import Executor, load_qbin_into_executor

# 单 linear 层：x [M,1024] @ Wq^T [2048,1024]，PF M=128 / DC M=1
qb = build_linear_qbin("linear.qbin", "Qwen3-0.6B", cfg, "BF16",
                       (2048, 1024), m_pf=128, m_dc=1,
                       weight_bytes=wq_bf16.tobytes())
exe = Executor(); load_qbin_into_executor(exe, qb)
exe.write_bytes("hbm", INPUT_HBM, x_bf16.tobytes())
exe.run(qb.pf_program)            # 或 qb.dc_program
out = exe.read_bytes("hbm", OUTPUT_HBM, M*N*4)
```

验收入口：
- 数值：`python3 qsim/test_m2a.py`（{PF,DC}×{BF16,INT8} 四例）。
- 字段级断言：`python3 qsim/test_isa_fields.py`（对 02-isa 的编码/语义断言）。
