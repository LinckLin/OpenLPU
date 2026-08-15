# P2.1 工具链选择记录

## 结论（版本写死）

| 项 | 选择 | 版本 | 理由 |
|----|------|------|------|
| MLIR 前端 | LLVM/MLIR 源码编译（仅 MLIR 子项目） | **llvmorg-21.1.8** | 官方 snapshot wheel 不可用（见下），走计划 §4 路线② |
| 构建系统 | CMake + Ninja | cmake 3.22 / ninja 1.11 | 系统自带 |
| MLIR Python 绑定 | nanobind（+ pybind11） | nanobind 2.14 / pybind11 3.1 | LLVM 21+ 的 MLIR 绑定改用 nanobind |
| 数值执行器 | Python + numpy + ml_dtypes | numpy 1.26.4 / ml_dtypes 0.5.4 | BF16 原生处理（Golden 用 numpy≥2.1 原生 bf16，两者 bf16 dtype 互通） |

## 双路线实测（2026-08-13）

计划 §4 P2.1 要求：**① 先试 LLVM 官方 snapshot wheel（GitHub llvm-project releases 的 snapshot-build
含 mlir_core 绑定），拿到后第一步验证 Python 绑定存在；② 缺失则源码编译**。

1. `pip index versions mlir` → **No matching distribution**（PyPI 无官方 `mlir` 包）。
2. `llvm-project` GitHub releases 逐条核查：无 `snapshot-build` 发布；最新 stable 发布
   （llvmorg-22.1.8 / 21.1.8 / 20.1.8）的资产为完整 LLVM 二进制 tarball（1.9 GB，不含
   Python 绑定 wheel）。
3. **判定：路线①不可用 → 按计划 §6 风险预案走路线②**，仅编译 MLIR 子项目（非全 LLVM）。

## 源码编译

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
  -Dpybind11_DIR=<.../pybind11/share/cmake/pybind11> \
  -Dnanobind_DIR=<.../nanobind/cmake>
ninja -C build -j128 MLIRPythonModules mlir-tblgen mlir-opt
```

> 排障记录：初版 configure 报缺 pybind11 → `pip install pybind11`；再报缺 **nanobind**
> （LLVM 21 的 MLIR Python 绑定改用 nanobind）→ `pip install nanobind` 后 configure 通过。
> 本机为 256 核（计划文本写 64 核有误），故用 `-j128`（计划 `-j64` 为保守值，非硬约束）。

路径（不在交付目录内，属环境安装）：
- 源码 `~/.cache/llvm-mlir/llvm-project-21.1.8.src`
- 构建 `~/.cache/llvm-mlir/build`
- 安装 venv `~/.local/llvm-mlir-venv`

## 验证结果（构建完成，2026-08-13）

- `ninja MLIRPythonModules mlir-tblgen mlir-opt` → **4150/4150 目标完成**；`mlir-opt` / `mlir-tblgen`
  及全部 `_mlir_libs/*.so` 绑定模块产出。
- Python 绑定验证（计划要求的第一步）：
  `PYTHONPATH=build/tools/mlir/python_packages/mlir_core python3 -c "import mlir.ir"` → **OK**；
  `from mlir.dialects import func, arith, linalg, scf` → **OK**（Context 创建成功）。
- 方言 TableGen 编译验证：`mlir-tblgen -I <mlir/include> -I <llvm/include>
  compiler/mlir/qnn.td -gen-op-decls` → **1414 行生成**；`compiler/mlir/qisa.td` → **15347 行生成**。

> 注：本机缺 `python3-venv`（ensurepip 不可用），未走 `python3 -m venv` 装 wheel；改用 build-tree
> `PYTHONPATH` 直接使用绑定（等效，已验证）。独立 venv 方案：`apt install python3.10-venv` 后
> `pip install build/tools/mlir/python_packages/mlir_core/dist/*.whl`。
