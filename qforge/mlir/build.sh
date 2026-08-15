#!/usr/bin/env bash
# Build the qnn/qisa dialect + qnn-to-qisa pass as QnnPlugin.so (loadable by
# mlir-opt). Reuses the MLIR 21.1.8 build tree (built in P2.1).
set -euo pipefail

MLIR_DIR="${MLIR_DIR:-$HOME/.cache/llvm-mlir/build/lib/cmake/mlir}"
LLVM_DIR="${LLVM_DIR:-$HOME/.cache/llvm-mlir/build/lib/cmake/llvm}"
BUILD_DIR="${BUILD_DIR:-/tmp/qnn-build}"

cmake -G Ninja -S "$(dirname "$0")" -B "$BUILD_DIR" \
  -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C "$BUILD_DIR" QnnPlugin
echo "built: $BUILD_DIR/lib/QnnPlugin.so"
