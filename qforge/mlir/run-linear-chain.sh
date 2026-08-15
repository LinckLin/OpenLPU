#!/usr/bin/env bash
# Run the qnn->qisa lowering pass on the linear-chain test via mlir-opt.
# Note: mlir-opt 21.1.8 registers plugin-loaded passes only inside an explicit
# --pass-pipeline (the top-level --qnn-to-qisa flag is registered at static
# init for statically-linked passes only). The pass argument name is
# "qnn-to-qisa"; it is anchored under func.func (the pass is a FuncOp pass).
set -euo pipefail

MLIR_OPT="${MLIR_OPT:-$HOME/.cache/llvm-mlir/build/bin/mlir-opt}"
PLUGIN="${PLUGIN:-/tmp/qnn-build/lib/QnnPlugin.so}"
INPUT="${1:-$(dirname "$0")/test/linear-chain.mlir}"

"$MLIR_OPT" \
  --load-dialect-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(func.func(qnn-to-qisa))' \
  "$INPUT"
