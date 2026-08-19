#!/usr/bin/env bash
# Verify array-grid control and arithmetic on a representative 2x2 tile grid.
set -euo pipefail
cd "$(dirname "$0")"

BUILD=obj_dir_matrix_pe_array
verilator --cc --exe --build \
  --top-module matrix_int8_pe_array \
  -GTILE_ROWS=2 -GTILE_COLS=2 -GPE_ROWS=32 -GPE_COLS=32 \
  -GROW_ADDR_W=5 \
  -Wno-fatal -Wno-WIDTH -Wno-UNUSED \
  -I. matrix_int8_pe_array.sv matrix_int8_pe_array_check_main.cpp \
  --Mdir "$BUILD" -o matrix_int8_pe_array_check
"$BUILD"/matrix_int8_pe_array_check
