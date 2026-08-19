#!/usr/bin/env bash
# Exercise the production 8x8 tile grid with 256 back-to-back PF waves.
set -euo pipefail
cd "$(dirname "$0")"

BUILD=obj_dir_matrix_pe_array_full
JOBS="${VERILATOR_JOBS:-32}"
verilator --cc --exe \
  --top-module matrix_int8_pe_array_probe \
  -Wno-fatal -Wno-WIDTH -Wno-WIDTHCONCAT -Wno-UNUSED \
  -I. matrix_int8_pe_array_probe.sv \
  matrix_int8_pe_array_full_check_main.cpp \
  --Mdir "$BUILD" -o matrix_int8_pe_array_full_check
make -C "$BUILD" -f Vmatrix_int8_pe_array_probe.mk -j"$JOBS"
"$BUILD"/matrix_int8_pe_array_full_check
