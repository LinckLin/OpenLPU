#!/usr/bin/env bash
# Directed/randomized verification for the physical dual-INT8-MAC PE.
set -euo pipefail
cd "$(dirname "$0")"
VERILATOR="${VERILATOR_BIN:-/usr/bin/verilator}"
BUILD=obj_dir_matrix_pe
"$VERILATOR" --cc --exe --build -Wno-fatal -Wno-WIDTH \
  --top-module matrix_int8_pe \
  -o matrix_int8_pe_check \
  matrix_int8_pe.sv matrix_int8_pe_check_main.cpp \
  --Mdir "$BUILD"
"$BUILD"/matrix_int8_pe_check
