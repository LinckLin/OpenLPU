#!/usr/bin/env bash
# Verify the registered-boundary 16x16 dual-MAC PE tile wavefront.
set -euo pipefail
cd "$(dirname "$0")"
VERILATOR="${VERILATOR_BIN:-/usr/bin/verilator}"
BUILD=obj_dir_matrix_pe_tile
"$VERILATOR" --cc --exe --build -Wno-fatal -Wno-WIDTH \
  --top-module matrix_int8_pe_tile_probe \
  -o matrix_int8_pe_tile_check \
  -I. matrix_int8_pe_tile_probe.sv matrix_int8_pe_tile_check_main.cpp \
  --Mdir "$BUILD"
"$BUILD"/matrix_int8_pe_tile_check
