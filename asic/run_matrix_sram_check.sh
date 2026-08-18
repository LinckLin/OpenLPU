#!/usr/bin/env bash
# Directed numeric/address verification for the SMIC28 macro-backed matrix core.
set -euo pipefail
cd "$(dirname "$0")"
VERILATOR=${VERILATOR_BIN:-/usr/bin/verilator}
BUILD=obj_dir_mxsram
rm -rf "$BUILD"
"$VERILATOR" --cc --exe --build -Wno-fatal -Wno-WIDTH \
  -I../rtl -I. \
  --top-module matrix_engine \
  -o matrix_sram_check \
  matrix_engine_sram.sv matrix_sram_check_main.cpp \
  --Mdir "$BUILD"
"$BUILD"/matrix_sram_check
