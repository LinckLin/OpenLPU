#!/usr/bin/env bash
# run_sram_check.sh — QCore P10b SRAM parametrization unit test (Verilator).
# Validates rtl/sram.sv sram_top for default (512 KiB/bank) + shrink
# (256 KiB/bank, 128 KiB/bank) instances.  See asic/sram_check.sv.
# Uses the project-locked Verilator 4.038 (system), not the pip verilator.
set -euo pipefail
cd "$(dirname "$0")"
VERILATOR=${VERILATOR_BIN:-/usr/bin/verilator}
BUILD=obj_dir_sram
rm -rf "$BUILD"
"$VERILATOR" --cc --exe --build \
  -I../rtl -I. \
  --top-module sram_check \
  -o sram_check \
  sram_check.sv sram_check_main.cpp \
  --Mdir "$BUILD"
"$BUILD"/sram_check
