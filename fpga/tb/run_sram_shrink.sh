#!/usr/bin/env bash
# run_sram_shrink.sh — ddr_if SRAM_BYTES shrink-instance smoke (Verilator).
# Instantiates ddr_if at 256 KiB/bank (4 MiB) and 128 KiB/bank (2 MiB) and runs
# the self-checking read/write round-trip + DDR-unaffected + host-port sequence.
# See fpga/tb/sram_shrink_tb.sv.  Uses the project-locked Verilator 4.038.
set -euo pipefail
cd "$(dirname "$0")"
VERILATOR=${VERILATOR_BIN:-/usr/bin/verilator}
BUILD=obj_sram_shrink
rm -rf "$BUILD"
"$VERILATOR" --cc --exe --build \
  -Wno-fatal -Wno-WIDTH \
  -I.. \
  --top-module sram_shrink_tb \
  -o sram_shrink \
  sram_shrink_tb.sv sim_sram_shrink.cpp \
  --Mdir "$BUILD"
"$BUILD"/sram_shrink
