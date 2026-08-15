#!/usr/bin/env bash
# run_fp_equiv.sh — QCore P10b pipelined-datapath vs softfloat equivalence
# spot check (Verilator).  See asic/fp_equiv.sv.
set -euo pipefail
cd "$(dirname "$0")"
VERILATOR=${VERILATOR_BIN:-/usr/bin/verilator}
BUILD=obj_dir_fpeq
"$VERILATOR" --cc --exe --build -Wno-fatal -Wno-WIDTH \
  -I../rtl -I. \
  --top-module fp_equiv \
  -o fp_equiv \
  fp_equiv.sv fp_equiv_main.cpp \
  --Mdir "$BUILD"
"$BUILD"/fp_equiv
