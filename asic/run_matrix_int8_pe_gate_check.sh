#!/usr/bin/env bash
# Verify the registered RTL probe and its mapped SMIC28 TT gate netlist.
set -euo pipefail
cd "$(dirname "$0")"
VERILATOR="${VERILATOR_BIN:-/usr/bin/verilator}"
IVERILOG="${IVERILOG_BIN:-/usr/bin/iverilog}"
VVP="${VVP_BIN:-/usr/bin/vvp}"
STDCELL_MODEL="${SMIC28_STDCELL_VERILOG:-/home/public/PDK/SMIC28/STDcell/SCC28NHKCP_HDC30P140_RVT_V0p2/verilog/scc28nhkcp_hdc30p140_rvt.v}"
GATE_NETLIST="dc/reports/matrix_int8_pe_smic28_tt_025C_1v80_pe1c.v"

if [[ ! -f "$STDCELL_MODEL" ]]; then
  echo "missing SMIC28 standard-cell model: $STDCELL_MODEL" >&2
  exit 2
fi
if [[ ! -f "$GATE_NETLIST" ]]; then
  echo "missing gate netlist: $GATE_NETLIST (run DC TT synthesis first)" >&2
  exit 2
fi

"$VERILATOR" --cc --exe --build -Wno-fatal -Wno-WIDTH -Wno-CASEX \
  -Wno-TIMESCALEMOD -Wno-PINMISSING \
  -I. --top-module matrix_int8_pe_probe \
  -o matrix_int8_pe_probe_rtl_check \
  matrix_int8_pe_probe.sv matrix_int8_pe_probe_check_main.cpp \
  --Mdir obj_dir_matrix_pe_probe_rtl
obj_dir_matrix_pe_probe_rtl/matrix_int8_pe_probe_rtl_check

mkdir -p obj_dir_matrix_pe_probe_gate
"$IVERILOG" -g2012 -Dfunctional -s matrix_int8_pe_gate_tb \
  -o obj_dir_matrix_pe_probe_gate/matrix_int8_pe_gate_check.vvp \
  "$GATE_NETLIST" "$STDCELL_MODEL" matrix_int8_pe_gate_tb.sv
"$VVP" obj_dir_matrix_pe_probe_gate/matrix_int8_pe_gate_check.vvp
