#!/usr/bin/env bash
# Verify the TT-mapped 16x16 dual-MAC PE tile with the official cell model.
set -euo pipefail
cd "$(dirname "$0")"
IVERILOG="${IVERILOG_BIN:-/usr/bin/iverilog}"
VVP="${VVP_BIN:-/usr/bin/vvp}"
STDCELL_MODEL="${SMIC28_STDCELL_VERILOG:-/home/public/PDK/SMIC28/STDcell/SCC28NHKCP_HDC30P140_RVT_V0p2/verilog/scc28nhkcp_hdc30p140_rvt.v}"
GATE_NETLIST="dc/reports/matrix_int8_pe_tile_smic28_tt_025C_1v80_tile1c.v"

if [[ ! -f "$STDCELL_MODEL" ]]; then
  echo "missing SMIC28 standard-cell model: $STDCELL_MODEL" >&2
  exit 2
fi
if [[ ! -f "$GATE_NETLIST" ]]; then
  echo "missing tile gate netlist: $GATE_NETLIST (run TT tile synthesis first)" >&2
  exit 2
fi

mkdir -p obj_dir_matrix_pe_tile_gate
"$IVERILOG" -g2012 -Dfunctional -s matrix_int8_pe_tile_gate_tb \
  -o obj_dir_matrix_pe_tile_gate/matrix_int8_pe_tile_gate_check.vvp \
  "$GATE_NETLIST" "$STDCELL_MODEL" matrix_int8_pe_tile_gate_tb.sv
"$VVP" obj_dir_matrix_pe_tile_gate/matrix_int8_pe_tile_gate_check.vvp
