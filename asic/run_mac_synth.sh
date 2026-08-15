#!/usr/bin/env bash
# run_mac_synth.sh — QCore P10 MAC synthesis (Yosys + sky130).
# Usage: ./asic/run_mac_synth.sh <corner>   (corner e.g. tt_025C_1v80)
#
# Both mac_bf16 and mac_int8 are synthesized in one pass (hierarchy -check, no
# -top) — this is the reproducible configuration behind the §2 MAC numbers.
set -euo pipefail
CORNER="${1:-tt_025C_1v80}"
LIB="${LIB:-$HOME/.eda/liberty/sky130_fd_sc_hd__${CORNER}.lib}"
YOSYS="${YOSYS:-$HOME/.eda/yosys-src/yosys}"
cd "$(dirname "$0")/.."
python3 asic/preprocess.py
mkdir -p asic/netlist
"$YOSYS" -p "read_liberty -lib $LIB; \
  read_verilog -sv asic/synth_datapath.sv asic/synth_mac.sv; \
  hierarchy -check; proc; memory; opt; techmap; opt; \
  dfflibmap -liberty $LIB; abc -liberty $LIB; opt_clean; \
  stat -liberty $LIB; stat -width; \
  write_verilog -noattr -noexpr asic/netlist/synth_mac.v"
python3 asic/fix_netlist.py asic/netlist/synth_mac.v
