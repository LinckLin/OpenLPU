#!/usr/bin/env bash
# run_dc.sh — QCore DC synthesis (Design Compiler O-2018.06-SP1).
# Usage: ./asic/dc/run_dc.sh <corner> <design>
#   corner : tt_025C_1v80 | ss_100C_1v60
#   design : synth_datapath | mac_bf16 | synth_top
#   DC_TECH env: sky130 (default) | smic28 — logic std cell library (synth_top only)
set -euo pipefail
CORNER="${1:-tt_025C_1v80}"
DESIGN="${2:-synth_datapath}"

DC=/home/public/app/synopsys/compat/bin/dc_shell
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."   # repo root

export DC_CORNER="$CORNER"
export DC_DESIGN="$DESIGN"

case "$DESIGN" in
  synth_datapath|mac_bf16)
    TCL=asic/dc/dc_flow.tcl
    ;;
  synth_top)
    TCL=asic/dc/dc_top.tcl
    ;;
  *)
    echo "unknown design '$DESIGN'" >&2; exit 2;;
esac

mkdir -p asic/dc/reports
python3 asic/dc/desugar_dc.py
if [ "$DESIGN" = "synth_top" ]; then
  python3 asic/dc/hoist_dc.py
fi
"$DC" -no_gui -f "$TCL"
