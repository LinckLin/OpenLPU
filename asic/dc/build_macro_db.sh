#!/usr/bin/env bash
# build_macro_db.sh — SMIC28 SRAM macro liberty -> Synopsys .db (DC).
#
# Reads the compiled SMIC28 macro NLDM liberty from asic/sram_macros/ and writes
# DC .db files into asic/dc/db/, renamed to the sky130 corner label so the
# multi-corner DC flow (dc_top.tcl) can reference them side by side with the
# sky130 logic .db.
#
# Corner mapping (cross-technology reachability probe, NOT a tapeout corner):
#   SMIC tt_ctypical_0p90v_0p90v_25c  -> sky130 tt_025C_1v80
#   SMIC ssg_cworstt_0p81v_0p81v_125c -> sky130 ss_100C_1v60
#
# Env:  LC  lc_shell binary (default: Synopsys compat path)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

LC="${LC:-/home/public/app/synopsys/compat/bin/lc_shell}"
mkdir -p asic/dc/db

declare -A SMIC_CORNER=(
  [tt_025C_1v80]=tt_ctypical_0p90v_0p90v_25c
  [ss_100C_1v60]=ssg_cworstt_0p81v_0p81v_125c
)

for macro in kh4096x64 kn128x16 ang4096x64; do
  for sky in tt_025C_1v80 ss_100C_1v60; do
    smic="${SMIC_CORNER[$sky]}"
    lib="asic/sram_macros/${macro}/${macro}_${smic}.lib"
    db="asic/dc/db/${macro}_${sky}.db"
    if [ ! -f "$lib" ]; then
      echo "WARN: missing $lib (skipping $macro $sky)" >&2
      continue
    fi
    MACRO_LIB="$lib" MACRO_DB="$db" \
      "$LC" -batch -x "source asic/dc/lc_build_macro_db.tcl"
    echo "built $db"
  done
done
