#!/usr/bin/env bash
# build_macro_db.sh — SMIC28 SRAM macro liberty -> Synopsys .db (DC).
#
# Reads the compiled SMIC28 macro NLDM liberty from the local generation dir
# (SMIC28_MACRO_DIR, default /home/public/PDK/SMIC28/macros_out — populated by
# asic/smc28/setup_smic28.sh) and writes DC .db files into asic/dc/db/, renamed
# to the flow corner label so the multi-corner DC flow (dc_top.tcl) can reference
# them side by side with the logic std cell .db.
#
# Corner mapping (flow label -> SMIC28 macro liberty corner):
#   tt_025C_1v80 -> tt_ctypical_0p90v_0p90v_25c
#   ss_100C_1v60 -> ssg_cworstt_0p81v_0p81v_125c
#
# Env:  LC               lc_shell binary (default: Synopsys compat path)
#       SMIC28_MACRO_DIR macro generation dir (default below)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

LC="${LC:-/home/public/app/synopsys/compat/bin/lc_shell}"
SMIC28_MACRO_DIR="${SMIC28_MACRO_DIR:-/home/public/PDK/SMIC28/macros_out}"
mkdir -p asic/dc/db

declare -A SMIC_CORNER=(
  [tt_025C_1v80]=tt_ctypical_0p90v_0p90v_25c
  [ss_100C_1v60]=ssg_cworstt_0p81v_0p81v_125c
)

for macro in kh4096x64 kn128x16 ang4096x64; do
  for sky in tt_025C_1v80 ss_100C_1v60; do
    smic="${SMIC_CORNER[$sky]}"
    lib="$SMIC28_MACRO_DIR/${macro}/${macro}_${smic}.lib"
    db="asic/dc/db/${macro}_${sky}.db"
    if [ ! -f "$lib" ]; then
      echo "WARN: missing $lib (skipping $macro $sky; run asic/smc28/setup_smic28.sh)" >&2
      continue
    fi
    MACRO_LIB="$lib" MACRO_DB="$db" \
      "$LC" -batch -x "source asic/dc/lc_build_macro_db.tcl"
    echo "built $db"
  done
done
