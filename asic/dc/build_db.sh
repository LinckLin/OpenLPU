#!/usr/bin/env bash
# build_db.sh — sky130 liberty -> Synopsys .db (Library Compiler O-2018.06-SP1).
#
# Per corner: clean_lib.py (semantic-neutral bulk-well LBDB-27 fix, see its
# docstring) -> lc_shell read_lib/write_lib.  Outputs asic/dc/db/*.{lib,db}
# (gitignored; regenerable).  OpenSTA keeps reading the *original* raw liberty.
#
# Env:  LC        lc_shell binary (default: Synopsys compat path)
#       LIB_DIR   raw sky130 liberty dir (default ~/.eda/liberty)
#       CORNERS   space-separated corner list (default tt_025C_1v80 ss_100C_1v60)
#
# Raw liberty source / acquisition: docs/reproduction.md §6.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

LC="${LC:-/home/public/app/synopsys/compat/bin/lc_shell}"
LIB_DIR="${LIB_DIR:-$HOME/.eda/liberty}"
CORNERS="${CORNERS:-tt_025C_1v80 ss_100C_1v60}"

mkdir -p asic/dc/db
for corner in $CORNERS; do
  libname="sky130_fd_sc_hd__${corner}"
  python3 asic/dc/clean_lib.py \
    "$LIB_DIR/${libname}.lib" "asic/dc/db/${libname}.lib"
  LIB_CORNER="$corner" "$LC" -no_gui -f asic/dc/lc_build_db.tcl
done
echo "built .db for: $CORNERS"
