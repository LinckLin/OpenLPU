#!/usr/bin/env bash
# setup_smic28.sh — SMIC28 (28HKCP, 0.9 V) DC flow setup.
#
# Reproduces the SMIC28 library inputs locally (commercial PDK outputs are NOT
# committed to the open repo — RTL + flow scripts only; see README + D18):
#
#   1. Macro .lib/.v/.lef (and physical views) — regenerated from the ARM
#      memory compiler package (SRAM_Ccompiler_ARM20240823) into a stable dir
#      SMIC28_MACRO_DIR (default /home/public/PDK/SMIC28/macros_out), following
#      the per-macro GEN.md commands.  Skipped when a macro already exists.
#   2. Std cell .db — pre-extracted in the PDK tree (SMIC28_STD_DIR); staged
#      into asic/dc/db/ with the corner mapping below (tt/ssg <-> macro tt/ssg).
#   3. Macro .db — built into asic/dc/db/ via asic/dc/build_macro_db.sh.
#
# Corner mapping (flow label -> std cell basic liberty -> macro liberty):
#   tt_025C_1v80 -> tt_v0p9_25c           -> tt_ctypical_0p90v_0p90v_25c
#   ss_100C_1v60 -> ssg_v0p81_125c        -> ssg_cworstt_0p81v_0p81v_125c
#
# Env:  SMIC28_MACRO_DIR  macro generation dir (default below)
#       SMIC28_STD_DIR    std cell library tree root (default below)
#       LC                lc_shell binary (default: Synopsys compat path)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SMIC28_MACRO_DIR="${SMIC28_MACRO_DIR:-/home/public/PDK/SMIC28/macros_out}"
SMIC28_STD_DIR="${SMIC28_STD_DIR:-/home/public/PDK/SMIC28/STDcell/SCC28NHKCP_HDC30P140_RVT_V0p2}"
LC="${LC:-/home/public/app/synopsys/compat/bin/lc_shell}"
PKG_ROOT="${SMIC28_PKG_ROOT:-/home/public/PDK/SMIC28/SRAM_Ccompiler_ARM20240823}"
SKILL_DIR="${SMIC28_SKILL_DIR:-/home/lzl/.agents/skills/smic28-sram-compiler}"

CORNERS="tt_ctypical_0p90v_0p90v_25c,ssg_cworstt_0p81v_0p81v_125c"

# macro -> compiler kind (sram | rfsp)
declare -A KIND=([kh4096x64]=sram [ang4096x64]=sram [kn128x16]=rfsp)
# macro -> "-words .. -bits .. -mux .. -mvt .."
declare -A CFG=(
  [kh4096x64]="-words 4096 -bits 64 -mux 8 -mvt BASE"
  [ang4096x64]="-words 4096 -bits 64 -mux 8 -mvt BASE"
  [kn128x16]="-words 128 -bits 16 -mux 2 -mvt BASE"
)
COMMON="-write_mask off -pipeline off -bmux off -redundancy off -ser none"

echo "INFO: SMIC28_MACRO_DIR=$SMIC28_MACRO_DIR"
echo "INFO: SMIC28_STD_DIR=$SMIC28_STD_DIR"
mkdir -p "$SMIC28_MACRO_DIR"

# --- 1. macro generation ----------------------------------------------------
regenerate() {
  local macro="$1"
  local shim_root shim_dir entry
  shim_root="$(mktemp -d /tmp/smic28-compiler.XXXXXX)"
  shim_dir="$shim_root/${KIND[$macro]}"
  entry="$("$SKILL_DIR/scripts/prepare_compiler.sh" "${KIND[$macro]}" "$shim_dir")"
  local cfg="-instname $macro ${CFG[$macro]} $COMMON"
  local out="$SMIC28_MACRO_DIR/$macro"
  mkdir -p "$out"
  cd "$out"
  zsh "$entry" verilog $cfg
  zsh "$entry" liberty $cfg -libertyviewstyle nldm -libname "$macro" -corners "$CORNERS"
  for view in lef-fp gds2 lvs; do
    zsh "$entry" "$view" $cfg -keeplogs
  done
  cd "$REPO_ROOT"
  rm -rf "$shim_root"
  echo "INFO: regenerated $macro -> $out"
}

for macro in kh4096x64 ang4096x64 kn128x16; do
  out="$SMIC28_MACRO_DIR/$macro"
  if [[ -f "$out/$macro.v" && -f "$out/$macro.lef" && \
        -f "$out/${macro}_tt_ctypical_0p90v_0p90v_25c.lib" && \
        -f "$out/${macro}_ssg_cworstt_0p81v_0p81v_125c.lib" ]]; then
    echo "INFO: $macro already present, skip regeneration"
  else
    regenerate "$macro"
  fi
done

# --- 2. std cell .db staging ------------------------------------------------
mkdir -p asic/dc/db
for corner in tt_v0p9_25c ssg_v0p81_125c; do
  src="$SMIC28_STD_DIR/liberty/0.9v/scc28nhkcp_hdc30p140_rvt_${corner}_basic.db"
  dst="asic/dc/db/scc28nhkcp_hdc30p140_rvt_${corner}_basic.db"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: missing std cell .db: $src" >&2
    exit 1
  fi
  cp -p "$src" "$dst"
  echo "INFO: staged std cell .db -> $dst"
done

# --- 3. macro .db -----------------------------------------------------------
SMIC28_MACRO_DIR="$SMIC28_MACRO_DIR" LC="$LC" bash asic/dc/build_macro_db.sh

echo "DONE: SMIC28 libraries ready under asic/dc/db/"
