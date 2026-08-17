# ============================================================================
# lc_build_macro_db.tcl — SMIC28 SRAM macro liberty -> Synopsys .db.
#
# Reads a compiled SMIC28 macro NLDM liberty (ARM memory compiler output) and
# writes a Library Compiler binary .db for DC synthesis.  The SMIC liberty is
# already DC/LC-clean (no bulk-well LBDB-27 constructs like sky130), so no
# cleaning pass is needed.
#
# Env inputs:  MACRO_LIB    absolute/relative path to the macro .lib
#              MACRO_DB     output .db path (relative to repo root)
#
# Run via build_macro_db.sh (maps SMIC corners -> sky130 corner labels).
# ============================================================================
set libfile [getenv MACRO_LIB]
set dbfile  [getenv MACRO_DB]

puts "INFO: read_lib $libfile"
read_lib "$libfile"

set libname [lindex [split [file tail $libfile] "."] 0]
puts "INFO: write_lib $libname -> $dbfile"
write_lib "$libname" -f db -o "$dbfile"
exit
