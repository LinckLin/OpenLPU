# ============================================================================
# lc_build_db.tcl — Synopsys Library Compiler: cleaned sky130 liberty -> .db
# (P10 §10.2 DC flow).
#
# Prereq (run first, per corner):
#   python3 asic/dc/clean_lib.py "$LIB_DIR/sky130_fd_sc_hd__${corner}.lib" \
#                                asic/dc/db/sky130_fd_sc_hd__${corner}.lib
#   (clean_lib.py fixes the two bulk-well LBDB-27 constructs; see its docstring)
#
# Env inputs:  LIB_CORNER   tt_025C_1v80 | ss_100C_1v60 | ...
# Reads  asic/dc/db/sky130_fd_sc_hd__${corner}.lib   (cleaned liberty)
# Writes asic/dc/db/sky130_fd_sc_hd__${corner}.db    (binary .db)
#
# Run via:  LIB_CORNER=tt_025C_1v80 lc_shell -no_gui -f asic/dc/lc_build_db.tcl
#           (or the wrapper asic/dc/build_db.sh, which runs clean_lib.py first)
# ============================================================================
set corner [getenv LIB_CORNER]
set libname "sky130_fd_sc_hd__${corner}"

puts "INFO: read_lib asic/dc/db/${libname}.lib"
read_lib "asic/dc/db/${libname}.lib"

puts "INFO: write_lib ${libname} -> asic/dc/db/${libname}.db"
write_lib "$libname" -f db -o "asic/dc/db/${libname}.db"
exit
