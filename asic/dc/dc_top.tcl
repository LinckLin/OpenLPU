# ============================================================================
# dc_top.tcl — DC synthesis of the full design synth_top (P10 §10, step 5).
#
# synth_top = command_processor @ MAX_VEC=128 + 8 MiB scratchpad SRAM black box.
# The SRAM is `sram_macro` (empty module -> black box) protected with
# set_dont_touch; its area/power stay the separate OpenRAM/density estimate.
#
# The engine RTL is the DC-desugared tree asic/dc/gen_full/ (see hoist_dc.py:
# loop-break `k=-1`->`break`, variable-bound loop -> fixed bound + guard, and
# module-level declaration hoist).  synth_top.sv / sram_macro.sv are copied
# verbatim (DC-clean).
#
# NOTE: the engines are the *co-sim functional model* (runtime-bound loops,
# inferred RAMs), NOT the physical 128-lane datapath — see asic-report.md §7.
# This run records whether DC can elaborate/link/compile it and at what cost;
# any frontend incompatibility or resource blow-up is reported honestly.
#

# Env:  DC_CORNER  tt_025C_1v80 | ss_100C_1v60
#       DC_TOP_COMPILE  0 = elaborate+link+report only (fast frontend check)
#                       1 = full compile_ultra (bounded externally by timeout)
# ============================================================================
set corner [getenv DC_CORNER]
set compile 1
if {[info exists env(DC_TOP_COMPILE)] && $env(DC_TOP_COMPILE) == "0"} { set compile 0 }

set dc_dir asic/dc
set dw_dir /home/public/app/synopsys/syn/O-2018.06-SP1/libraries/syn
set search_path [concat $search_path \
    [list . asic/dc/gen_full asic $dc_dir $dc_dir/db $dc_dir/reports $dw_dir]]
set target_library  [list $dc_dir/db/sky130_fd_sc_hd__${corner}.db]
set synthetic_library [list dw_foundation.sldb]
set link_library [concat {*} $target_library $synthetic_library]

puts "INFO: corner=$corner compile=$compile"
puts "INFO: link_library=$link_library"

# --- read + elaborate --------------------------------------------------------
# synth_top.sv `include`s qcore_pkg.sv + command_processor.sv (which transitively
# includes softfloat/matrix_engine/vector_engine/rope_lut/dma_engine/kv_addrgen);
analyze -format sverilog asic/dc/gen_full/synth_top.sv
analyze -format sverilog asic/dc/gen_full/sram_macro.sv
elaborate synth_top
link

# --- SRAM macro: black box + dont_touch (plan step 5) ------------------------
set_dont_touch [get_cells u_sram]
set_dont_touch [get_references sram_macro]

# --- constraints: 1 ns probe clock (sta.tcl caliber) ------------------------
create_clock -name clk -period 1.0 [get_ports clk]
set_input_delay  -clock clk 0.0 [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk 0.0 [all_outputs]

# --- synthesize (optional; bounded by external timeout) ----------------------
if {$compile == 1} {
  compile_ultra
  write -format verilog -hierarchy -output $dc_dir/reports/synth_top_${corner}.v
}

# --- reports -----------------------------------------------------------------
redirect -tee $dc_dir/reports/synth_top_${corner}.rpt {
  puts "==== DC corner=$corner design=synth_top compile=$compile ===="
  report_timing -path_type full -delay_type max -max_paths 10 -sort_by slack -nosplit
  report_area -hierarchy
  report_power
  report_reference
}
puts "INFO: wrote $dc_dir/reports/synth_top_${corner}.rpt"
exit
