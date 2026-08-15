# ============================================================================
# dc_flow.tcl — QCore DC synthesis flow for the representative datapath / MAC
# (P10 §10).  Elaborate -> compile_ultra -> report_timing/area/power.
#
# Mirrors the OpenSTA baseline caliber exactly:
#   * 1 ns probe clock on `clk`, 0 ns input/output delay (sta.tcl),
#   * Fmax = 1 / critical-path data-arrival time,
#   * report_power is used for *leakage* + area only; dynamic power stays the
#     §5 activity-factor estimate (VCD back-annotation is a later step).
#
# Reads the DC-local desugared sources under asic/dc/gen/ (see desugar_dc.py).
#
# Env inputs:  DC_CORNER  tt_025C_1v80 | ss_100C_1v60
#              DC_DESIGN  synth_datapath | mac_bf16
# ============================================================================
set corner [getenv DC_CORNER]
set design [getenv DC_DESIGN]

set dc_dir asic/dc
set dw_dir /home/public/app/synopsys/syn/O-2018.06-SP1/libraries/syn

set search_path [concat $search_path \
    [list . asic/dc/gen $dc_dir $dc_dir/db $dc_dir/reports $dw_dir]]
set target_library  [list $dc_dir/db/sky130_fd_sc_hd__${corner}.db]
set synthetic_library [list dw_foundation.sldb]
set link_library [concat {*} $target_library $synthetic_library]

puts "INFO: corner=$corner design=$design"
puts "INFO: target_library=$target_library"

# --- read + elaborate --------------------------------------------------------
if {$design == "synth_datapath"} {
  analyze -format sverilog asic/dc/gen/synth_datapath.sv
  elaborate synth_datapath
} elseif {$design == "mac_bf16"} {
  analyze -format sverilog asic/dc/gen/synth_mac.sv
  elaborate mac_bf16
} else {
  puts "ERROR: unknown DC_DESIGN '$design'"
  exit 1
}
link
uniquify

# --- constraints: 1 ns probe clock (sta.tcl caliber) ------------------------
create_clock -name clk -period 1.0 [get_ports clk]
set_input_delay  -clock clk 0.0 [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk 0.0 [all_outputs]
compile_ultra
write -format verilog -hierarchy -output $dc_dir/reports/${design}_${corner}.v


# --- reports -----------------------------------------------------------------
redirect -tee $dc_dir/reports/${design}_${corner}.rpt {
  puts "==== DC corner=$corner design=$design (compile_ultra) ===="
  report_timing -path_type full -delay_type max -max_paths 10 -sort_by slack -nosplit
  report_area -hierarchy
  report_power
  report_reference
}
puts "INFO: wrote $dc_dir/reports/${design}_${corner}.rpt"
exit
