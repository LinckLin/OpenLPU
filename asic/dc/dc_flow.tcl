# ============================================================================
# dc_flow.tcl — QCore DC synthesis flow for the representative datapath / MAC
# (P10 §10).  Elaborate -> compile_ultra -> report_timing/area/power.
#
# Mirrors the OpenSTA baseline caliber by default:
#   * 1 ns probe clock on `clk` (overridable with DC_PERIOD),
#     0 ns input/output delay (sta.tcl),
#   * Fmax = 1 / critical-path data-arrival time,
#   * report_power is used for *leakage* + area only; dynamic power stays the
#     §5 activity-factor estimate (VCD back-annotation is a later step).
#
# Reads the DC-local desugared sources under asic/dc/gen/ (see desugar_dc.py).
#
# Env inputs:  DC_CORNER  tt_025C_1v80 | ss_100C_1v60
#              DC_DESIGN  synth_datapath | mac_bf16 | matrix_int8_pe
#              DC_TECH    sky130 (default) | smic28
#              DC_LABEL   optional report-name suffix
#              DC_PERIOD  optional clock period in ns (default: 1.0)
# ============================================================================
set corner [getenv DC_CORNER]
set design [getenv DC_DESIGN]
set tech "sky130"
if {[info exists env(DC_TECH)] && $env(DC_TECH) != ""} { set tech $env(DC_TECH) }
set clock_period 1.0
if {[info exists env(DC_PERIOD)] && $env(DC_PERIOD) != ""} { set clock_period $env(DC_PERIOD) }

set dc_dir asic/dc
set dw_dir /home/public/app/synopsys/syn/O-2018.06-SP1/libraries/syn

set search_path [concat $search_path \
    [list . asic/dc/gen $dc_dir $dc_dir/db $dc_dir/reports $dw_dir]]
# Keep flow labels stable while mapping SMIC28 to the foundry corner names.
if {$tech == "smic28"} {
  if {$corner == "tt_025C_1v80"} {
    set std_corner "tt_v0p9_25c"
  } elseif {$corner == "ss_100C_1v60"} {
    set std_corner "ssg_v0p81_125c"
  } else {
    puts "ERROR: unknown corner '$corner' for DC_TECH=smic28"; exit 1
  }
  set target_library [list $dc_dir/db/scc28nhkcp_hdc30p140_rvt_${std_corner}_basic.db]
} elseif {$tech == "sky130"} {
  set target_library [list $dc_dir/db/sky130_fd_sc_hd__${corner}.db]
} else {
  puts "ERROR: unknown DC_TECH '$tech'"; exit 1
}
set synthetic_library [list dw_foundation.sldb]
set link_library [concat {*} $target_library $synthetic_library]

set tag $corner
if {$tech == "smic28"} { set tag "smic28_${corner}" }
if {[info exists env(DC_LABEL)] && $env(DC_LABEL) != ""} { set tag "${tag}_$env(DC_LABEL)" }

puts "INFO: tech=$tech corner=$corner design=$design period_ns=$clock_period"
puts "INFO: target_library=$target_library"

# --- read + elaborate --------------------------------------------------------
if {$design == "synth_datapath"} {
  analyze -format sverilog asic/dc/gen/synth_datapath.sv
  elaborate synth_datapath
} elseif {$design == "mac_bf16"} {
  analyze -format sverilog asic/dc/gen/synth_mac.sv
  elaborate mac_bf16
} elseif {$design == "matrix_int8_pe"} {
  analyze -format sverilog asic/dc/gen/matrix_int8_pe_probe.sv
  elaborate matrix_int8_pe_probe
} else {
  puts "ERROR: unknown DC_DESIGN '$design'"
  exit 1
}
link
uniquify

# Preserve the physical PE boundary so report_area -hierarchy can separate the
# arithmetic cell from probe-only launch flops.  Boundary optimization would
# otherwise flatten u_pe during compile_ultra and make the scaled area caliber
# impossible to audit.
if {$design == "matrix_int8_pe"} {
  set pe_cells [get_cells u_pe]
  set_ungroup $pe_cells false
  set_boundary_optimization $pe_cells false
}

# --- constraints: configurable probe clock (1 ns by default) ----------------
create_clock -name clk -period $clock_period [get_ports clk]
set_input_delay  -clock clk 0.0 [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk 0.0 [all_outputs]
compile_ultra
write -format verilog -hierarchy -output $dc_dir/reports/${design}_${tag}.v


# --- reports -----------------------------------------------------------------
redirect -tee $dc_dir/reports/${design}_${tag}.rpt {
  puts "==== DC tech=$tech corner=$corner design=$design period_ns=$clock_period (compile_ultra) ===="
  report_timing -path_type full -delay_type max -max_paths 10 -sort_by slack -nosplit
  report_area -hierarchy
  report_power
  report_reference
}
puts "INFO: wrote $dc_dir/reports/${design}_${tag}.rpt"
exit
