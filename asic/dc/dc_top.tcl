# ============================================================================
# dc_top.tcl — DC synthesis of the full design synth_top (P10 §10, step 5).
#
# synth_top = command_processor @ MAX_VEC=128 + 8 MiB scratchpad SRAM black box.
# The SRAM is `sram_macro` (empty module -> black box) protected with
# set_dont_touch; its area/power stay the separate OpenRAM/density estimate.
#
# The engine RTL is the DC-desugared tree asic/dc/gen_full/ (see hoist_dc.py:
# loop-break `k=-1`->`break`, variable-bound loop -> fixed bound + guard,
# module-level declaration hoist, and the full-design third transform — the
# co-sim numeric cores matrix_engine / vector_engine become `(* blackbox *)`
# macros and the CP instruction array becomes a bb_sram black box).  synth_top.sv
# / sram_macro.sv / bb_sram.sv are copied verbatim (DC-clean).
#
# NOTE: the engines are the *co-sim functional model* (runtime-bound loops,
# inferred RAMs), NOT the physical 128-lane datapath — see asic-report.md §7.
# The numeric-core macros' primitives (mac_bf16/mac_int8, synth_datapath) are
# DC-synthesized separately (§10.4); here the full design records whether DC can
# elaborate/link/compile the control plane past the storage expansion (§10.5).
#

# Env:  DC_CORNER  tt_025C_1v80 | ss_100C_1v60   (flow corner label)
#       DC_TECH    sky130 (default) | smic28     (logic std cell library)
#       DC_TOP_COMPILE  0 = elaborate+link+report only (fast frontend check)
#                       1 = full compile_ultra (bounded externally by timeout)
# ============================================================================
set corner [getenv DC_CORNER]
set tech "sky130"
if {[info exists env(DC_TECH)] && $env(DC_TECH) != ""} { set tech $env(DC_TECH) }
set compile 1
if {[info exists env(DC_TOP_COMPILE)] && $env(DC_TOP_COMPILE) == "0"} { set compile 0 }

set dc_dir asic/dc
set dw_dir /home/public/app/synopsys/syn/O-2018.06-SP1/libraries/syn
set search_path [concat $search_path \
    [list . asic/dc/gen_full asic $dc_dir $dc_dir/db $dc_dir/reports $dw_dir]]

# --- logic std cell library selection ---------------------------------------
# sky130 = 130 nm / 1.8 V (legacy reachability caliber, P10 §10).
# smic28 = 28HKCP / 0.9 V HDC30P140 RVT (new baseline, D18).  Corner mapping:
#   tt_025C_1v80 -> tt_v0p9_25c ;  ss_100C_1v60 -> ssg_v0p81_125c
# (both match the SMIC28 macro corners tt_ctypical / ssg_cworstt below).
if {$tech == "smic28"} {
  if {$corner == "tt_025C_1v80"} {
    set std_corner "tt_v0p9_25c"
  } elseif {$corner == "ss_100C_1v60"} {
    set std_corner "ssg_v0p81_125c"
  } else {
    puts "ERROR: unknown corner '$corner' for DC_TECH=smic28"; exit 1
  }
  set logic_lib [list $dc_dir/db/scc28nhkcp_hdc30p140_rvt_${std_corner}_basic.db]
} else {
  set logic_lib [list $dc_dir/db/sky130_fd_sc_hd__${corner}.db]
}

set target_library [concat $logic_lib \
    [list $dc_dir/db/kh4096x64_${corner}.db $dc_dir/db/kn128x16_${corner}.db]]
set synthetic_library [list dw_foundation.sldb]
set link_library [concat {*} $target_library $synthetic_library]

puts "INFO: tech=$tech corner=$corner compile=$compile"
puts "INFO: link_library=$link_library"
set tag $corner
if {$tech == "smic28"} { set tag "smic28_${corner}" }

# --- read + elaborate --------------------------------------------------------
# synth_top.sv `include`s qcore_pkg.sv + command_processor.sv (which transitively
# includes softfloat/matrix_engine/vector_engine/rope_lut/dma_engine/kv_addrgen);
analyze -format sverilog asic/dc/gen_full/synth_top.sv
analyze -format sverilog asic/dc/gen_full/sram_macro.sv
analyze -format sverilog asic/dc/gen_full/bb_sram.sv
elaborate synth_top
# --- black boxes: scratchpad SRAM, engine macros, instruction SRAM -----------
# set_dont_touch keeps DC from optimizing/expanding them (plan step 5; §10.5).
# kh4096x64 / kn128x16 are compiled SMIC28 SRAM macros (timing from the mapped
# .db above).  sky130 mode pairs them with 130 nm logic = cross-technology
# reachability probe; smic28 mode pairs them with 28 nm logic = same-technology
# baseline (D18).
set_dont_touch [get_cells u_sram]
set_dont_touch [get_references {sram_macro matrix_engine vector_engine bb_sram \
                                kh4096x64 kn128x16}]

# --- constraints: 1 ns probe clock (sta.tcl caliber) ------------------------
create_clock -name clk -period 1.0 [get_ports clk]
set_input_delay  -clock clk 0.0 [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk 0.0 [all_outputs]

# --- synthesize (optional; bounded by external timeout) ----------------------
if {$compile == 1} {
  compile_ultra
  write -format verilog -hierarchy -output $dc_dir/reports/synth_top_${tag}.v
}

# --- reports -----------------------------------------------------------------
redirect -tee $dc_dir/reports/synth_top_${tag}.rpt {
  puts "==== DC tech=$tech corner=$corner design=synth_top compile=$compile ===="
  report_timing -path_type full -delay_type max -max_paths 10 -sort_by slack -nosplit
  report_area -hierarchy
  report_power
  report_reference
}
puts "INFO: wrote $dc_dir/reports/synth_top_${tag}.rpt"
exit
