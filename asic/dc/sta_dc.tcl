# sta_dc.tcl — OpenSTA on the DC-produced netlist (cross-validation, P10 §10).
#
# Same caliber as asic/sta.tcl (1 ns probe clock, 0 in/out delay, Fmax = 1 /
# arrival), but reads the DC compile_ultra netlist instead of the Yosys netlist.
# This isolates the *synthesis-tool* difference (Yosys abc vs DC compile_ultra)
# from the *STA-tool* difference (OpenSTA vs DC's timing engine): if OpenSTA on
# the DC netlist still reports ~3 ns, the gap is in the netlist, not the STA.
#
# Env:  LIB        path to sky130 corner liberty
#       STA_NETLIST path to the DC netlist (.v)
read_liberty $env(LIB)
read_verilog $env(STA_NETLIST)
link_design synth_datapath
create_clock -name clk -period 1.0 [get_ports clk]
set_input_delay  -clock clk 0.0 [all_inputs]
set_output_delay -clock clk 0.0 [all_outputs]
report_checks -path_delay max -format full_clock_expanded -digits 4 -group_count 6
exit
