# sta.tcl — QCore P10b STA (OpenSTA + sky130 multi-corner).
# P10b: the datapath is now clocked/pipelined (synth_datapath.sv takes clk/rst_n).
# The critical path is the worst pipeline stage (reg->reg or input->reg /
# reg->output).  We time at a 1 ns probe clock and read the data arrival time;
# Fmax = 1 / arrival_time.  (The 1 GHz cycle-model clock is the frozen model;
# the physical Fmax is reported honestly in docs/p10/asic-report.md.)
read_liberty $env(LIB)
read_verilog asic/netlist/synth_datapath.v
link_design synth_datapath
create_clock -name clk -period 1.0 [get_ports clk]
set_input_delay  -clock clk 0.0 [all_inputs]
set_output_delay -clock clk 0.0 [all_outputs]
report_checks -path_delay max -format full_clock_expanded -digits 4 -group_count 6
exit
