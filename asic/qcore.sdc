# ============================================================================
# qcore.sdc — QCore ASIC timing constraints (P10 / M9).
#
# Target: single clock domain, 1 GHz (1 cyc = 1 ns), frozen cycle model
# (qcore_pkg §2, docs/p7).  Inputs/outputs are the HBM byte interface + status;
# the 8 MiB SRAM is an internal macro (modelled via mem_stub.lib).
# ============================================================================

# --- clock -------------------------------------------------------------------
create_clock -name clk -period 1.000 [get_ports clk]

# --- reset -------------------------------------------------------------------
set_input_delay -clock clk 0.100 [get_ports rst_n]
set_false_path -from [get_ports rst_n]

# --- HBM byte interface + start (async inputs; conservative margins) --------
set_input_delay  -clock clk -max 0.300 [get_ports {hbm_rd_data start}]
set_input_delay  -clock clk -min 0.000 [get_ports {hbm_rd_data start}]
set_output_delay -clock clk -max 0.300 [get_ports {hbm_addr hbm_we hbm_wdata}]
set_output_delay -clock clk -min 0.000 [get_ports {hbm_addr hbm_we hbm_wdata}]

# --- status/trace outputs (unconstrained, debug-only) ------------------------
set_false_path -to [get_ports {done total_cycles trace_valid trace_index trace_cycles}]
