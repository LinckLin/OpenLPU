// ============================================================================
// clock_reset_tb.sv — C++-driven wrapper for fpga/clock_reset.sv (no `#` timing
// delays: Verilator 4.038 is cycle-based).  sim_clock_reset.cpp toggles clk_i
// and checks rst_n's async assert + synchronous release.
// ============================================================================
module clock_reset_tb (
  input  logic clk_i,
  input  logic async_rst_n_i,
  output logic clk,
  output logic rst_n
);
  clock_reset #(.RST_STAGES(3)) dut (
    .clk_i(clk_i), .async_rst_n_i(async_rst_n_i),
    .clk(clk), .rst_n(rst_n)
  );
endmodule
