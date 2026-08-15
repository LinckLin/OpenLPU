// ============================================================================
// clock_reset.sv — QCore FPGA clock-domain skeleton (P9, board-independent).
//
// Project convention (plans/p6-p7-plan.md §4.1, docs/p7/rtl-report.md §2):
//   * single clock domain, 1 GHz (1 cyc = 1 ns)
//   * async reset, active-low, synchronous release
//
// This module is the *only* place a reset crosses a clock boundary: the
// external async reset (async_rst_n_i) asserts the internal reset
// asynchronously (so the device can be held in reset before clocks are
// stable) and releases it synchronously on posedge clk_i through a 2-stage
// synchroniser, avoiding metastability.  The single clock is passed through
// unchanged — the board clock (or an MMCM/PLL output) drives clk_i; a DDR
// MIG clock is added at P9 freeze, still a single functional domain.
//
// Simulatable under Verilator and synthesizable (no vendor primitives).
// ============================================================================
`ifndef CLOCK_RESET_SV
`define CLOCK_RESET_SV

module clock_reset #(
  parameter int RST_STAGES = 2     // release synchroniser depth (>= 2)
) (
  input  logic clk_i,              // external single-domain clock
  input  logic async_rst_n_i,      // external async reset, active-low
  output logic clk,                // buffered single-domain clock
  output logic rst_n               // synchronized reset, active-low
);

  // Release synchroniser: asserts asynchronously (negedge of async_rst_n_i
  // drives the chain to 0 immediately), releases after RST_STAGES rising
  // edges.  The chain is reset to '0 so rst_n is a glitch-free active-low
  // reset for the whole design.
  logic [RST_STAGES-1:0] sync_ff;

  always_ff @(posedge clk_i or negedge async_rst_n_i) begin
    if (!async_rst_n_i)
      sync_ff <= {RST_STAGES{1'b0}};
    else
      sync_ff <= {sync_ff[RST_STAGES-2:0], 1'b1};
  end

  assign clk   = clk_i;
  assign rst_n = sync_ff[RST_STAGES-1];

endmodule

`endif // CLOCK_RESET_SV
