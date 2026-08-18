// ============================================================================
// synth_top.sv — QCore ASIC synthesis top (P10 / M9).
//
// The co-sim `qcore_top` uses `qmem` (flat byte SRAM + sparse associative HBM)
// which is NOT synthesizable.  This top is the *physical-scale* synthesis
// instance:
//
//   * command_processor instantiated at MAX_VEC = 128 — the frozen physical
//     vector-engine lane width (qcore_pkg LANES = 128, spec 04 §3.2; the
//     co-sim's MAX_VEC = 4096 is a functional flattening that serializes the
//     vector op in one cycle, charged ceil(len/128) cycles by the cycle model).
//   * 8 MiB scratchpad SRAM = black-box macro (sram_macro.sv).  Area/power are
//     estimated separately in docs/p10/asic-report.md (OpenRAM / density),
//     NOT synthesized as gates.
//   * HBM (16 GiB, D3) = off-die; exposed as byte-level I/O pins (the HBM PHY
//     is a licensed macro, out of scope).
//
// In the current SMIC28 full-top flow, the command/control logic and the
// matrix-engine state/control shell are synthesized, and that shell contains
// nine characterized kh4096x64 SRAM macros.  The 128x128 matrix arithmetic
// array and 128-lane vector numeric core remain physical-core black boxes;
// their representative BF16/INT8 primitives are synthesized separately.
// The legacy sky130 flow remains available for comparison.
//
// Source of truth: rtl/ref/asicsnap/ (frozen snapshot; rtl/ is read-only here).
// ============================================================================
`ifndef SYNTH_TOP_SV
`define SYNTH_TOP_SV

`include "qcore_pkg.sv"
`include "command_processor.sv"

module synth_top #(
  parameter int NINST  = 4096,
  parameter int MAX_VEC = 128     // physical vector lane width (spec LANES=128)
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  // ---- HBM off-die interface (byte-level; PHY is out of scope) ------------
  input  logic [7:0]  hbm_rd_data,     // read data return (async in co-sim model)
  output logic [39:0] hbm_addr,        // shared read/write address
  output logic        hbm_we,
  output logic [7:0]  hbm_wdata,
  // ---- status / trace ------------------------------------------------------
  output logic        done,
  output logic [63:0] total_cycles,
  output logic        trace_valid,
  output logic [15:0] trace_index,
  output logic [31:0] trace_cycles
);
  import qcore_pkg::*;

  // Command-processor memory port (byte-level, co-sim semantics).
  logic        cp_rd_sel;
  logic [39:0] cp_rd_addr;
  logic [7:0]  cp_rd_data;
  logic        cp_wr_en, cp_wr_sel;
  logic [39:0] cp_wr_addr;
  logic [7:0]  cp_wr_data;

  command_processor #(.NINST(NINST), .MAX_VEC(MAX_VEC)) u_cp (
    .clk(clk), .rst_n(rst_n), .start(start),
    .imem_waddr(12'b0), .imem_we(1'b0), .imem_wdata(128'b0), .prog_len(16'b0),
    .mem_rd_sel(cp_rd_sel), .mem_rd_addr(cp_rd_addr), .mem_rd_data(cp_rd_data),
    .mem_wr_en(cp_wr_en), .mem_wr_sel(cp_wr_sel),
    .mem_wr_addr(cp_wr_addr), .mem_wr_data(cp_wr_data),
    .done(done), .total_cycles(total_cycles),
    .trace_valid(trace_valid), .trace_index(trace_index), .trace_cycles(trace_cycles)
  );

  // ---- 8 MiB scratchpad SRAM macro (black box; estimated separately) -------
  logic [7:0] sram_rd_data;
  logic       sram_wr_en;
  assign sram_wr_en = cp_wr_en && ~cp_wr_sel;

  sram_macro u_sram (
    .clk(clk),
    .addr(sram_wr_en ? cp_wr_addr[22:0] : cp_rd_addr[22:0]),
    .rd_data(sram_rd_data),
    .wr_data(cp_wr_data),
    .wr_en(sram_wr_en)
  );

  // Read mux: SRAM vs HBM (co-sim combinational read semantics).
  assign cp_rd_data = cp_rd_sel ? hbm_rd_data : sram_rd_data;

  // ---- HBM off-die pins ----------------------------------------------------
  assign hbm_addr  = cp_rd_sel ? cp_rd_addr : cp_wr_addr;
  assign hbm_we    = cp_wr_en && cp_wr_sel;
  assign hbm_wdata = cp_wr_data;

endmodule
`endif // SYNTH_TOP_SV
