// ============================================================================
// qcore_top.sv — QCore top level (co-sim instance).
//
// Instantiates the co-sim memory (qmem) and the Command Processor (which owns
// the matrix/vector/dma engines + kv_addrgen).  The CP drives qmem's byte
// read/write ports; the testbench accesses qmem directly through the backdoor
// (bd_*) to preload programs/data and dump results.
//
// This is the M6 acceptance instance: full-size 128x128 matrix engine, single
// clock domain, 1 GHz (1 cyc = 1 ns).  The synthesizable 16-bank SRAM
// (sram.sv) is a separate deliverable; byte-address semantics here are
// identical to the flat array (bank arbitration affects timing only, captured
// by the frozen SRAM_*_BPC constants).
// ============================================================================
`ifndef QCORE_TOP_SV
`define QCORE_TOP_SV

`include "qcore_pkg.sv"
`include "qmem.sv"
`include "command_processor.sv"

module qcore_top #(
  parameter int NINST = 4096
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  // instruction memory backdoor
  input  logic [11:0] imem_waddr,
  input  logic        imem_we,
  input  logic [127:0] imem_wdata,
  input  logic [15:0] prog_len,
  // qmem backdoor (testbench preload/dump)
  input  logic        bd_en,
  input  logic        bd_sel,
  input  logic [39:0] bd_addr,
  input  logic [7:0]  bd_wdata,
  output logic [7:0]  bd_rdata,
  // status / trace
  output logic        done,
  output logic [63:0] total_cycles,
  output logic        trace_valid,
  output logic [15:0] trace_index,
  output logic [31:0] trace_cycles
);
  logic        cp_rd_sel;
  logic [39:0] cp_rd_addr;
  logic [7:0]  cp_rd_data;
  logic        cp_wr_en, cp_wr_sel;
  logic [39:0] cp_wr_addr;
  logic [7:0]  cp_wr_data;

  command_processor #(.NINST(NINST)) u_cp (
    .clk(clk), .rst_n(rst_n), .start(start),
    .imem_waddr(imem_waddr), .imem_we(imem_we), .imem_wdata(imem_wdata),
    .prog_len(prog_len),
    .mem_rd_sel(cp_rd_sel), .mem_rd_addr(cp_rd_addr), .mem_rd_data(cp_rd_data),
    .mem_wr_en(cp_wr_en), .mem_wr_sel(cp_wr_sel),
    .mem_wr_addr(cp_wr_addr), .mem_wr_data(cp_wr_data),
    .done(done), .total_cycles(total_cycles),
    .trace_valid(trace_valid), .trace_index(trace_index), .trace_cycles(trace_cycles)
  );

  qmem u_qmem (
    .clk(clk), .rst_n(rst_n),
    .rd_sel(cp_rd_sel), .rd_addr(cp_rd_addr), .rd_data(cp_rd_data),
    .wr_en(cp_wr_en), .wr_sel(cp_wr_sel), .wr_addr(cp_wr_addr), .wr_data(cp_wr_data),
    .bd_en(bd_en), .bd_sel(bd_sel), .bd_addr(bd_addr),
    .bd_wdata(bd_wdata), .bd_rdata(bd_rdata)
  );

endmodule
`endif // QCORE_TOP_SV
