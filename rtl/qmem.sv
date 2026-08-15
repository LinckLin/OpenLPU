// ============================================================================
// qmem.sv — QCore co-sim memory: SRAM (flat 8 MiB) + HBM (sparse associative).
//
// The 16-bank SRAM structure (sram.sv) is the synthesizable microarchitecture
// deliverable; byte-address semantics are identical to a flat array, and bank
// arbitration only affects *timing* (already captured by the frozen SRAM_*_BPC
// constants in qcore_pkg).  The co-sim therefore uses a flat byte array for
// SRAM and a sparse associative array for HBM (only touched blocks exist,
// mirroring qsim's SparseMemory).
//
// Ports:
//   * byte read (combinational): rd_sel (0=SRAM,1=HBM) + rd_addr -> rd_data
//   * byte write (posedge):      wr_en + wr_sel + wr_addr + wr_data
//   * backdoor (testbench preload/dump, posedge): bd_* with same layout
// ============================================================================
`ifndef QMEM_SV
`define QMEM_SV

`include "qcore_pkg.sv"
import qcore_pkg::*;

module qmem (
  input  logic        clk,
  input  logic        rst_n,
  // -- engine-facing byte access -------------------------------------
  input  logic        rd_sel,          // 0 = SRAM, 1 = HBM
  input  logic [39:0] rd_addr,         // SRAM: [22:0]; HBM: [39:0]
  output logic [7:0]  rd_data,
  input  logic        wr_en,
  input  logic        wr_sel,
  input  logic [39:0] wr_addr,
  input  logic [7:0]  wr_data,
  // -- testbench backdoor --------------------------------------------
  input  logic        bd_en,           // 1 = write, 0 = read
  input  logic        bd_sel,
  input  logic [39:0] bd_addr,
  input  logic [7:0]  bd_wdata,
  output logic [7:0]  bd_rdata
);
  logic [7:0] sram [0:SRAM_BYTES-1];
  logic [7:0] hbm  [bit[39:0]];

  // Engine combinational read.
  always_comb begin
    rd_data = 8'b0;
    if (rd_sel)      rd_data = hbm[rd_addr];
    else             rd_data = sram[rd_addr[22:0]];
  end

  // Engine synchronous write.
  always_ff @(posedge clk) begin
    if (wr_en) begin
      if (wr_sel)    hbm[wr_addr] <= wr_data;
      else           sram[wr_addr[22:0]] <= wr_data;
    end
  end

  // Backdoor read (combinational) + write (posedge).
  always_comb begin
    bd_rdata = 8'b0;
    if (bd_sel)      bd_rdata = hbm[bd_addr];
    else             bd_rdata = sram[bd_addr[22:0]];
  end

  always_ff @(posedge clk) begin
    if (bd_en) begin
      if (bd_sel)    hbm[bd_addr] <= bd_wdata;
      else           sram[bd_addr[22:0]] <= bd_wdata;
    end
  end
endmodule
`endif // QMEM_SV
