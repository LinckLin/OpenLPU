// ============================================================================
// ddr_if.sv — QCore DDR memory-subsystem abstraction (P9, board-independent).
//
// Replaces qmem's HBM model: the byte-addressable memory interface the Command
// Processor already drives (rd_sel 0=SRAM, 1=DDR; byte read combinational /
// byte write posedge) is preserved 1:1, so the unmodified CP runs unchanged.
//
// Memory map (mirrors qmem, spec 03):
//   * SRAM  (rd_sel=0): on-chip scratchpad, flat byte array.  On the board this
//     is UltraRAM/BRAM (ZCU104 URAM 3.38 MiB + BRAM 1.37 MiB, ~59% of the 8 MiB
//     budget — the shrink is a P9-freeze decision, see docs/p9/porting.md).
//   * DDR   (rd_sel=1): off-chip DDR window, sparse associative model (only
//     touched blocks exist — mirrors qsim's SparseMemory).  On the board this is
//     the DDR4 controller (MIG), reached through the AXI4-style narrow master
//     port below.
//
// Board-facing side — AXI4-style narrow master port (64 B burst, parameterised
// latency).  In co-sim (this build) the master is idle (req_valid tied low) and
// the functional arrays above serve engine+host traffic with qmem-identical
// zero-latency semantics, so the M6 co-sim criterion (bf16 <= 1 ULP, trace
// identical) is preserved exactly.  At P9 freeze the DMA/KV data path drives
// the master's request interface (write-combining line buffer + read staging),
// and the physical MIG replaces the sparse array — only physical-IP adaptation.
//
// Host port (hd_*) replaces qmem's testbench backdoor (bd_*): it is the
// register-mapped qbin-load / logits-readback path driven by host_if.
// ============================================================================
`ifndef DDR_IF_SV
`define DDR_IF_SV

`include "ddr_axi4_master.sv"

module ddr_if #(
  parameter int SRAM_BYTES      = 8 * 1024 * 1024,      // on-chip scratchpad
  parameter int DDR_BYTES       = 1024 * 1024 * 1024,   // DDR window (1 GiB)
  parameter int DDR_ADDR_BITS   = 40,
  parameter int DDR_DATA_BYTES  = 8,                    // narrow AXI4 width
  parameter int DDR_BURST_BYTES = 64,                   // 64 B burst
  parameter int DDR_RD_LATENCY  = 100,                  // = qcore_pkg T_FIRST
  parameter int DDR_WR_LATENCY  = 100
) (
  input  logic clk,
  input  logic rst_n,
  // -- engine-facing byte access (identical to qmem) ----------------------
  input  logic                 rd_sel,       // 0 = SRAM, 1 = DDR
  input  logic [DDR_ADDR_BITS-1:0] rd_addr,  // SRAM: [SRAM_ADDR_BITS-1:0]
  output logic [7:0]           rd_data,
  input  logic                 wr_en,
  input  logic                 wr_sel,
  input  logic [DDR_ADDR_BITS-1:0] wr_addr,
  input  logic [7:0]           wr_data,
  // -- host-facing byte access (qbin load / logits readback) --------------
  input  logic                 hd_en,        // 1 = write, 0 = read
  input  logic                 hd_sel,
  input  logic [DDR_ADDR_BITS-1:0] hd_addr,
  input  logic [7:0]           hd_wdata,
  output logic [7:0]           hd_rdata,
  // -- AXI4-style narrow master port (toward DDR controller / MIG) --------
  output logic [DDR_ADDR_BITS-1:0] m_axi_araddr,
  output logic [7:0]            m_axi_arlen,
  output logic [2:0]            m_axi_arsize,
  output logic [1:0]            m_axi_arburst,
  output logic                  m_axi_arvalid,
  input  logic                  m_axi_arready,
  input  logic [DDR_DATA_BYTES*8-1:0] m_axi_rdata,
  input  logic [1:0]            m_axi_rresp,
  input  logic                  m_axi_rlast,
  input  logic                  m_axi_rvalid,
  output logic                  m_axi_rready,
  output logic [DDR_ADDR_BITS-1:0] m_axi_awaddr,
  output logic [7:0]            m_axi_awlen,
  output logic [2:0]            m_axi_awsize,
  output logic [1:0]            m_axi_awburst,
  output logic                  m_axi_awvalid,
  input  logic                  m_axi_awready,
  output logic [DDR_DATA_BYTES*8-1:0] m_axi_wdata,
  output logic [DDR_DATA_BYTES-1:0]   m_axi_wstrb,
  output logic                  m_axi_wlast,
  output logic                  m_axi_wvalid,
  input  logic                  m_axi_wready,
  input  logic [1:0]            m_axi_bresp,
  input  logic                  m_axi_bvalid,
  output logic                  m_axi_bready
);

  localparam int SRAM_ADDR_BITS = $clog2(SRAM_BYTES);   // 23 (8 MiB)

  logic [7:0] sram [0:SRAM_BYTES-1];
  logic [7:0] ddr  [bit [DDR_ADDR_BITS-1:0]];           // sparse associative

  // Engine combinational read.
  always_comb begin
    rd_data = 8'b0;
    if (rd_sel)      rd_data = ddr[rd_addr];
    else             rd_data = sram[rd_addr[SRAM_ADDR_BITS-1:0]];
  end

  // Engine synchronous write.
  always_ff @(posedge clk) begin
    if (wr_en) begin
      if (wr_sel)    ddr[wr_addr] <= wr_data;
      else           sram[wr_addr[SRAM_ADDR_BITS-1:0]] <= wr_data;
    end
  end

  // Host combinational read + synchronous write (qbin load / logits readback).
  always_comb begin
    hd_rdata = 8'b0;
    if (hd_sel)      hd_rdata = ddr[hd_addr];
    else             hd_rdata = sram[hd_addr[SRAM_ADDR_BITS-1:0]];
  end

  always_ff @(posedge clk) begin
    if (hd_en) begin
      if (hd_sel)    ddr[hd_addr] <= hd_wdata;
      else           sram[hd_addr[SRAM_ADDR_BITS-1:0]] <= hd_wdata;
    end
  end

  // Board-facing AXI4 master.  Idle in co-sim (req_valid tied low); the P9
  // freeze data path drives it (see header).  The master itself is unit-tested
  // standalone (fpga/tb/ddr_axi4_tb.sv) against a behavioral DDR slave.
  ddr_axi4_master #(
    .ADDR_BITS  (DDR_ADDR_BITS),
    .DATA_BYTES (DDR_DATA_BYTES),
    .BURST_BYTES(DDR_BURST_BYTES),
    .RD_LATENCY (DDR_RD_LATENCY),
    .WR_LATENCY (DDR_WR_LATENCY)
  ) u_axi (
    .clk(clk), .rst_n(rst_n),
    .req_valid(1'b0), .req_rw(1'b0), .req_addr({DDR_ADDR_BITS{1'b0}}),
    .req_wdata({DDR_BURST_BYTES*8{1'b0}}),
    .req_ready(), .resp_valid(), .resp_rdata(), .resp_rw(), .resp_ready(1'b0),
    .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
    .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
    .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
    .m_axi_rready(m_axi_rready),
    .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
    .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
    .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
    .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
    .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
    .m_axi_wready(m_axi_wready),
    .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
    .m_axi_bready(m_axi_bready)
  );

endmodule

`endif // DDR_IF_SV
