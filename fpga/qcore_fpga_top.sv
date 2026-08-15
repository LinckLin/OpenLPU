// ============================================================================
// qcore_fpga_top.sv — QCore FPGA prototype top (P9, board-independent).
//
// Wraps the unmodified QCore Command Processor (rtl/command_processor.sv) in
// the P9 board-independent interface layer:
//
//   clock_reset  — single clock domain + async reset (active-low, sync release)
//   host_if      — host control plane (config / cmd-queue load / qbin load /
//                  logits readback) as a flat register file
//   ddr_if       — DDR memory subsystem (replaces qmem's HBM model) with an
//                  AXI4-style narrow master port (64 B burst, parameterised
//                  latency) toward the board DDR controller (MIG)
//
// The CP's byte-level memory port and imem backdoor are preserved, so the M6
// co-sim criterion (per-instruction cycle trace identical to qsim baseline,
// bf16 <= 1 ULP) holds with the same instruction programs and initial memory.
//
// Observability ports (done / total_cycles / trace_*) mirror qcore_top for the
// testbench; the host reads the same values through host_if's STATUS / TOTAL /
// TRACE registers (logits readback path).  The AXI4 master port is the board
// DDR interface (idle in co-sim; unit-tested standalone).
// ============================================================================
`ifndef QCORE_FPGA_TOP_SV
`define QCORE_FPGA_TOP_SV

`include "command_processor.sv"
`include "clock_reset.sv"
`include "ddr_if.sv"
`include "host_if.sv"

module qcore_fpga_top #(
  parameter int NINST      = 4096,
  parameter int SRAM_BYTES = 8 * 1024 * 1024   // on-chip scratchpad (P10b shrinkable)
) (
  input  logic clk_i,
  input  logic async_rst_n_i,
  // -- host register interface ---------------------------------------------
  input  logic [11:0] host_addr,
  input  logic [31:0] host_wdata,
  input  logic        host_wen,
  input  logic        host_ren,
  output logic [31:0] host_rdata,
  output logic        host_ready,
  // -- AXI4-style narrow DDR master port (toward MIG) -----------------------
  output logic [39:0] m_axi_araddr,
  output logic [7:0]  m_axi_arlen,
  output logic [2:0]  m_axi_arsize,
  output logic [1:0]  m_axi_arburst,
  output logic        m_axi_arvalid,
  input  logic        m_axi_arready,
  input  logic [63:0] m_axi_rdata,
  input  logic [1:0]  m_axi_rresp,
  input  logic        m_axi_rlast,
  input  logic        m_axi_rvalid,
  output logic        m_axi_rready,
  output logic [39:0] m_axi_awaddr,
  output logic [7:0]  m_axi_awlen,
  output logic [2:0]  m_axi_awsize,
  output logic [1:0]  m_axi_awburst,
  output logic        m_axi_awvalid,
  input  logic        m_axi_awready,
  output logic [63:0] m_axi_wdata,
  output logic [7:0]  m_axi_wstrb,
  output logic        m_axi_wlast,
  output logic        m_axi_wvalid,
  input  logic        m_axi_wready,
  input  logic [1:0]  m_axi_bresp,
  input  logic        m_axi_bvalid,
  output logic        m_axi_bready,
  // -- observability (mirror qcore_top) -------------------------------------
  output logic        done,
  output logic [63:0] total_cycles,
  output logic        trace_valid,
  output logic [15:0] trace_index,
  output logic [31:0] trace_cycles
);

  logic clk, rst_n;

  clock_reset u_cr (
    .clk_i(clk_i), .async_rst_n_i(async_rst_n_i),
    .clk(clk), .rst_n(rst_n)
  );

  // CP <-> host_if
  logic         start;
  logic [11:0]  imem_waddr;
  logic         imem_we;
  logic [127:0] imem_wdata;
  logic [15:0]  prog_len;

  // CP <-> ddr_if
  logic        cp_rd_sel;
  logic [39:0] cp_rd_addr;
  logic [7:0]  cp_rd_data;
  logic        cp_wr_en, cp_wr_sel;
  logic [39:0] cp_wr_addr;
  logic [7:0]  cp_wr_data;

  // host_if <-> ddr_if
  logic        hd_en, hd_sel;
  logic [39:0] hd_addr;
  logic [7:0]  hd_wdata, hd_rdata;

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

  host_if #(.NINST(NINST)) u_host (
    .clk(clk), .rst_n(rst_n),
    .host_addr(host_addr), .host_wdata(host_wdata),
    .host_wen(host_wen), .host_ren(host_ren),
    .host_rdata(host_rdata), .host_ready(host_ready),
    .start(start),
    .imem_waddr(imem_waddr), .imem_we(imem_we), .imem_wdata(imem_wdata),
    .prog_len(prog_len),
    .done(done), .total_cycles(total_cycles),
    .trace_valid(trace_valid), .trace_index(trace_index), .trace_cycles(trace_cycles),
    .hd_en(hd_en), .hd_sel(hd_sel), .hd_addr(hd_addr),
    .hd_wdata(hd_wdata), .hd_rdata(hd_rdata)
  );
  ddr_if #(.SRAM_BYTES(SRAM_BYTES)) u_ddr (
    .clk(clk), .rst_n(rst_n),
    .rd_sel(cp_rd_sel), .rd_addr(cp_rd_addr), .rd_data(cp_rd_data),
    .wr_en(cp_wr_en), .wr_sel(cp_wr_sel), .wr_addr(cp_wr_addr), .wr_data(cp_wr_data),
    .hd_en(hd_en), .hd_sel(hd_sel), .hd_addr(hd_addr),
    .hd_wdata(hd_wdata), .hd_rdata(hd_rdata),
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

`endif // QCORE_FPGA_TOP_SV
