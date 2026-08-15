// ============================================================================
// ddr_axi4_master.sv — AXI4-style narrow burst master (DDR controller side).
//
// Board-facing half of ddr_if.  Converts a full-line request (DDR_BURST_BYTES)
// into an AXI4 INCR burst over a *narrow* data port (DDR_DATA_BYTES per beat;
// 64 B burst / 8 B beat = 8 beats).  Read and write fixed latencies are
// modelled as parameterised cycle counts between request issue and the first
// response data beat (the physical MIG controller's real latency is plugged
// into these parameters at P9 freeze).
//
// The narrow data width + 64 B burst mirrors the frozen HBM model
// (qcore_pkg: HBM_BURST=64, T_FIRST=100) so the board memory path matches the
// cycle model the Command Processor already charges (DMA.STORE / KV.* =
// T_FIRST + hbm_*_cycles).  In co-sim the master is idle (req_valid tied low);
// in board mode (P9 freeze) the DMA/KV data path drives `req_*`.
//
// Protocol: one transaction in flight (req_ready when IDLE), INCR burst,
//   arlen/awlen = BURST_BYTES/DATA_BYTES - 1, arsize/awsize = log2(DATA_BYTES),
//   wstrb = all-ones (full-line writes), read/write fixed latency via counter.
// ============================================================================
`ifndef DDR_AXI4_MASTER_SV
`define DDR_AXI4_MASTER_SV

module ddr_axi4_master #(
  parameter int ADDR_BITS   = 40,
  parameter int DATA_BYTES  = 8,     // narrow AXI4 data width (bytes/beat)
  parameter int BURST_BYTES = 64,    // one transaction = 64 B
  parameter int RD_LATENCY  = 100,   // modelled read  latency (cycles)
  parameter int WR_LATENCY  = 100    // modelled write latency (cycles)
) (
  input  logic clk,
  input  logic rst_n,
  // -- transaction request/response (from ddr_if control) -----------------
  input  logic                     req_valid,
  input  logic                     req_rw,      // 0 = read, 1 = write
  input  logic [ADDR_BITS-1:0]     req_addr,
  input  logic [BURST_BYTES*8-1:0] req_wdata,
  output logic                     req_ready,
  output logic                     resp_valid,
  output logic [BURST_BYTES*8-1:0] resp_rdata,
  output logic                     resp_rw,
  input  logic                     resp_ready,
  // -- AXI4 master port (toward DDR controller / MIG) ----------------------
  output logic [ADDR_BITS-1:0]  m_axi_araddr,
  output logic [7:0]            m_axi_arlen,
  output logic [2:0]            m_axi_arsize,
  output logic [1:0]            m_axi_arburst,
  output logic                  m_axi_arvalid,
  input  logic                  m_axi_arready,
  input  logic [DATA_BYTES*8-1:0] m_axi_rdata,
  input  logic [1:0]            m_axi_rresp,
  input  logic                  m_axi_rlast,
  input  logic                  m_axi_rvalid,
  output logic                  m_axi_rready,
  output logic [ADDR_BITS-1:0]  m_axi_awaddr,
  output logic [7:0]            m_axi_awlen,
  output logic [2:0]            m_axi_awsize,
  output logic [1:0]            m_axi_awburst,
  output logic                  m_axi_awvalid,
  input  logic                  m_axi_awready,
  output logic [DATA_BYTES*8-1:0] m_axi_wdata,
  output logic [DATA_BYTES-1:0] m_axi_wstrb,
  output logic                  m_axi_wlast,
  output logic                  m_axi_wvalid,
  input  logic                  m_axi_wready,
  input  logic [1:0]            m_axi_bresp,
  input  logic                  m_axi_bvalid,
  output logic                  m_axi_bready
);

  localparam int BEATS  = BURST_BYTES / DATA_BYTES;   // 64 / 8 = 8
  localparam int BLEN   = BEATS - 1;                  // AXI awlen/arlen
  localparam int BSIZE  = $clog2(DATA_BYTES);         // AXI awsize/arsize

  localparam logic [3:0]
    S_IDLE=0, S_RADDR=1, S_RLAT=2, S_RDATA=3, S_RDONE=4,
    S_WADDR=5, S_WDATA=6, S_WLAT=7, S_WRESP=8, S_WDONE=9;

  logic [3:0] state;
  logic [3:0] beat;                    // 0 .. BEATS-1
  logic [31:0] lat_cnt;
  logic        rw_q;
  logic [ADDR_BITS-1:0]     addr_q;
  logic [BURST_BYTES*8-1:0] wdata_q;
  logic [BURST_BYTES*8-1:0] rdata_q;

  assign req_ready = (state == S_IDLE);

  // read address channel
  assign m_axi_araddr  = addr_q;
  assign m_axi_arlen   = 8'(BLEN);
  assign m_axi_arsize  = 3'(BSIZE);
  assign m_axi_arburst = 2'b01;                 // INCR
  assign m_axi_arvalid = (state == S_RADDR);

  // write address channel
  assign m_axi_awaddr  = addr_q;
  assign m_axi_awlen   = 8'(BLEN);
  assign m_axi_awsize  = 3'(BSIZE);
  assign m_axi_awburst = 2'b01;                 // INCR
  assign m_axi_awvalid = (state == S_WADDR);

  // write data channel (full-line writes: wstrb = all ones)
  assign m_axi_wdata  = wdata_q[(beat + 1) * DATA_BYTES * 8 - 1 -: DATA_BYTES * 8];
  assign m_axi_wstrb  = {DATA_BYTES{1'b1}};
  assign m_axi_wlast  = (beat == BLEN[3:0]);
  assign m_axi_wvalid = (state == S_WDATA);

  assign m_axi_rready = (state == S_RDATA);
  assign m_axi_bready = (state == S_WRESP);

  // response handshake (pulsed one cycle, held until resp_ready)
  assign resp_valid  = (state == S_RDONE) || (state == S_WDONE);
  assign resp_rw     = rw_q;
  assign resp_rdata  = rdata_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      beat    <= 4'b0;
      lat_cnt <= 32'b0;
      rw_q    <= 1'b0;
      addr_q  <= {ADDR_BITS{1'b0}};
      wdata_q <= {BURST_BYTES*8{1'b0}};
      rdata_q <= {BURST_BYTES*8{1'b0}};
    end else begin
      case (state)
        S_IDLE: if (req_valid) begin
          rw_q    <= req_rw;
          addr_q  <= req_addr;
          wdata_q <= req_wdata;
          beat    <= 4'b0;
          state   <= req_rw ? S_WADDR : S_RADDR;
        end

        S_RADDR: if (m_axi_arready) begin
          lat_cnt <= RD_LATENCY;
          state   <= S_RLAT;
        end

        S_RLAT: begin
          if (lat_cnt == 0) begin beat <= 4'b0; state <= S_RDATA; end
          else lat_cnt <= lat_cnt - 1;
        end

        S_RDATA: if (m_axi_rvalid && m_axi_rready) begin
          rdata_q[beat * DATA_BYTES * 8 +: DATA_BYTES * 8] <= m_axi_rdata;
          if (m_axi_rlast) state <= S_RDONE;
          else             beat  <= beat + 1;
        end

        S_RDONE: if (resp_ready) state <= S_IDLE;

        S_WADDR: if (m_axi_awready) begin
          beat  <= 4'b0;
          state <= S_WDATA;
        end

        S_WDATA: if (m_axi_wvalid && m_axi_wready) begin
          if (m_axi_wlast) begin
            lat_cnt <= WR_LATENCY;
            state   <= S_WLAT;
          end else beat <= beat + 1;
        end

        S_WLAT: begin
          if (lat_cnt == 0) state <= S_WRESP;
          else lat_cnt <= lat_cnt - 1;
        end

        S_WRESP: if (m_axi_bvalid && m_axi_bready) state <= S_WDONE;

        S_WDONE: if (resp_ready) state <= S_IDLE;

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`endif // DDR_AXI4_MASTER_SV
