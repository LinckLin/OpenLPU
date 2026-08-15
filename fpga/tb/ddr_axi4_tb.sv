// ============================================================================
// ddr_axi4_tb.sv — self-checking testbench for fpga/ddr_axi4_master.sv.
//
// Instantiates the AXI4-style narrow burst master and a minimal behavioral
// AXI4 slave memory, then runs a clocked FSM (Verilator 4.038 is cycle-based,
// so no `#` timing delays) that:
//   1. issues a 64 B write burst and checks the write response,
//   2. issues a 64 B read burst and checks the returned data matches,
//   3. checks the parameterised latency is honoured (response not before
//      BEATS + latency cycles after the request was accepted).
//
// clk is toggled by sim_ddr_axi4.cpp; the FSM drives done/err_count outputs.
// ============================================================================

// minimal AXI4 slave memory (zero-latency protocol; the master models the DDR
// latency).  Always-ready address channels, combinational read from a sparse
// byte array, per-beat byte-strobe writes.
module axi4_slave_mem #(
  parameter int ADDR_BITS = 40,
  parameter int DATA_BYTES = 8,
  parameter int BEATS = 8
) (
  input  logic clk, rst_n,
  input  logic [ADDR_BITS-1:0] s_axi_araddr,
  input  logic [7:0] s_axi_arlen,
  input  logic [2:0] s_axi_arsize,
  input  logic [1:0] s_axi_arburst,
  input  logic s_axi_arvalid,
  output logic s_axi_arready,
  output logic [DATA_BYTES*8-1:0] s_axi_rdata,
  output logic [1:0] s_axi_rresp,
  output logic s_axi_rlast,
  output logic s_axi_rvalid,
  input  logic s_axi_rready,
  input  logic [ADDR_BITS-1:0] s_axi_awaddr,
  input  logic [7:0] s_axi_awlen,
  input  logic [2:0] s_axi_awsize,
  input  logic [1:0] s_axi_awburst,
  input  logic s_axi_awvalid,
  output logic s_axi_awready,
  input  logic [DATA_BYTES*8-1:0] s_axi_wdata,
  input  logic [DATA_BYTES-1:0] s_axi_wstrb,
  input  logic s_axi_wlast,
  input  logic s_axi_wvalid,
  output logic s_axi_wready,
  output logic [1:0] s_axi_bresp,
  output logic s_axi_bvalid,
  input  logic s_axi_bready
);
  logic [7:0] mem [bit [ADDR_BITS-1:0]];

  localparam int BLEN = BEATS - 1;

  logic [ADDR_BITS-1:0] ra_q;
  logic [3:0] rbeat;
  logic       rbusy;
  logic [ADDR_BITS-1:0] wa_q;
  logic [3:0] wbeat;
  logic       wbusy;
  logic       wdone;

  assign s_axi_arready = !rbusy;
  assign s_axi_rresp   = 2'b00;
  assign s_axi_rvalid  = rbusy;
  assign s_axi_rlast   = (rbeat == BLEN[3:0]);
  assign s_axi_rdata   = {mem[ra_q + rbeat * DATA_BYTES + 7],
                          mem[ra_q + rbeat * DATA_BYTES + 6],
                          mem[ra_q + rbeat * DATA_BYTES + 5],
                          mem[ra_q + rbeat * DATA_BYTES + 4],
                          mem[ra_q + rbeat * DATA_BYTES + 3],
                          mem[ra_q + rbeat * DATA_BYTES + 2],
                          mem[ra_q + rbeat * DATA_BYTES + 1],
                          mem[ra_q + rbeat * DATA_BYTES + 0]};

  assign s_axi_awready = !wbusy;
  assign s_axi_wready  = wbusy;
  assign s_axi_bresp   = 2'b00;
  assign s_axi_bvalid  = wdone;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rbusy <= 0; rbeat <= 0; ra_q <= 0;
      wbusy <= 0; wbeat <= 0; wa_q <= 0; wdone <= 0;
    end else begin
      if (!rbusy && s_axi_arvalid && s_axi_arready) begin
        rbusy <= 1; rbeat <= 0; ra_q <= s_axi_araddr;
      end else if (rbusy && s_axi_rvalid && s_axi_rready) begin
        if (s_axi_rlast) rbusy <= 0;
        else             rbeat <= rbeat + 1;
      end
      if (!wbusy && s_axi_awvalid && s_axi_awready) begin
        wbusy <= 1; wbeat <= 0; wa_q <= s_axi_awaddr; wdone <= 0;
      end else if (wbusy && s_axi_wvalid && s_axi_wready) begin
        if (s_axi_wstrb[0]) mem[wa_q + wbeat * DATA_BYTES + 0] <= s_axi_wdata[7:0];
        if (s_axi_wstrb[1]) mem[wa_q + wbeat * DATA_BYTES + 1] <= s_axi_wdata[15:8];
        if (s_axi_wstrb[2]) mem[wa_q + wbeat * DATA_BYTES + 2] <= s_axi_wdata[23:16];
        if (s_axi_wstrb[3]) mem[wa_q + wbeat * DATA_BYTES + 3] <= s_axi_wdata[31:24];
        if (s_axi_wstrb[4]) mem[wa_q + wbeat * DATA_BYTES + 4] <= s_axi_wdata[39:32];
        if (s_axi_wstrb[5]) mem[wa_q + wbeat * DATA_BYTES + 5] <= s_axi_wdata[47:40];
        if (s_axi_wstrb[6]) mem[wa_q + wbeat * DATA_BYTES + 6] <= s_axi_wdata[55:48];
        if (s_axi_wstrb[7]) mem[wa_q + wbeat * DATA_BYTES + 7] <= s_axi_wdata[63:56];
        if (s_axi_wlast) begin wdone <= 1; wbusy <= 0; end
        else             wbeat <= wbeat + 1;
      end
      if (wdone && s_axi_bvalid && s_axi_bready) wdone <= 0;
    end
  end
endmodule


module ddr_axi4_tb #(
  parameter int ADDR_BITS   = 40,
  parameter int DATA_BYTES  = 8,
  parameter int BURST_BYTES = 64,
  parameter int RD_LATENCY  = 7,
  parameter int WR_LATENCY  = 5
) (
  input  logic clk,
  input  logic rst_n,
  output logic done,
  output logic [31:0] err_count
);
  localparam int BEATS = BURST_BYTES / DATA_BYTES;   // 8

  logic req_valid, req_rw, req_ready;
  logic [ADDR_BITS-1:0] req_addr;
  logic [BURST_BYTES*8-1:0] req_wdata;
  logic resp_valid, resp_rw, resp_ready;
  logic [BURST_BYTES*8-1:0] resp_rdata;
  logic [ADDR_BITS-1:0] araddr, awaddr;
  logic [7:0] arlen, awlen;
  logic [2:0] arsize, awsize;
  logic [1:0] arburst, awburst;
  logic arvalid, arready, awvalid, awready;
  logic [DATA_BYTES*8-1:0] rdata, wdata;
  logic [DATA_BYTES-1:0] wstrb;
  logic [1:0] rresp, bresp;
  logic rlast, rvalid, rready;
  logic wlast, wvalid, wready, bvalid, bready;

  ddr_axi4_master #(
    .ADDR_BITS(ADDR_BITS), .DATA_BYTES(DATA_BYTES),
    .BURST_BYTES(BURST_BYTES), .RD_LATENCY(RD_LATENCY), .WR_LATENCY(WR_LATENCY)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .req_valid(req_valid), .req_rw(req_rw), .req_addr(req_addr),
    .req_wdata(req_wdata), .req_ready(req_ready),
    .resp_valid(resp_valid), .resp_rdata(resp_rdata), .resp_rw(resp_rw),
    .resp_ready(resp_ready),
    .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize),
    .m_axi_arburst(arburst), .m_axi_arvalid(arvalid), .m_axi_arready(arready),
    .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid), .m_axi_rready(rready),
    .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize),
    .m_axi_awburst(awburst), .m_axi_awvalid(awvalid), .m_axi_awready(awready),
    .m_axi_wdata(wdata), .m_axi_wstrb(wstrb), .m_axi_wlast(wlast),
    .m_axi_wvalid(wvalid), .m_axi_wready(wready),
    .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready)
  );

  axi4_slave_mem #(
    .ADDR_BITS(ADDR_BITS), .DATA_BYTES(DATA_BYTES), .BEATS(BEATS)
  ) slave (
    .clk(clk), .rst_n(rst_n),
    .s_axi_araddr(araddr), .s_axi_arlen(arlen), .s_axi_arsize(arsize),
    .s_axi_arburst(arburst), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rlast(rlast),
    .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .s_axi_awaddr(awaddr), .s_axi_awlen(awlen), .s_axi_awsize(awsize),
    .s_axi_awburst(awburst), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wlast(wlast),
    .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready)
  );

  // -- test FSM ----------------------------------------------------------
  localparam logic [2:0]
    S_WRITE_REQ = 0, S_WRITE_WAIT = 1, S_READ_REQ = 2, S_READ_WAIT = 3, S_FINISH = 4;

  logic [2:0]  state;
  logic [31:0] errors;
  logic [31:0] wait_cyc;
  logic [BURST_BYTES*8-1:0] golden;
  integer i;

  assign req_valid  = (state == S_WRITE_REQ) || (state == S_READ_REQ);
  assign req_rw     = (state == S_WRITE_REQ);
  assign req_addr   = {{(ADDR_BITS-16){1'b0}}, 16'h1234};
  assign req_wdata  = golden;
  assign resp_ready = 1'b1;

  assign done      = (state == S_FINISH);
  assign err_count = errors;

  initial begin
    golden = {BURST_BYTES*8{1'b0}};
    for (i = 0; i < BURST_BYTES; i++)
      golden[i*8 +: 8] = 8'(i * 7 + 3);
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_WRITE_REQ; errors <= 0; wait_cyc <= 0;
    end else begin
      case (state)
        S_WRITE_REQ: if (req_ready) begin
          wait_cyc <= 0;
          state <= S_WRITE_WAIT;
        end

        S_WRITE_WAIT: begin
          if (resp_valid) begin
            if (resp_rw !== 1'b1) begin
              $display("FAIL: write resp_rw=%b", resp_rw);
              errors <= errors + 1;
            end
            if (wait_cyc < BEATS + WR_LATENCY) begin
              $display("FAIL: write latency wait_cyc=%0d < %0d", wait_cyc, BEATS + WR_LATENCY);
              errors <= errors + 1;
            end
            state <= S_READ_REQ;
          end else wait_cyc <= wait_cyc + 1;
        end

        S_READ_REQ: if (req_ready) begin
          wait_cyc <= 0;
          state <= S_READ_WAIT;
        end

        S_READ_WAIT: begin
          if (resp_valid) begin
            if (resp_rw !== 1'b0) begin
              $display("FAIL: read resp_rw=%b", resp_rw);
              errors <= errors + 1;
            end
            if (resp_rdata !== golden) begin
              $display("FAIL: read data mismatch");
              for (int k = 0; k < BURST_BYTES; k++) begin
                if (resp_rdata[k*8 +: 8] !== golden[k*8 +: 8])
                  $display("  byte[%0d]: got %02x exp %02x", k,
                           resp_rdata[k*8 +: 8], golden[k*8 +: 8]);
              end
              errors <= errors + 1;
            end
            if (wait_cyc < BEATS + RD_LATENCY) begin
              $display("FAIL: read latency wait_cyc=%0d < %0d", wait_cyc, BEATS + RD_LATENCY);
              errors <= errors + 1;
            end
            state <= S_FINISH;
          end else wait_cyc <= wait_cyc + 1;
        end
        S_FINISH: begin
          if (errors == 0)
            $display("PASS: ddr_axi4_master 64B burst write+read (RD_LAT=%0d, WR_LAT=%0d)",
                     RD_LATENCY, WR_LATENCY);
          else
            $display("FAIL: %0d error(s)", errors);
          $finish;
        end

      endcase
    end
  end

endmodule
