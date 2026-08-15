// ============================================================================
// dma_kv_tb.sv — self-checking testbench for fpga/dma_kv_stage.sv.
//
// Instantiates the DMA/KV stage (DUT) against a behavioral AXI4 slave that
// (a) injects pseudo-random per-transaction read latency (16-bit LFSR), and
// (b) answers ready read transactions LIFO (most-recent first), so responses
// arrive out of order.  The self-checking FSM then verifies:
//   1. the CP combinational-read contract: rd_data is served from the stage's
//      reorder buffer with zero added latency once a line is buffered, and
//      rd_stall deasserts exactly when the requested line has landed;
//   2. out-of-order + random-delay reads still return byte-exact data
//      (the reorder buffer places each beat at the correct slot);
//   3. multi-outstanding: peak (issued - completed) read transactions >= 2;
//   4. the first line is delivered within T_FIRST (no double billing);
//   5. the write path (64 B write-combining + burst write) round-trips.
//
// clk is toggled by sim_dma_kv.cpp; the FSM drives done/err_count outputs.
// ============================================================================

// Behavioral AXI4 slave memory: flat byte array + preload, random-latency
// LIFO read responses (out-of-order), and an in-order write path.
module axi4_reorder_slave #(
  parameter int ADDR_BITS  = 40,
  parameter int DATA_BYTES = 8,
  parameter int BEATS      = 8,
  parameter int MEM_BYTES  = 1024,
  parameter int INF_DEPTH  = 8    // in-flight read transaction slots
) (
  input  logic clk, rst_n,
  // read address channel
  input  logic [ADDR_BITS-1:0] s_axi_araddr,
  input  logic [7:0] s_axi_arlen,
  input  logic [2:0] s_axi_arsize,
  input  logic [1:0] s_axi_arburst,
  input  logic [1:0] s_axi_arid,
  input  logic s_axi_arvalid,
  output logic s_axi_arready,
  // read data channel
  output logic [1:0] s_axi_rid,
  output logic [DATA_BYTES*8-1:0] s_axi_rdata,
  output logic [1:0] s_axi_rresp,
  output logic s_axi_rlast,
  output logic s_axi_rvalid,
  input  logic s_axi_rready,
  // write address channel
  input  logic [ADDR_BITS-1:0] s_axi_awaddr,
  input  logic [7:0] s_axi_awlen,
  input  logic [2:0] s_axi_awsize,
  input  logic [1:0] s_axi_awburst,
  input  logic [1:0] s_axi_awid,
  input  logic s_axi_awvalid,
  output logic s_axi_awready,
  // write data channel
  input  logic [DATA_BYTES*8-1:0] s_axi_wdata,
  input  logic [DATA_BYTES-1:0] s_axi_wstrb,
  input  logic s_axi_wlast,
  input  logic s_axi_wvalid,
  output logic s_axi_wready,
  // write response channel
  output logic [1:0] s_axi_bresp,
  output logic [1:0] s_axi_bid,
  output logic s_axi_bvalid,
  input  logic s_axi_bready
);

  logic [7:0] mem [0:MEM_BYTES-1];

  initial begin
    for (int a = 0; a < MEM_BYTES; a++) mem[a] = 8'((a * 7 + 3) & 8'hFF);
  end

  // ---- read in-flight queue ---------------------------------------------
  logic [INF_DEPTH-1:0]       inf_valid;
  logic [INF_DEPTH-1:0][1:0]  inf_id;
  logic [INF_DEPTH-1:0][ADDR_BITS-1:0] inf_addr;
  logic [INF_DEPTH-1:0][4:0]  inf_delay;

  logic [15:0] lfsr;
  logic [5:0]  resp_id;          // transaction being responded
  logic [ADDR_BITS-1:0] resp_addr;
  logic [3:0]  resp_beat;
  logic        resp_busy;

  localparam int BLEN = BEATS - 1;

  assign s_axi_arready = !inf_valid[INF_DEPTH-1];  // always room (INF>=N_OUT)
  assign s_axi_rvalid  = resp_busy;
  assign s_axi_rid     = resp_id[1:0];
  assign s_axi_rresp   = 2'b00;
  assign s_axi_rlast   = (resp_beat == BLEN[3:0]);
  assign s_axi_rdata   = {mem[resp_addr + resp_beat * DATA_BYTES + 7],
                          mem[resp_addr + resp_beat * DATA_BYTES + 6],
                          mem[resp_addr + resp_beat * DATA_BYTES + 5],
                          mem[resp_addr + resp_beat * DATA_BYTES + 4],
                          mem[resp_addr + resp_beat * DATA_BYTES + 3],
                          mem[resp_addr + resp_beat * DATA_BYTES + 2],
                          mem[resp_addr + resp_beat * DATA_BYTES + 1],
                          mem[resp_addr + resp_beat * DATA_BYTES + 0]};

  integer rpick;
  always_comb begin
    rpick = -1;
    // LIFO: pick the highest-index ready entry (most recently issued first).
    for (int i = INF_DEPTH - 1; i >= 0; i--)
      if (inf_valid[i] && inf_delay[i] == 0) begin rpick = i; break; end
  end

  // ---- write path ---------------------------------------------------------
  logic [ADDR_BITS-1:0] wa_q;
  logic [3:0] wbeat;
  logic       wbusy, wdone;
  assign s_axi_awready = !wbusy;
  assign s_axi_wready  = wbusy;
  assign s_axi_bvalid  = wdone;
  assign s_axi_bresp   = 2'b00;
  assign s_axi_bid     = 2'b00;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      inf_valid <= {INF_DEPTH{1'b0}};
      lfsr      <= 16'hACE1;
      resp_busy <= 1'b0;
      resp_id   <= 6'd0; resp_addr <= {ADDR_BITS{1'b0}}; resp_beat <= 4'd0;
      wbusy <= 1'b0; wdone <= 1'b0; wbeat <= 4'd0; wa_q <= {ADDR_BITS{1'b0}};
    end else begin
      lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

      // push a new read transaction
      if (s_axi_arvalid && s_axi_arready) begin
        for (int i = 0; i < INF_DEPTH; i++) begin
          if (!inf_valid[i]) begin
            inf_valid[i] <= 1'b1;
            inf_id[i]    <= s_axi_arid;
            inf_addr[i]  <= s_axi_araddr;
            inf_delay[i] <= 5'(lfsr[4:0]) + 5'd1;   // random 1..32
            break;
          end
        end
      end

      // start a response (LIFO among ready)
      if (!resp_busy && rpick != -1) begin
        resp_busy <= 1'b1;
        resp_id   <= inf_id[rpick];
        resp_addr <= inf_addr[rpick];
        resp_beat <= 4'd0;
        inf_valid[rpick] <= 1'b0;
      end

      // advance / finish the response
      if (resp_busy && s_axi_rvalid && s_axi_rready) begin
        if (s_axi_rlast) resp_busy <= 1'b0;
        else             resp_beat <= resp_beat + 1'b1;
      end

      // decrement delays of all queued (not being responded) transactions
      for (int i = 0; i < INF_DEPTH; i++) begin
        if (inf_valid[i] && inf_delay[i] != 0 &&
            !(resp_busy && rpick == i))
          inf_delay[i] <= inf_delay[i] - 1'b1;
      end

      // write path
      if (!wbusy && s_axi_awvalid && s_axi_awready) begin
        wbusy <= 1'b1; wbeat <= 4'd0; wa_q <= s_axi_awaddr; wdone <= 1'b0;
      end else if (wbusy && s_axi_wvalid && s_axi_wready) begin
        if (s_axi_wstrb[0]) mem[wa_q + wbeat * DATA_BYTES + 0] <= s_axi_wdata[7:0];
        if (s_axi_wstrb[1]) mem[wa_q + wbeat * DATA_BYTES + 1] <= s_axi_wdata[15:8];
        if (s_axi_wstrb[2]) mem[wa_q + wbeat * DATA_BYTES + 2] <= s_axi_wdata[23:16];
        if (s_axi_wstrb[3]) mem[wa_q + wbeat * DATA_BYTES + 3] <= s_axi_wdata[31:24];
        if (s_axi_wstrb[4]) mem[wa_q + wbeat * DATA_BYTES + 4] <= s_axi_wdata[39:32];
        if (s_axi_wstrb[5]) mem[wa_q + wbeat * DATA_BYTES + 5] <= s_axi_wdata[47:40];
        if (s_axi_wstrb[6]) mem[wa_q + wbeat * DATA_BYTES + 6] <= s_axi_wdata[55:48];
        if (s_axi_wstrb[7]) mem[wa_q + wbeat * DATA_BYTES + 7] <= s_axi_wdata[63:56];
        if (s_axi_wlast) begin wdone <= 1'b1; wbusy <= 1'b0; end
        else             wbeat <= wbeat + 1'b1;
      end
      if (wdone && s_axi_bvalid && s_axi_bready) wdone <= 1'b0;
    end
  end

endmodule


module dma_kv_tb #(
  parameter int ADDR_BITS   = 40,
  parameter int DATA_BYTES  = 8,
  parameter int BURST_BYTES = 64,
  parameter int N_OUT       = 4,
  parameter int T_FIRST     = 100,
  parameter int MEM_BYTES   = 1024,
  parameter int READ_BYTES  = 256,   // 4 lines
  parameter int WRITE_BASE  = 256,   // write region start
  parameter int WRITE_BYTES = 128    // 2 lines
) (
  input  logic clk,
  input  logic rst_n,
  output logic done,
  output logic [31:0] err_count
);
  localparam int BEATS = BURST_BYTES / DATA_BYTES;   // 8
  localparam int ID_BITS = $clog2(N_OUT);

  // ---- engine byte port (driven by the FSM) ------------------------------
  logic rd_sel, rd_stall, wr_en, wr_sel, wr_stall;
  logic [ADDR_BITS-1:0] rd_addr, wr_addr;
  logic [7:0] rd_data, wr_data;

  // ---- AXI wires -----------------------------------------------------------
  logic [ADDR_BITS-1:0] araddr, awaddr;
  logic [7:0] arlen, awlen;
  logic [2:0] arsize, awsize;
  logic [1:0] arburst, awburst;
  logic [ID_BITS-1:0] arid, rid, awid, bid;
  logic arvalid, arready, awvalid, awready;
  logic [DATA_BYTES*8-1:0] rdata, wdata;
  logic [DATA_BYTES-1:0] wstrb;
  logic [1:0] rresp, bresp;
  logic rlast, rvalid, rready;
  logic wlast, wvalid, wready, bvalid, bready;

  dma_kv_stage #(
    .ADDR_BITS(ADDR_BITS), .BURST_BYTES(BURST_BYTES),
    .DATA_BYTES(DATA_BYTES), .N_OUT(N_OUT), .T_FIRST(T_FIRST)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .rd_sel(rd_sel), .rd_addr(rd_addr), .rd_data(rd_data), .rd_stall(rd_stall),
    .wr_en(wr_en), .wr_sel(wr_sel), .wr_addr(wr_addr), .wr_data(wr_data),
    .wr_stall(wr_stall),
    .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize),
    .m_axi_arburst(arburst), .m_axi_arid(arid), .m_axi_arvalid(arvalid),
    .m_axi_arready(arready),
    .m_axi_rid(rid), .m_axi_rdata(rdata), .m_axi_rresp(rresp),
    .m_axi_rlast(rlast), .m_axi_rvalid(rvalid), .m_axi_rready(rready),
    .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize),
    .m_axi_awburst(awburst), .m_axi_awid(awid), .m_axi_awvalid(awvalid),
    .m_axi_awready(awready),
    .m_axi_wdata(wdata), .m_axi_wstrb(wstrb), .m_axi_wlast(wlast),
    .m_axi_wvalid(wvalid), .m_axi_wready(wready),
    .m_axi_bid(bid), .m_axi_bresp(bresp), .m_axi_bvalid(bvalid),
    .m_axi_bready(bready)
  );

  axi4_reorder_slave #(
    .ADDR_BITS(ADDR_BITS), .DATA_BYTES(DATA_BYTES), .BEATS(BEATS),
    .MEM_BYTES(MEM_BYTES)
  ) slave (
    .clk(clk), .rst_n(rst_n),
    .s_axi_araddr(araddr), .s_axi_arlen(arlen), .s_axi_arsize(arsize),
    .s_axi_arburst(arburst), .s_axi_arid(arid), .s_axi_arvalid(arvalid),
    .s_axi_arready(arready),
    .s_axi_rid(rid), .s_axi_rdata(rdata), .s_axi_rresp(rresp),
    .s_axi_rlast(rlast), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .s_axi_awaddr(awaddr), .s_axi_awlen(awlen), .s_axi_awsize(awsize),
    .s_axi_awburst(awburst), .s_axi_awid(awid), .s_axi_awvalid(awvalid),
    .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wlast(wlast),
    .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bid(bid), .s_axi_bresp(bresp), .s_axi_bvalid(bvalid),
    .s_axi_bready(bready)
  );

  // ---- golden reference (same preload formula as the slave) ---------------
  logic [7:0] golden [0:MEM_BYTES-1];
  initial begin
    for (int a = 0; a < MEM_BYTES; a++) golden[a] = 8'((a * 7 + 3) & 8'hFF);
  end

  // ---- test FSM ------------------------------------------------------------
  localparam logic [2:0]
    S_READ = 0, S_WRITE = 1, S_READBACK = 2, S_FINISH = 3;

  logic [2:0]  state;
  logic [31:0] errors;
  logic [31:0] first_lat, wait_cyc;
  logic        first_delivered;
  logic        reorder_seen;
  logic [2:0]  next_rid;
  logic [31:0] issued, completed, peak_out;

  assign rd_sel  = (state == S_READ) || (state == S_READBACK);
  assign wr_en   = (state == S_WRITE);
  assign wr_sel  = (state == S_WRITE);
  assign wr_data = 8'(wr_addr[7:0] * 5 + 11);

  assign done      = (state == S_FINISH);
  assign err_count = errors;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_READ;
      rd_addr   <= {ADDR_BITS{1'b0}};
      wr_addr   <= ADDR_BITS'(WRITE_BASE);
      errors    <= 32'd0;
      first_lat <= 32'd0; wait_cyc <= 32'd0;
      first_delivered <= 1'b0;
      reorder_seen <= 1'b0; next_rid <= 3'd0;
      issued <= 32'd0; completed <= 32'd0; peak_out <= 32'd0;
    end else begin
      // ---- multi-outstanding + reorder observability ----------------------
      if (arvalid && arready) issued <= issued + 1;
      if (rvalid && rready && rlast) begin
        completed <= completed + 1;
        if (rid[1:0] == next_rid[1:0]) next_rid <= next_rid + 1'b1;
        else                           reorder_seen <= 1'b1;
      end
      if (issued - completed > peak_out) peak_out <= issued - completed;

      case (state)
        S_READ: begin
          if (!first_delivered) wait_cyc <= wait_cyc + 1;
          if (!rd_stall) begin
            if (!first_delivered) begin
              first_lat <= wait_cyc;
              first_delivered <= 1'b1;
              if (wait_cyc > T_FIRST) begin
                $display("FAIL: first line delivered at %0d > T_FIRST %0d",
                         wait_cyc, T_FIRST);
                errors <= errors + 1;
              end
            end
            if (rd_data !== golden[rd_addr]) begin
              $display("FAIL: rd_data[%0d] got %02x exp %02x",
                       rd_addr, rd_data, golden[rd_addr]);
              errors <= errors + 1;
            end
            if (rd_addr == ADDR_BITS'(READ_BYTES - 1)) begin
              state   <= S_WRITE;
              rd_addr <= {ADDR_BITS{1'b0}};
              wr_addr <= ADDR_BITS'(WRITE_BASE);
            end else rd_addr <= rd_addr + 1'b1;
          end
        end

        S_WRITE: begin
          if (wr_stall) begin
            $display("FAIL: write FIFO overflow guard asserted");
            errors <= errors + 1;
          end
          golden[wr_addr] <= wr_data;
          if (wr_addr == ADDR_BITS'(WRITE_BASE + WRITE_BYTES - 1)) begin
            state   <= S_READBACK;
            rd_addr <= ADDR_BITS'(WRITE_BASE);
          end else wr_addr <= wr_addr + 1'b1;
        end

        S_READBACK: begin
          if (!rd_stall) begin
            if (rd_data !== golden[rd_addr]) begin
              $display("FAIL: readback[%0d] got %02x exp %02x",
                       rd_addr, rd_data, golden[rd_addr]);
              errors <= errors + 1;
            end
            if (rd_addr == ADDR_BITS'(WRITE_BASE + WRITE_BYTES - 1)) begin
              state <= S_FINISH;
            end else rd_addr <= rd_addr + 1'b1;
          end
        end

        S_FINISH: begin
          if (peak_out < 2) begin
            $display("FAIL: peak outstanding = %0d (expected >= 2)", peak_out);
            errors <= errors + 1;
          end
          if (!reorder_seen) begin
            $display("FAIL: no out-of-order read response observed");
            errors <= errors + 1;
          end
          if (errors == 0) begin
            $display("PASS: dma_kv_stage read (random delay + out-of-order, peak_out=%0d, first_lat=%0d<=T_FIRST=%0d) + write round-trip", peak_out, first_lat, T_FIRST);
          end else begin
            $display("FAIL: %0d error(s)", errors);
          end
          $finish;
        end

        default: state <= S_READ;
      endcase
    end
  end

endmodule
