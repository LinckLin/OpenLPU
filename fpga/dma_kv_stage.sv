// ============================================================================
// dma_kv_stage.sv — DDR-side DMA/KV data-path stage (P9b, board-independent).
//
// Pre-arbitration proposal for porting.md §5.6 option B ("ddr_if side absorbs
// the latency", CP unchanged).  This module is the board-mode data path that
// the co-sim keeps idle: in co-sim the functional zero-latency arrays serve
// engine traffic and the AXI master is tied off, so the M6 criterion (trace
// identical, bf16 <= 1 ULP) is preserved exactly.  On the board, ddr_if wires
// this stage between its engine byte port and the physical DDR controller.
//
// Hook point: ddr_if engine byte port, DDR side only (rd_sel=1 / wr_sel=1).
// SRAM (rd_sel=0) stays on ddr_if's own flat array — untouched, still zero
// latency.  The CP's combinational read contract is preserved 1:1: rd_data is
// still a pure combinational function of rd_sel/rd_addr, served from an
// on-chip reorder buffer instead of a functional DDR array.
//
// Latency accounting (the critical point):
//   * The frozen model already bills T_FIRST (= qcore_pkg.T_FIRST = 100) once
//     per DDR transfer (DMA.STORE / KV.*  = T_FIRST + hbm/sram_write_cycles).
//   * This stage must therefore deliver the first byte of a transfer *within*
//     that already-billed T_FIRST window — NOT add a second 100 on top
//     (never 200).  The physical DDR read latency is absorbed by rd_stall:
//     the stage prefetches 64 B lines ahead of the CP's byte stream, buffers
//     them in a reorder buffer, and only asserts rd_stall while the requested
//     line has not landed.  Once buffered, rd_data is zero-added-latency.
//   * rd_stall/wr_stall are the *only* integration points: at P9 bring-up the
//     DMA engine's byte-stream advance (dma_engine bpos) is gated by them.
//     This does not change the CP's combinational-read semantics nor the
//     charged latency model; it merely realizes the T_FIRST the CP already
//     bills.  (A 1-line rtl/ gate on dma_engine bpos is the P9-bring-up
//     consequence, reviewed then — rtl/ is untouched in this task.)
//
// Multi-outstanding — implementation choice (reported in docs/p9/porting.md):
//   the stage carries its OWN AXI read engine (option "stage 自带 AXI 引擎"),
//   NOT an extension of ddr_axi4_master.  N_OUT outstanding read transactions
//   (default 4 = the frozen DMA_IN_FLIGHT hardware pool), each an independent
//   AXI4 INCR burst, tagged by arid/rid, landing in a per-slot reorder buffer
//   so out-of-order responses are correctly placed.  ddr_axi4_master (the
//   single-transaction, fixed-latency model) is left untouched.
//
// Write path: 64 B write-combining line buffer -> 4-deep line FIFO -> AXI4
// INCR write bursts (byte-granular wstrb so partial end-of-stream lines are
// flushed without corrupting neighbours).  Fill rate (1 B/cyc from the DMA
// engine) is ~6x below the drain rate (1 line / ~10 cyc), so the FIFO never
// stalls for the frozen stream; wr_stall exists only as a completeness guard.
// ============================================================================
`ifndef DMA_KV_STAGE_SV
`define DMA_KV_STAGE_SV

module dma_kv_stage #(
  parameter int ADDR_BITS   = 40,     // byte address width
  parameter int BURST_BYTES = 64,     // one line = one AXI INCR burst
  parameter int DATA_BYTES  = 8,      // narrow AXI data width (bytes/beat)
  parameter int N_OUT       = 4,      // outstanding read transactions (slots)
  parameter int WF_DEPTH    = 4,      // write-combining line FIFO depth
  parameter int T_FIRST     = 100     // absorbed latency budget = DDR_RD_LATENCY
) (
  input  logic clk,
  input  logic rst_n,
  // -- engine byte port, DDR side (CP combinational-read contract kept) ------
  input  logic                 rd_sel,      // 1 = DDR (this stage), 0 = SRAM
  input  logic [ADDR_BITS-1:0] rd_addr,
  output logic [7:0]           rd_data,     // combinational from reorder buffer
  output logic                 rd_stall,    // 1 = requested line not buffered
  input  logic                 wr_en,       // posedge write
  input  logic                 wr_sel,      // 1 = DDR write
  input  logic [ADDR_BITS-1:0] wr_addr,
  input  logic [7:0]           wr_data,
  output logic                 wr_stall,    // 1 = write FIFO full (guard)
  // -- AXI4 read port (multi-outstanding; arid/rid = slot tag) ---------------
  output logic [ADDR_BITS-1:0]       m_axi_araddr,
  output logic [7:0]                 m_axi_arlen,
  output logic [2:0]                 m_axi_arsize,
  output logic [1:0]                 m_axi_arburst,
  output logic [$clog2(N_OUT)-1:0]   m_axi_arid,
  output logic                       m_axi_arvalid,
  input  logic                       m_axi_arready,
  input  logic [$clog2(N_OUT)-1:0]   m_axi_rid,
  input  logic [DATA_BYTES*8-1:0]    m_axi_rdata,
  input  logic [1:0]                 m_axi_rresp,
  input  logic                       m_axi_rlast,
  input  logic                       m_axi_rvalid,
  output logic                       m_axi_rready,
  // -- AXI4 write port --------------------------------------------------------
  output logic [ADDR_BITS-1:0]       m_axi_awaddr,
  output logic [7:0]                 m_axi_awlen,
  output logic [2:0]                 m_axi_awsize,
  output logic [1:0]                 m_axi_awburst,
  output logic [1:0]                 m_axi_awid,
  output logic                       m_axi_awvalid,
  input  logic                       m_axi_awready,
  output logic [DATA_BYTES*8-1:0]    m_axi_wdata,
  output logic [DATA_BYTES-1:0]      m_axi_wstrb,
  output logic                       m_axi_wlast,
  output logic                       m_axi_wvalid,
  input  logic                       m_axi_wready,
  input  logic [1:0]                 m_axi_bid,
  input  logic [1:0]                 m_axi_bresp,
  input  logic                       m_axi_bvalid,
  output logic                       m_axi_bready
);

  localparam int LOG2_BURST = $clog2(BURST_BYTES);   // 6
  localparam int BEATS      = BURST_BYTES / DATA_BYTES;   // 8
  localparam int BLEN       = BEATS - 1;                  // 7
  localparam int BSIZE      = $clog2(DATA_BYTES);         // 3
  localparam int ID_BITS    = $clog2(N_OUT);
  localparam int LINE_BITS  = ADDR_BITS - LOG2_BURST;     // 34

  typedef logic [LINE_BITS-1:0] line_t;   // line index = addr >> LOG2_BURST

  // ==========================================================================
  // Read path — prefetch engine + per-slot reorder buffer
  // ==========================================================================

  // slot states: 0 IDLE, 1 AR (address phase), 2 DATA (beats), 3 VALID
  logic [1:0]  r_state [N_OUT];
  line_t       r_addr  [N_OUT];
  logic [BURST_BYTES*8-1:0] r_data [N_OUT];
  logic [3:0]  r_beat  [N_OUT];

  // AR channel owner (one outstanding address phase at a time)
  logic        ar_pending;
  line_t       ar_line;
  logic [ID_BITS-1:0] ar_slot;

  line_t       need_line;
  assign need_line = rd_addr[ADDR_BITS-1:LOG2_BURST];

  // combinational: rd_data from a matching VALID slot; rd_stall otherwise.
  always_comb begin
    rd_data  = 8'b0;
    rd_stall = 1'b0;
    if (rd_sel) begin
      rd_stall = 1'b1;
      for (int i = 0; i < N_OUT; i++) begin
        if (r_addr[i] == need_line && r_state[i] == 2'd3) begin
          rd_data  = r_data[i][rd_addr[LOG2_BURST-1:0]*8 +: 8];
          rd_stall = 1'b0;
        end
      end
    end
  end
  // combinational: is lookahead line need+k (k=1..N_OUT-1) present in a slot?
  logic [N_OUT-1:0] la_present;
  always_comb begin
    la_present = {N_OUT{1'b0}};
    for (int k = 1; k < N_OUT; k++) begin
      la_present[k] = 1'b0;
      for (int i = 0; i < N_OUT; i++)
        if (r_addr[i] == (need_line + line_t'(k)) && r_state[i] != 2'd0)
          la_present[k] = 1'b1;
    end
  end

  // combinational: is need_line in flight / buffered.
  logic need_pending, need_buffered;
  always_comb begin
    need_pending = 1'b0;
    need_buffered = 1'b0;
    for (int i = 0; i < N_OUT; i++) begin
      if (r_addr[i] == need_line) begin
        if (r_state[i] == 2'd3)       need_buffered = 1'b1;
        else if (r_state[i] != 2'd0)  need_pending  = 1'b1;
      end
    end
  end

  // combinational: free (IDLE) and evictable (VALID) slot search.
  integer free_idx, evict_idx;
  always_comb begin
    free_idx  = -1;
    evict_idx = -1;
    for (int i = 0; i < N_OUT; i++) begin
      if (free_idx  == -1 && r_state[i] == 2'd0) free_idx  = i;
      if (evict_idx == -1 && r_state[i] == 2'd3) evict_idx = i;
    end
  end

  // combinational: allocate one fetch this cycle.  Priority 1 = the required
  // line (rd_addr's line) if it is neither buffered nor in flight; priority 2
  // = sequential lookahead prefetch up to N_OUT-1 lines ahead (fills the
  // pipeline so T_FIRST latency is hidden across line boundaries — up to
  // N_OUT transactions in flight, the multi-outstanding guarantee).  At most
  // one AR per cycle (single AR port).
  logic        alloc_valid;
  line_t       alloc_line;
  logic [ID_BITS-1:0] alloc_slot;
  always_comb begin
    alloc_valid = 1'b0;
    alloc_line  = {LINE_BITS{1'b0}};
    alloc_slot  = {ID_BITS{1'b0}};
    if (rd_sel) begin
      if (!need_buffered && !need_pending) begin
        // required line must be fetched now (evict a buffered line if needed)
        if (free_idx  != -1)      begin alloc_valid = 1'b1; alloc_line = need_line; alloc_slot = ID_BITS'(free_idx);  end
        else if (evict_idx != -1) begin alloc_valid = 1'b1; alloc_line = need_line; alloc_slot = ID_BITS'(evict_idx); end
      end else begin
        // lookahead prefetch: first un-present line at need+1 .. need+N_OUT-1
        // into a truly free slot (never evict on the prefetch path).
        for (int k = 1; k < N_OUT; k++) begin
          if (free_idx != -1 && !la_present[k]) begin
            alloc_valid = 1'b1;
            alloc_line  = need_line + line_t'(k);
            alloc_slot  = ID_BITS'(free_idx);
            break;
          end
        end
      end
    end
  end

  // AXI read port combinational drives.
  assign m_axi_arvalid = ar_pending;
  assign m_axi_araddr  = {ar_line, {LOG2_BURST{1'b0}}};
  assign m_axi_arlen   = 8'(BLEN);
  assign m_axi_arsize  = 3'(BSIZE);
  assign m_axi_arburst = 2'b01;                 // INCR
  assign m_axi_arid    = ar_slot;
  assign m_axi_rready  = 1'b1;

  // Read engine sequencing: AR handshake + beat collection + allocation.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ar_pending <= 1'b0;
      ar_line    <= {LINE_BITS{1'b0}};
      ar_slot    <= {ID_BITS{1'b0}};
      for (int i = 0; i < N_OUT; i++) begin
        r_state[i] <= 2'd0;
        r_addr[i]  <= {LINE_BITS{1'b0}};
        r_data[i]  <= {BURST_BYTES*8{1'b0}};
        r_beat[i]  <= 4'd0;
      end
    end else begin
      // 1. AR handshake completes: owner slot AR -> DATA.
      if (ar_pending && m_axi_arready) begin
        ar_pending <= 1'b0;
        r_state[ar_slot] <= 2'd2;
      end

      // 2. issue a new AR when the channel is free (no back-to-back same-cyc).
      if (alloc_valid && !ar_pending) begin
        ar_pending <= 1'b1;
        ar_line    <= alloc_line;
        ar_slot    <= alloc_slot;
        r_addr[alloc_slot] <= alloc_line;
        r_beat[alloc_slot] <= 4'd0;
        r_state[alloc_slot] <= 2'd1;
      end

      // 3. collect read beats (tagged by rid -> reorder buffer).
      if (m_axi_rvalid && m_axi_rready) begin
        r_data[m_axi_rid][r_beat[m_axi_rid]*DATA_BYTES*8 +: DATA_BYTES*8] <= m_axi_rdata;
        if (m_axi_rlast) r_state[m_axi_rid] <= 2'd3;
        else             r_beat[m_axi_rid]  <= r_beat[m_axi_rid] + 1'b1;
      end
    end
  end

  // ==========================================================================
  // Write path — 64 B write-combining buffer -> line FIFO -> AXI write burst
  // ==========================================================================

  // accumulator (current line being assembled)
  logic        acc_active;
  line_t       acc_line;
  logic [BURST_BYTES*8-1:0] acc_data;
  logic [BURST_BYTES-1:0]   acc_strb;   // per-byte valid (partial-line flush)
  logic [6:0]  acc_cnt;

  // line FIFO
  line_t       wf_line [WF_DEPTH];
  logic [BURST_BYTES*8-1:0] wf_data [WF_DEPTH];
  logic [BURST_BYTES-1:0]   wf_strb [WF_DEPTH];
  logic [2:0]  wf_wptr, wf_rptr;
  logic [2:0]  wf_cnt;

  // AXI write FSM
  localparam logic [1:0] WS_IDLE = 2'd0, WS_AW = 2'd1, WS_WD = 2'd2, WS_B = 2'd3;
  logic [1:0]  w_state;
  logic [3:0]  w_beat;
  line_t       aw_line;
  logic [BURST_BYTES*8-1:0] aw_data;
  logic [BURST_BYTES-1:0]   aw_strb;

  // combinational: raw DDR write request this cycle (not stall-gated).
  logic byte_is_ddr;
  assign byte_is_ddr = wr_en && wr_sel;

  // combinational: does this cycle complete the current 64 B line?
  logic line_done;
  assign line_done = byte_is_ddr && acc_active &&
                     (acc_cnt == BURST_BYTES - 1) &&
                     (wr_addr[ADDR_BITS-1:LOG2_BURST] == acc_line);

  // combinational: does this cycle jump to a new line (partial-line flush)?
  logic line_jump;
  assign line_jump = byte_is_ddr && acc_active &&
                     (wr_addr[ADDR_BITS-1:LOG2_BURST] != acc_line);

  logic push_any;
  assign push_any = line_done || line_jump;

  // combinational: the accumulator line with this cycle's byte merged in
  // (used by the line_done push so the 64th byte is included in the burst).
  logic [BURST_BYTES*8-1:0] merged_data;
  logic [BURST_BYTES-1:0]   merged_strb;
  always_comb begin
    merged_data = acc_data;
    merged_strb = acc_strb;
    if (byte_is_ddr) begin
      merged_data[wr_addr[LOG2_BURST-1:0]*8 +: 8] = wr_data;
      merged_strb[wr_addr[LOG2_BURST-1:0]] = 1'b1;
    end
  end

  // completeness guard only — the frozen 1 B/cyc stream never trips it.
  assign wr_stall = push_any && (wf_cnt == WF_DEPTH[2:0]);

  // AXI write port combinational drives.
  assign m_axi_awvalid = (w_state == WS_AW);
  assign m_axi_awaddr  = {aw_line, {LOG2_BURST{1'b0}}};
  assign m_axi_awlen   = 8'(BLEN);
  assign m_axi_awsize  = 3'(BSIZE);
  assign m_axi_awburst = 2'b01;                 // INCR
  assign m_axi_awid    = 2'b00;
  assign m_axi_wvalid  = (w_state == WS_WD);
  assign m_axi_wdata   = aw_data[w_beat*DATA_BYTES*8 +: DATA_BYTES*8];
  assign m_axi_wstrb   = aw_strb[w_beat*DATA_BYTES +: DATA_BYTES];
  assign m_axi_wlast   = (w_beat == BLEN[3:0]);
  assign m_axi_bready  = (w_state == WS_B);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      acc_active <= 1'b0;
      acc_line   <= {LINE_BITS{1'b0}};
      acc_data   <= {BURST_BYTES*8{1'b0}};
      acc_strb   <= {BURST_BYTES{1'b0}};
      acc_cnt    <= 7'd0;
      wf_wptr    <= 3'd0;
      wf_rptr    <= 3'd0;
      wf_cnt     <= 3'd0;
      w_state    <= WS_IDLE;
      w_beat     <= 4'd0;
      aw_line    <= {LINE_BITS{1'b0}};
      aw_data    <= {BURST_BYTES*8{1'b0}};
      aw_strb    <= {BURST_BYTES{1'b0}};
    end else begin
      // ---- accumulator: byte-granular write combining ---------------------
      if (byte_is_ddr && !wr_stall) begin
        if (!acc_active) begin
          acc_active <= 1'b1;
          acc_line   <= wr_addr[ADDR_BITS-1:LOG2_BURST];
          acc_cnt    <= 7'd1;
          acc_strb   <= {BURST_BYTES{1'b0}};
        end else if (line_jump) begin
          // flush (below) then start the new line with this byte
          acc_line   <= wr_addr[ADDR_BITS-1:LOG2_BURST];
          acc_cnt    <= 7'd1;
          acc_strb   <= {BURST_BYTES{1'b0}};
        end else begin
          // same line: continue, or complete on the 64th byte
          acc_cnt <= line_done ? 7'd0 : (acc_cnt + 1'b1);
          if (line_done) acc_active <= 1'b0;
        end
        acc_data[wr_addr[LOG2_BURST-1:0]*8 +: 8] <= wr_data;
        acc_strb[wr_addr[LOG2_BURST-1:0]] <= 1'b1;
      end

      // ---- push completed / jumped line into FIFO --------------------------
      if (push_any && !wr_stall) begin
        wf_line[wf_wptr[1:0]] <= acc_line;
        if (line_done) begin
          wf_data[wf_wptr[1:0]] <= merged_data;
          wf_strb[wf_wptr[1:0]] <= merged_strb;
        end else begin
          wf_data[wf_wptr[1:0]] <= acc_data;
          wf_strb[wf_wptr[1:0]] <= acc_strb;
        end
        wf_wptr <= wf_wptr + 1'b1;
        wf_cnt  <= wf_cnt + 1'b1;
      end

      // ---- AXI write burst drain -------------------------------------------
      case (w_state)
        WS_IDLE: if (wf_cnt != 0) begin
          aw_line <= wf_line[wf_rptr[1:0]];
          aw_data <= wf_data[wf_rptr[1:0]];
          aw_strb <= wf_strb[wf_rptr[1:0]];
          wf_rptr <= wf_rptr + 1'b1;
          wf_cnt  <= wf_cnt - 1'b1;
          w_state <= WS_AW;
        end

        WS_AW: if (m_axi_awready) begin
          w_beat  <= 4'd0;
          w_state <= WS_WD;
        end

        WS_WD: if (m_axi_wvalid && m_axi_wready) begin
          if (m_axi_wlast) w_state <= WS_B;
          else             w_beat  <= w_beat + 1'b1;
        end

        WS_B: if (m_axi_bvalid && m_axi_bready) w_state <= WS_IDLE;

        default: w_state <= WS_IDLE;
      endcase
    end
  end

endmodule

`endif // DMA_KV_STAGE_SV
