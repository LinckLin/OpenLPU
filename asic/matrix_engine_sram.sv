// ============================================================================
// matrix_engine_sram.sv - SMIC28 macro-backed matrix-engine implementation.
//
// This is the physical ASIC implementation of rtl/matrix_engine.sv.  The
// co-sim model keeps four inferred arrays for fast random access; this version
// maps the same state to nine kh4096x64 single-port SRAM macros:
//
//   * 4 banks x 4096 x 64: {partial, accumulator}
//   * 4 banks x 4096 x 64: C seed (low 32 bits used)
//   * 1 bank  x 4096 x 64: dequant scale (low 32 bits used)
//
// Accumulator addresses are interleaved on index[1:0].  The arithmetic-array
// boundary returns valid+address tags; each bank has a one-entry writeback
// queue, so a result that collides with a new single-port read is committed on
// the next free bank cycle.  This sustains one issued MAC per cycle when the
// fixed arithmetic-core latency is shorter than the M*N element-reuse interval.
// Numeric order remains k -> m -> n, matching the co-sim model and qsim.
// ============================================================================
`ifndef MATRIX_ENGINE_SRAM_SV
`define MATRIX_ENGINE_SRAM_SV

`include "qcore_pkg.sv"
`include "sram_macros.sv"
`include "matrix_compute_core.sv"

module matrix_engine #(
  parameter int MAX_M = 128,
  parameter int MAX_N = 128,
  parameter int MAX_K = 4096
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [7:0]  M,
  input  logic [7:0]  N,
  input  logic [15:0] K,
  input  logic [2:0]  srcA,
  input  logic [2:0]  srcB,
  input  logic        acc_init,
  input  logic        dequant,
  input  logic        start,
  input  logic        step,
  input  logic [31:0] a_slice [MAX_M],
  input  logic [31:0] b_slice [MAX_N],
  input  logic [13:0] cin_waddr,
  input  logic [31:0] cin_wdata,
  input  logic        cin_we,
  input  logic [11:0] scale_waddr,
  input  logic [31:0] scale_wdata,
  input  logic        scale_we,
  input  logic [13:0] c_raddr,
  output logic [31:0] c_rdata,
  output logic        done
);
  import qcore_pkg::*;

  localparam logic [2:0] SRAM_EMA  = 3'b011;
  localparam logic [1:0] SRAM_EMAW = 2'b01;

  logic [15:0] kk;
  logic [7:0]  mm, nn;
  logic        mac_active;

  logic [8:0]  group_count;
  logic [13:0] issue_idx;
  logic [11:0] issue_word_addr;
  logic [1:0]  issue_bank;
  logic [11:0] issue_scale_addr;
  logic        issue_fire;

  assign group_count      = K[15:7];
  assign issue_idx        = ({6'b0, mm} * {6'b0, N}) + {6'b0, nn};
  assign issue_word_addr  = issue_idx[13:2];
  assign issue_bank       = issue_idx[1:0];
  assign issue_scale_addr = ({4'b0, nn} * group_count) + kk[15:7];
  assign issue_fire       = mac_active && step;

  // Read metadata.  Macro Q changes after the request edge; these registers
  // select the corresponding bank/data during the following compute cycle.
  logic        pipe_valid;
  logic [13:0] pipe_idx;
  logic [1:0]  pipe_bank;
  logic [15:0] pipe_kk;
  logic [31:0] pipe_a, pipe_b;
  logic [2:0]  pipe_srcA, pipe_srcB;
  logic        pipe_acc_init, pipe_dequant, pipe_final;

  // Per-bank result queues decouple arithmetic-core latency from the
  // single-port SRAM schedule.  A read has priority for its selected bank;
  // queued writes use any other bank in the same cycle.
  logic [3:0]  pending_valid, pending_final;
  logic [11:0] pending_word_addr [0:3];
  logic [63:0] pending_data      [0:3];
  logic [3:0]  ap_write_fire;
  logic [2:0]  write_count;
  logic [26:0] outstanding;

  logic [63:0] ap_q  [0:3];
  logic [63:0] cin_q [0:3];
  logic [63:0] scale_q;
  logic [3:0]  ap_cen, ap_wen, cin_cen, cin_wen;
  logic [11:0] ap_addr  [0:3];
  logic [11:0] cin_addr [0:3];
  logic [63:0] ap_d     [0:3];
  logic [63:0] cin_d    [0:3];

  logic c_read_fire;
  logic c_prefetch_zero;
  logic [1:0] c_bank_q;
  wire [13:0] c_read_addr = c_prefetch_zero ? 14'b0 : c_raddr;
  assign c_read_fire = !mac_active && (outstanding == 27'b0);
  assign write_count = {2'b0, ap_write_fire[0]} +
                       {2'b0, ap_write_fire[1]} +
                       {2'b0, ap_write_fire[2]} +
                       {2'b0, ap_write_fire[3]};

  genvar bank;
  generate
    for (bank = 0; bank < 4; bank = bank + 1) begin : g_state_bank
      localparam logic [1:0] BANK = bank;
      wire ap_do_issue = issue_fire && (issue_bank == BANK);
      wire ap_do_write = pending_valid[bank] && !ap_do_issue;
      wire ap_do_cread = c_read_fire && (c_read_addr[1:0] == BANK);
      wire cin_do_write = cin_we && (cin_waddr[1:0] == BANK);
      wire cin_do_read  = issue_fire && (issue_bank == BANK);

      always_comb begin
        ap_write_fire[bank] = ap_do_write;
        ap_cen[bank] = ~(ap_do_write || ap_do_issue || ap_do_cread);
        ap_wen[bank] = ~ap_do_write;
        ap_addr[bank] = ap_do_write ? pending_word_addr[bank] :
                        ap_do_issue ? issue_word_addr : c_read_addr[13:2];
        ap_d[bank] = pending_data[bank];

        cin_cen[bank] = ~(cin_do_write || cin_do_read);
        cin_wen[bank] = ~cin_do_write;
        cin_addr[bank] = cin_do_write ? cin_waddr[13:2] : issue_word_addr;
        cin_d[bank] = {32'b0, cin_wdata};
      end

      kh4096x64 u_acc_partial (
        .Q(ap_q[bank]), .CLK(clk), .CEN(ap_cen[bank]), .WEN(ap_wen[bank]),
        .A(ap_addr[bank]), .D(ap_d[bank]), .EMA(SRAM_EMA), .EMAW(SRAM_EMAW),
        .EMAS(1'b0), .RET1N(1'b1)
      );

      kh4096x64 u_cin (
        .Q(cin_q[bank]), .CLK(clk), .CEN(cin_cen[bank]), .WEN(cin_wen[bank]),
        .A(cin_addr[bank]), .D(cin_d[bank]), .EMA(SRAM_EMA), .EMAW(SRAM_EMAW),
        .EMAS(1'b0), .RET1N(1'b1)
      );
    end
  endgenerate

  wire scale_do_write = scale_we;
  wire scale_do_read  = issue_fire && !scale_we;
  wire scale_cen      = ~(scale_do_write || scale_do_read);
  wire scale_wen      = ~scale_do_write;
  wire [11:0] scale_addr = scale_do_write ? scale_waddr : issue_scale_addr;

  kh4096x64 u_scale (
    .Q(scale_q), .CLK(clk), .CEN(scale_cen), .WEN(scale_wen),
    .A(scale_addr), .D({32'b0, scale_wdata}), .EMA(SRAM_EMA),
    .EMAW(SRAM_EMAW), .EMAS(1'b0), .RET1N(1'b1)
  );

  logic [63:0] ap_read_word, cin_read_word;
  logic [63:0] compute_next_word;
  logic        compute_out_valid, compute_out_final;
  logic [1:0]  compute_out_bank;
  logic [11:0] compute_out_word_addr;

  always_comb begin
    ap_read_word = ap_q[0];
    cin_read_word = cin_q[0];
    case (pipe_bank)
      2'd1: begin ap_read_word = ap_q[1]; cin_read_word = cin_q[1]; end
      2'd2: begin ap_read_word = ap_q[2]; cin_read_word = cin_q[2]; end
      2'd3: begin ap_read_word = ap_q[3]; cin_read_word = cin_q[3]; end
      default: begin end
    endcase

  end

  matrix_compute_core u_compute (
    .clk(clk), .rst_n(rst_n), .in_valid(pipe_valid),
    .in_bank(pipe_bank), .in_word_addr(pipe_idx[13:2]),
    .in_final(pipe_final),
    .state_word(ap_read_word), .cin_word(cin_read_word[31:0]),
    .scale_word(scale_q[31:0]), .a(pipe_a), .b(pipe_b), .kk(pipe_kk),
    .srcA(pipe_srcA), .srcB(pipe_srcB), .acc_init(pipe_acc_init),
    .dequant(pipe_dequant), .out_valid(compute_out_valid),
    .out_bank(compute_out_bank), .out_word_addr(compute_out_word_addr),
    .out_final(compute_out_final), .next_state_word(compute_next_word)
  );

  always_comb begin
    case (c_bank_q)
      2'd1: c_rdata = ap_q[1][31:0];
      2'd2: c_rdata = ap_q[2][31:0];
      2'd3: c_rdata = ap_q[3][31:0];
      default: c_rdata = ap_q[0][31:0];
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      kk <= 16'b0;
      mm <= 8'b0;
      nn <= 8'b0;
      mac_active <= 1'b0;
      pipe_valid <= 1'b0;
      pending_valid <= 4'b0;
      pending_final <= 4'b0;
      outstanding <= 27'b0;
      done <= 1'b0;
      c_prefetch_zero <= 1'b0;
      c_bank_q <= 2'b0;
    end else if (start) begin
      kk <= 16'b0;
      mm <= 8'b0;
      nn <= 8'b0;
      mac_active <= 1'b1;
      pipe_valid <= 1'b0;
      pending_valid <= 4'b0;
      pending_final <= 4'b0;
      outstanding <= 27'b0;
      done <= 1'b0;
      c_prefetch_zero <= 1'b0;
    end else begin
      if (c_read_fire) begin
        c_bank_q <= c_read_addr[1:0];
        c_prefetch_zero <= 1'b0;
      end

      if (|(ap_write_fire & pending_final)) begin
        done <= 1'b1;
        // CP clears rd_ptr while leaving S_MX_WAIT.  Force the synchronous
        // macro's one available transition-cycle read to result element 0.
        c_prefetch_zero <= 1'b1;
      end

      outstanding <= outstanding + (issue_fire ? 27'd1 : 27'd0) -
                     {{24{1'b0}}, write_count};

      for (integer p = 0; p < 4; p = p + 1) begin
        if (ap_write_fire[p]) begin
          pending_valid[p] <= 1'b0;
          pending_final[p] <= 1'b0;
        end
      end
      if (compute_out_valid) begin
        pending_valid[compute_out_bank] <= 1'b1;
        pending_final[compute_out_bank] <= compute_out_final;
        pending_word_addr[compute_out_bank] <= compute_out_word_addr;
        pending_data[compute_out_bank] <= compute_next_word;
      end

      pipe_valid <= issue_fire;
      if (issue_fire) begin
        pipe_idx <= issue_idx;
        pipe_bank <= issue_bank;
        pipe_kk <= kk;
        pipe_a <= a_slice[mm];
        pipe_b <= b_slice[nn];
        pipe_srcA <= srcA;
        pipe_srcB <= srcB;
        pipe_acc_init <= acc_init;
        pipe_dequant <= dequant;
        pipe_final <= (kk == K - 16'd1) &&
                      (mm == M - 8'd1) && (nn == N - 8'd1);

        if (nn == N - 8'd1) begin
          nn <= 8'b0;
          if (mm == M - 8'd1) begin
            mm <= 8'b0;
            if (kk == K - 16'd1) begin
              mac_active <= 1'b0;
            end else begin
              kk <= kk + 16'd1;
            end
          end else begin
            mm <= mm + 8'd1;
          end
        end else begin
          nn <= nn + 8'd1;
        end
      end
    end
  end

  // The four-bank hazard schedule needs four-column alignment and enough
  // element-reuse distance for the fixed-latency core to commit a result.  The
  // behavioral stress core uses 4 cycles; the frozen modes use N=128.
  // synopsys translate_off
  always_ff @(posedge clk) begin
    if (rst_n && start && ((({8'b0, M} * {8'b0, N}) < 16'd8) ||
                           (N[1:0] != 2'b0))) begin
      $error("matrix_engine_sram requires M*N >= 8 and N %% 4 == 0 (M=%0d N=%0d)",
             M, N);
    end
    if (rst_n && mac_active && (cin_we || scale_we)) begin
      $error("matrix_engine_sram seed/scale writes must finish before start");
    end
    if (rst_n && compute_out_valid && pending_valid[compute_out_bank] &&
        !ap_write_fire[compute_out_bank]) begin
      $error("matrix_engine_sram writeback queue overflow on bank %0d",
             compute_out_bank);
    end
  end
  // synopsys translate_on

endmodule

`endif // MATRIX_ENGINE_SRAM_SV
