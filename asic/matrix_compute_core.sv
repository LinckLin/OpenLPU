// matrix_compute_core.sv - functional model of the physical matrix array core.
//
// The ASIC full-top flow replaces this body with matrix_compute_core_bb.sv.
// SRAM/control/interface logic remains synthesizable, while array area and
// timing are represented by the separately synthesized BF16/INT8 primitives.
`ifndef MATRIX_COMPUTE_CORE_SV
`define MATRIX_COMPUTE_CORE_SV

`include "qcore_pkg.sv"
`include "softfloat.sv"

module matrix_compute_core (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [1:0]  in_bank,
  input  logic [11:0] in_word_addr,
  input  logic        in_final,
  input  logic [63:0] state_word,
  input  logic [31:0] cin_word,
  input  logic [31:0] scale_word,
  input  logic [31:0] a,
  input  logic [31:0] b,
  input  logic [15:0] kk,
  input  logic [2:0]  srcA,
  input  logic [2:0]  srcB,
  input  logic        acc_init,
  input  logic        dequant,
  output logic        out_valid,
  output logic [1:0]  out_bank,
  output logic [11:0] out_word_addr,
  output logic        out_final,
  output logic [63:0] next_state_word
);
  import qcore_pkg::*;
  import softfloat_pkg::*;

  logic [31:0] calc_acc, calc_partial;
  logic [31:0] calc_p_old, calc_np, calc_w_f, calc_base;
  logic signed [31:0] calc_prod8;
  logic [63:0] next_state_comb;
  localparam int TEST_LATENCY = 4;
  logic [TEST_LATENCY-1:0] valid_pipe;
  logic [1:0]  bank_pipe  [0:TEST_LATENCY-1];
  logic [11:0] addr_pipe  [0:TEST_LATENCY-1];
  logic        final_pipe [0:TEST_LATENCY-1];
  logic [63:0] state_pipe [0:TEST_LATENCY-1];

  always_comb begin
    calc_acc = state_word[31:0];
    calc_partial = state_word[63:32];
    calc_p_old = 32'b0;
    calc_np = 32'b0;
    calc_w_f = 32'b0;
    calc_base = 32'b0;
    calc_prod8 = 32'sd0;

    if (dequant) begin
      if (srcB == DT_INT4) begin
        calc_w_f = i32_to_f32({{28{b[3]}}, b[3:0]});
        calc_p_old = (kk[6:0] == 7'b0) ? 32'b0 : state_word[63:32];
        calc_np = fp32_add(calc_p_old, fp32_mul(a, calc_w_f));
        calc_partial = calc_np;
        if (kk[6:0] == 7'h7f) begin
          calc_base = (kk == 16'd127) ?
                      (acc_init ? 32'b0 : cin_word) : state_word[31:0];
          calc_acc = fp32_add(calc_base, fp32_mul(calc_np, scale_word));
          calc_partial = 32'b0;
        end
      end else begin
        calc_prod8 = $signed(a[7:0]) * $signed(b[7:0]);
        calc_p_old = (kk[6:0] == 7'b0) ? 32'b0 : state_word[63:32];
        calc_np = calc_p_old + calc_prod8;
        calc_partial = calc_np;
        if (kk[6:0] == 7'h7f) begin
          calc_base = (kk == 16'd127) ?
                      (acc_init ? 32'b0 : cin_word) : state_word[31:0];
          calc_acc = fp32_add(calc_base,
                              fp32_mul(i32_to_f32(calc_np), scale_word));
          calc_partial = 32'b0;
        end
      end
    end else if ((srcA == DT_INT8) && (srcB == DT_INT8)) begin
      calc_prod8 = $signed(a[7:0]) * $signed(b[7:0]);
      calc_acc = (kk == 16'b0) ?
                 ((acc_init ? 32'b0 : cin_word) + calc_prod8) :
                 (state_word[31:0] + calc_prod8);
      calc_partial = 32'b0;
    end else begin
      calc_acc = (kk == 16'b0) ?
                 fp32_add(acc_init ? 32'b0 : cin_word, fp32_mul(a, b)) :
                 fp32_add(state_word[31:0], fp32_mul(a, b));
      calc_partial = 32'b0;
    end
    next_state_comb = {calc_partial, calc_acc};
  end

  // Four cycles deliberately exercise the latency mod 4 == 0 collision case:
  // a result returns to the same bank selected by the new read.  The physical
  // array may use any fixed pipeline depth; tags and per-bank queues decouple
  // that latency from the SRAM schedule.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      valid_pipe <= '0;
      for (integer stage = 0; stage < TEST_LATENCY; stage = stage + 1) begin
        bank_pipe[stage] <= 2'b0;
        addr_pipe[stage] <= 12'b0;
        final_pipe[stage] <= 1'b0;
        state_pipe[stage] <= 64'b0;
      end
    end else begin
      valid_pipe[0] <= in_valid;
      if (in_valid) begin
        bank_pipe[0] <= in_bank;
        addr_pipe[0] <= in_word_addr;
        final_pipe[0] <= in_final;
        state_pipe[0] <= next_state_comb;
      end
      for (integer stage = 1; stage < TEST_LATENCY; stage = stage + 1) begin
        valid_pipe[stage] <= valid_pipe[stage-1];
        if (valid_pipe[stage-1]) begin
          bank_pipe[stage] <= bank_pipe[stage-1];
          addr_pipe[stage] <= addr_pipe[stage-1];
          final_pipe[stage] <= final_pipe[stage-1];
          state_pipe[stage] <= state_pipe[stage-1];
        end
      end
    end
  end

  assign out_valid = valid_pipe[TEST_LATENCY-1];
  assign out_bank = bank_pipe[TEST_LATENCY-1];
  assign out_word_addr = addr_pipe[TEST_LATENCY-1];
  assign out_final = final_pipe[TEST_LATENCY-1];
  assign next_state_word = state_pipe[TEST_LATENCY-1];
endmodule

`endif // MATRIX_COMPUTE_CORE_SV
