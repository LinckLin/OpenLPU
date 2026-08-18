// matrix_compute_core_bb.sv - DC boundary for the physical systolic array.
`ifndef MATRIX_COMPUTE_CORE_SV
`define MATRIX_COMPUTE_CORE_SV

(* blackbox *)
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
endmodule

`endif // MATRIX_COMPUTE_CORE_SV
