`include "softfloat.sv"
`include "rope_lut.sv"
// softfloat_check.sv — combinational harness exposing softfloat functions.
module softfloat_check #(
  parameter int N_CASES = 200000
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] a,
  input  logic [31:0] b,
  input  logic [3:0]  op,
  output logic [31:0] out,
  output logic [15:0] out_bf16
);
  import softfloat_pkg::*;
  import rope_lut_pkg::*;

  logic [63:0] sc;

  always_comb begin
    out = 32'b0;
    out_bf16 = 16'b0;
    sc = 64'b0;
    case (op)
      3'd0: out = fp32_add(a, b);
      3'd1: out = fp32_sub(a, b);
      3'd2: out = fp32_mul(a, b);
      3'd3: out = fp32_div(a, b);
      3'd4: out = fp32_exp(a);
      3'd5: out = fp32_rsqrt(a);
      3'd6: out = fp32_recip(a);
      3'd7: begin out = bf16_to_fp32(a[15:0]); end
      3'd8: begin out_bf16 = fp32_to_bf16(a); out = {16'b0, out_bf16}; end
      4'd9: out = fp32_exp2(a);
      4'd10: out = fp32_log2(a);
      4'd11: out = fp32_sin(a);
      4'd12: out = fp32_cos(a);
      4'd13: begin sc = rope_sincos(a); out = sc[31:0]; end          // sin
      4'd14: begin sc = rope_sincos(a); out = sc[63:32]; end         // cos
      default: out = 32'b0;
    endcase
  end
endmodule
