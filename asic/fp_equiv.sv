// ============================================================================
// fp_equiv.sv — functional-equivalence spot check for the pipelined
// synth_datapath primitives vs the frozen softfloat_pkg functions (P10b).
//
// Inputs are driven combinationally from a cycle counter.  The output seen at
// cycle `step` (inside the clocked check) corresponds to:
//   add_o/sub_o/i2f_o/f2i_o : input at step-2   (latency 2)
//   mul_o                    : input at step-3   (latency 3)
//   max_o/f2b_o              : input at step-1   (latency 1)
//   b2f_o                    : input at step     (latency 0, combinational)
// Build: asic/run_fp_equiv.sh
// ============================================================================
`ifndef FP_EQUIV_SV
`define FP_EQUIV_SV

`include "softfloat.sv"
`include "synth_datapath.sv"

module fp_equiv (
  input  logic clk,
  input  logic rst_n,
  output logic done,
  output logic pass,
  output logic [31:0] fail_count
);
  localparam int NV = 40;
  localparam logic [31:0] AV [0:NV-1] = '{
    32'h00000000, 32'h3F800000, 32'hBF800000, 32'h40000000, 32'h3F000000,
    32'hC0490FDB, 32'h40490FDB, 32'h3F7FFFFF, 32'h7F800000, 32'hFF800000,
    32'h7FC00000, 32'h7F800001, 32'h00000001, 32'h80000000, 32'h3E800000,
    32'h42C80000, 32'hC2C80000, 32'h3E000000, 32'h4E6E6B28, 32'h3E99999A,
    32'h3F000001, 32'h3F7F0000, 32'h41C80000, 32'hC1C80000, 32'h00000000,
    32'h3F800000, 32'h40000000, 32'h40400000, 32'h3FC00000, 32'hBF800000,
    32'h477FFF00, 32'h3DCCCCCD, 32'h7F7FFFFF, 32'h00800000, 32'hFF7FFFFF,
    32'h3F800000, 32'h3F000000, 32'h43FA0000, 32'hC3FA0000, 32'h7F800000
  };
  localparam logic [31:0] BV [0:NV-1] = '{
    32'h00000000, 32'h3F800000, 32'h3F800000, 32'h3F000000, 32'h40000000,
    32'h40490FDB, 32'hC0490FDB, 32'h3E800000, 32'h3F800000, 32'h3F800000,
    32'h3F800000, 32'h00000000, 32'h3F800000, 32'h3F800000, 32'h3F000000,
    32'h428C0000, 32'h42C80000, 32'h40800000, 32'h4E6E6B28, 32'h40A00000,
    32'h3F000000, 32'h3F7F0000, 32'h41C80000, 32'hC1C80000, 32'h3F800000,
    32'hBF800000, 32'h40400000, 32'h3FC00000, 32'h3F000000, 32'h3F800000,
    32'hC77FFF00, 32'h40A00000, 32'h00800000, 32'h7F7FFFFF, 32'hFF7FFFFF,
    32'h3F800000, 32'h3F800000, 32'h437A0000, 32'h43FA0000, 32'h00000000
  };
  localparam logic [31:0] IV [0:NV-1] = '{
    32'h00000000, 32'h00000001, 32'hFFFFFFFF, 32'h0000007F, 32'h00000080,
    32'h00FFFFFF, 32'h01000000, 32'h7FFFFFFF, 32'h80000000, 32'h00000003,
    32'h00000000, 32'h0000000F, 32'h7F800000, 32'h3F800000, 32'h00000005,
    32'h00000100, 32'hFFFFFF00, 32'h00000000, 32'h7FFFFFFF, 32'h80000000,
    32'h00000001, 32'h0000FFFF, 32'h7FFFFFFF, 32'h80000000, 32'h00000000,
    32'h00000002, 32'h00000100, 32'h00000000, 32'h00000040, 32'h00000001,
    32'h0000007F, 32'h00000000, 32'h00000001, 32'h7FFFFFFF, 32'h80000000,
    32'h00000001, 32'hFFFFFFFE, 32'h00000000, 32'h00000001, 32'h00000000
  };

  integer step;
  logic [31:0] a, b, i_in;
  logic [15:0] bf_in;
  logic [31:0] add_o, sub_o, mul_o, max_o, i2f_o, f2i_o, b2f_o;
  logic [15:0] f2b_o;

  assign a    = AV[step];
  assign b    = BV[step];
  assign i_in = IV[step];
  assign bf_in = AV[step][15:0];

  synth_datapath u_dp (
    .clk(clk), .rst_n(rst_n),
    .a(a), .b(b), .i_in(i_in), .bf_in(bf_in),
    .add_o(add_o), .sub_o(sub_o), .mul_o(mul_o), .max_o(max_o),
    .i2f_o(i2f_o), .f2i_o(f2i_o), .f2b_o(f2b_o), .b2f_o(b2f_o)
  );

  integer failures;
  logic   pass_r, done_r;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      step <= 0; failures <= 0; pass_r <= 0; done_r <= 0;
    end else begin
      if (step >= 3) begin
        if (add_o != softfloat_pkg::fp32_add(AV[step-2], BV[step-2])) begin
          failures <= failures + 1;
          $display("ADD step=%0d a=%h b=%h got=%h exp=%h", step, AV[step-2], BV[step-2],
                   add_o, softfloat_pkg::fp32_add(AV[step-2], BV[step-2]));
        end
        if (sub_o != softfloat_pkg::fp32_sub(AV[step-2], BV[step-2])) begin
          failures <= failures + 1;
          $display("SUB step=%0d a=%h b=%h got=%h exp=%h", step, AV[step-2], BV[step-2],
                   sub_o, softfloat_pkg::fp32_sub(AV[step-2], BV[step-2]));
        end
        if (mul_o != softfloat_pkg::fp32_mul(AV[step-3], BV[step-3])) begin
          failures <= failures + 1;
          $display("MUL step=%0d a=%h b=%h got=%h exp=%h", step, AV[step-3], BV[step-3],
                   mul_o, softfloat_pkg::fp32_mul(AV[step-3], BV[step-3]));
        end
        if (max_o != softfloat_pkg::fp32_max(AV[step-1], BV[step-1])) begin
          failures <= failures + 1;
          $display("MAX step=%0d a=%h b=%h got=%h exp=%h", step, AV[step-1], BV[step-1],
                   max_o, softfloat_pkg::fp32_max(AV[step-1], BV[step-1]));
        end
        if (i2f_o != softfloat_pkg::i32_to_f32(IV[step-2])) begin
          failures <= failures + 1;
          $display("I2F step=%0d i=%h got=%h exp=%h", step, IV[step-2],
                   i2f_o, softfloat_pkg::i32_to_f32(IV[step-2]));
        end
        if (f2i_o != softfloat_pkg::f32_to_i32_rne(AV[step-3])) begin
          failures <= failures + 1;
          $display("F2I step=%0d a=%h got=%h exp=%h", step, AV[step-3],
                   f2i_o, softfloat_pkg::f32_to_i32_rne(AV[step-3]));
        end

        if (f2b_o != softfloat_pkg::fp32_to_bf16(AV[step-1])) begin
          failures <= failures + 1;
          $display("F2B step=%0d a=%h got=%h exp=%h", step, AV[step-1],
                   f2b_o, softfloat_pkg::fp32_to_bf16(AV[step-1]));
        end
        if (b2f_o != softfloat_pkg::bf16_to_fp32(AV[step][15:0])) begin
          failures <= failures + 1;
          $display("B2F step=%0d bf=%h got=%h exp=%h", step, AV[step][15:0],
                   b2f_o, softfloat_pkg::bf16_to_fp32(AV[step][15:0]));
        end
      end
      step <= step + 1;
      if (step == NV + 8) begin
        pass_r <= (failures == 0);
        done_r <= 1'b1;
      end
    end
  end

  assign pass = pass_r;
  assign done = done_r;
  assign fail_count = failures;
endmodule

`endif // FP_EQUIV_SV
