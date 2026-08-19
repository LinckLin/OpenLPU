// matrix_int8_pe.sv - physical INT8 dual-MAC processing element.
//
// This is the arithmetic primitive for the frozen 128x128 weight-stationary
// array.  Each PE owns two signed INT8 weights, consumes two activations and a
// vertical INT32 partial sum, and produces two MACs every valid cycle.  The
// registered east/south boundaries keep one PE hop per cycle, preserving the
// specified 128-column + 128-row = 256-cycle array fill/drain distance.
`ifndef MATRIX_INT8_PE_SV
`define MATRIX_INT8_PE_SV

module matrix_int8_pe #(
  // The standalone PE/tile probes use one bank.  The full array sets this to
  // two so an inactive PF weight bank can be filled while the active bank is
  // still feeding the MACs.
  parameter integer WEIGHT_BANKS = 1
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               weight_we,
  input  logic               weight_bank_sel,
  input  logic               active_bank_sel,
  input  logic [7:0]         weight0_in,
  input  logic [7:0]         weight1_in,
  input  logic               in_valid,
  input  logic [7:0]         act0_west,
  input  logic [7:0]         act1_west,
  input  logic [31:0]        psum_north,
  output logic               act_valid_east,
  output logic [7:0]         act0_east,
  output logic [7:0]         act1_east,
  output logic               psum_valid_south,
  output logic [31:0]        psum_south
);
  logic [7:0] active_weight0, active_weight1;
  logic [15:0] product0, product1;
  logic [31:0] product0_ext, product1_ext;
  logic [31:0] next_psum;

  // Keep the single-bank implementation physically identical to the 2.4a
  // PE.  The dual-bank branch adds only the inactive-bank storage; the MAC
  // datapath always reads the selected active bank.
  generate
    if (WEIGHT_BANKS == 1) begin : gen_single_weight_bank
      logic [7:0] weight0, weight1;

      always_ff @(posedge clk) begin
        if (!rst_n) begin
          weight0 <= 8'd0;
          weight1 <= 8'd0;
        end else if (weight_we) begin
          weight0 <= weight0_in;
          weight1 <= weight1_in;
        end
      end

      assign active_weight0 = weight0;
      assign active_weight1 = weight1;
    end else begin : gen_dual_weight_bank
      logic [7:0] weight0_bank [0:1];
      logic [7:0] weight1_bank [0:1];

      always_ff @(posedge clk) begin
        if (!rst_n) begin
          weight0_bank[0] <= 8'd0;
          weight0_bank[1] <= 8'd0;
          weight1_bank[0] <= 8'd0;
          weight1_bank[1] <= 8'd0;
        end else if (weight_we) begin
          weight0_bank[weight_bank_sel] <= weight0_in;
          weight1_bank[weight_bank_sel] <= weight1_in;
        end
      end

      assign active_weight0 = weight0_bank[active_bank_sel];
      assign active_weight1 = weight1_bank[active_bank_sel];
    end
  endgenerate

  function automatic logic [15:0] signed_mul8(
    input logic [7:0] lhs,
    input logic [7:0] rhs
  );
    logic [7:0] lhs_mag, rhs_mag;
    logic [15:0] magnitude;
    begin
      lhs_mag = lhs[7] ? (~lhs + 8'd1) : lhs;
      rhs_mag = rhs[7] ? (~rhs + 8'd1) : rhs;
      magnitude = lhs_mag * rhs_mag;
      signed_mul8 = (lhs[7] ^ rhs[7]) ? (~magnitude + 16'd1) : magnitude;
    end
  endfunction

  always_comb begin
    product0 = signed_mul8(act0_west, active_weight0);
    product1 = signed_mul8(act1_west, active_weight1);
    product0_ext = {{16{product0[15]}}, product0};
    product1_ext = {{16{product1[15]}}, product1};
    next_psum = psum_north + product0_ext + product1_ext;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      act_valid_east <= 1'b0;
      act0_east <= 8'd0;
      act1_east <= 8'd0;
      psum_valid_south <= 1'b0;
      psum_south <= 32'd0;
    end else begin
      act_valid_east <= in_valid;
      psum_valid_south <= in_valid;
      if (in_valid) begin
        act0_east <= act0_west;
        act1_east <= act1_west;
        psum_south <= next_psum;
      end
    end
  end

  // Weight loading and compute occupy separate array phases.
  // synopsys translate_off
  always_ff @(posedge clk) begin
    if (rst_n && weight_we && in_valid &&
        ((WEIGHT_BANKS == 1) || (weight_bank_sel == active_bank_sel)))
      $error("matrix_int8_pe does not allow weight load and compute together");
  end
  // synopsys translate_on
endmodule

`endif // MATRIX_INT8_PE_SV
