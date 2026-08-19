// matrix_int8_pe_probe.sv - registered-boundary synthesis probe.
`ifndef MATRIX_INT8_PE_PROBE_SV
`define MATRIX_INT8_PE_PROBE_SV

`include "matrix_int8_pe.sv"

// Real launch registers prevent a zero-delay top-level input assumption from
// hiding the clock-to-Q portion of an adjacent PE hop.  report_area -hierarchy
// separates u_pe from these probe-only flops.
module matrix_int8_pe_probe (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               weight_we,
  input  logic [7:0]         weight0_in,
  input  logic [7:0]         weight1_in,
  input  logic               in_valid,
  input  logic [7:0]         act0_in,
  input  logic [7:0]         act1_in,
  input  logic [31:0]        psum_in,
  output logic               act_valid_out,
  output logic [7:0]         act0_out,
  output logic [7:0]         act1_out,
  output logic               psum_valid_out,
  output logic [31:0]        psum_out
);
  logic               launch_valid;
  logic [7:0]         launch_act0, launch_act1;
  logic [31:0]        launch_psum;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      launch_valid <= 1'b0;
      launch_act0 <= 8'd0;
      launch_act1 <= 8'd0;
      launch_psum <= 32'd0;
    end else begin
      launch_valid <= in_valid;
      if (in_valid) begin
        launch_act0 <= act0_in;
        launch_act1 <= act1_in;
        launch_psum <= psum_in;
      end
    end
  end

  matrix_int8_pe u_pe (
    .clk(clk), .rst_n(rst_n),
    .weight_we(weight_we),
    .weight0_in(weight0_in), .weight1_in(weight1_in),
    .in_valid(launch_valid),
    .act0_west(launch_act0), .act1_west(launch_act1),
    .psum_north(launch_psum),
    .act_valid_east(act_valid_out),
    .act0_east(act0_out), .act1_east(act1_out),
    .psum_valid_south(psum_valid_out), .psum_south(psum_out)
  );
endmodule

`endif // MATRIX_INT8_PE_PROBE_SV
