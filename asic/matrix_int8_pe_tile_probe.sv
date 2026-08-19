// matrix_int8_pe_tile_probe.sv - registered-boundary 16x16 tile probe.
`ifndef MATRIX_INT8_PE_TILE_PROBE_SV
`define MATRIX_INT8_PE_TILE_PROBE_SV

`include "matrix_int8_pe_tile.sv"

// Launch registers model the west/north boundary of an adjacent tile.  The
// tile's internal PE hierarchy remains visible to DC for area/timing reports.
module matrix_int8_pe_tile_probe (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         weight_we_in,
  input  logic [3:0]   weight_row_in,
  input  logic [3:0]   weight_col_in,
  input  logic [7:0]   weight0_in,
  input  logic [7:0]   weight1_in,
  input  logic [15:0]  act_valid_west_in,
  input  logic [127:0] act0_west_in,
  input  logic [127:0] act1_west_in,
  input  logic [15:0]  psum_valid_north_in,
  input  logic [511:0] psum_north_in,
  output logic [15:0]  act_valid_east_out,
  output logic [127:0] act0_east_out,
  output logic [127:0] act1_east_out,
  output logic [15:0]  psum_valid_south_out,
  output logic [511:0] psum_south_out
);
  logic         weight_we;
  logic [3:0]   weight_row, weight_col;
  logic [7:0]   weight0, weight1;
  logic [15:0]  act_valid_west;
  logic [127:0] act0_west, act1_west;
  logic [15:0]  psum_valid_north;
  logic [511:0] psum_north;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      weight_we <= 1'b0;
      weight_row <= 4'd0;
      weight_col <= 4'd0;
      weight0 <= 8'd0;
      weight1 <= 8'd0;
      act_valid_west <= 16'd0;
      act0_west <= 128'd0;
      act1_west <= 128'd0;
      psum_valid_north <= 16'd0;
      psum_north <= 512'd0;
    end else begin
      weight_we <= weight_we_in;
      weight_row <= weight_row_in;
      weight_col <= weight_col_in;
      weight0 <= weight0_in;
      weight1 <= weight1_in;
      act_valid_west <= act_valid_west_in;
      act0_west <= act0_west_in;
      act1_west <= act1_west_in;
      psum_valid_north <= psum_valid_north_in;
      psum_north <= psum_north_in;
    end
  end

  matrix_int8_pe_tile u_tile (
    .clk(clk),
    .rst_n(rst_n),
    .weight_we(weight_we),
    .weight_row_we(1'b0),
    .weight_bank_sel(1'b0),
    .active_bank_sel(1'b0),
    .weight_row(weight_row),
    .weight_col(weight_col),
    .weight0_in(weight0),
    .weight1_in(weight1),
    .weight0_row_in(128'b0),
    .weight1_row_in(128'b0),
    .act_valid_west(act_valid_west),
    .act0_west(act0_west),
    .act1_west(act1_west),
    .psum_valid_north(psum_valid_north),
    .psum_north(psum_north),
    .act_valid_east(act_valid_east_out),
    .act0_east(act0_east_out),
    .act1_east(act1_east_out),
    .psum_valid_south(psum_valid_south_out),
    .psum_south(psum_south_out)
  );
endmodule

`endif // MATRIX_INT8_PE_TILE_PROBE_SV
