// matrix_int8_pe_array_probe.sv - registered-boundary probe for the full grid.
//
// The default parameters are the production 8x8 tile grid.  Reduced grids
// are used by the fast functional scoreboard; they preserve the same global
// address, mode, and wavefront contracts.
`ifndef MATRIX_INT8_PE_ARRAY_PROBE_SV
`define MATRIX_INT8_PE_ARRAY_PROBE_SV

`include "matrix_int8_pe_array.sv"

module matrix_int8_pe_array_probe #(
  parameter integer TILE_ROWS = 8,
  parameter integer TILE_COLS = 8,
  parameter integer PE_ROWS = TILE_ROWS * 16,
  parameter integer PE_COLS = TILE_COLS * 16,
  parameter integer ROW_ADDR_W = (PE_ROWS > 16) ? $clog2(PE_ROWS) : 4,
  parameter integer MODE_SWITCH_CYCLES = 300
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    mode_dc_req_in,
  output logic                    mode_dc_active_out,
  output logic                    mode_switch_busy_out,
  input  logic                    weight_load_we_in,
  input  logic                    weight_load_bank_in,
  input  logic [ROW_ADDR_W-1:0]   weight_load_row_in,
  input  logic [PE_COLS*8-1:0]    weight0_row_in,
  input  logic [PE_COLS*8-1:0]    weight1_row_in,
  input  logic                    weight_commit_in,
  input  logic                    weight_commit_bank_in,
  output logic                    active_weight_bank_out,
  input  logic [PE_ROWS-1:0]      pf_act_valid_west_in,
  input  logic [PE_ROWS*8-1:0]    pf_act0_west_in,
  input  logic [PE_ROWS*8-1:0]    pf_act1_west_in,
  input  logic [PE_COLS-1:0]      pf_psum_valid_north_in,
  input  logic [PE_COLS*32-1:0]   pf_psum_north_in,
  input  logic [PE_ROWS-1:0]      dc_act_valid_batch_in,
  input  logic [PE_ROWS*8-1:0]    dc_act0_batch_in,
  input  logic [PE_ROWS*8-1:0]    dc_act1_batch_in,
  input  logic [PE_COLS-1:0]      dc_psum_valid_north_in,
  input  logic [PE_COLS*32-1:0]   dc_psum_north_in,
  output logic [PE_ROWS-1:0]      act_valid_east_out,
  output logic [PE_ROWS*8-1:0]    act0_east_out,
  output logic [PE_ROWS*8-1:0]    act1_east_out,
  output logic [PE_COLS-1:0]      psum_valid_south_out,
  output logic [PE_COLS*32-1:0]   psum_south_out
);
  logic mode_dc_req;
  logic weight_load_we, weight_load_bank;
  logic [ROW_ADDR_W-1:0] weight_load_row;
  logic [PE_COLS*8-1:0] weight0_row, weight1_row;
  logic weight_commit, weight_commit_bank;
  logic [PE_ROWS-1:0] pf_act_valid_west;
  logic [PE_ROWS*8-1:0] pf_act0_west, pf_act1_west;
  logic [PE_COLS-1:0] pf_psum_valid_north;
  logic [PE_COLS*32-1:0] pf_psum_north;
  logic [PE_ROWS-1:0] dc_act_valid_batch;
  logic [PE_ROWS*8-1:0] dc_act0_batch, dc_act1_batch;
  logic [PE_COLS-1:0] dc_psum_valid_north;
  logic [PE_COLS*32-1:0] dc_psum_north;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      mode_dc_req <= 1'b0;
      weight_load_we <= 1'b0;
      weight_load_bank <= 1'b0;
      weight_load_row <= '0;
      weight0_row <= '0;
      weight1_row <= '0;
      weight_commit <= 1'b0;
      weight_commit_bank <= 1'b0;
      pf_act_valid_west <= '0;
      pf_act0_west <= '0;
      pf_act1_west <= '0;
      pf_psum_valid_north <= '0;
      pf_psum_north <= '0;
      dc_act_valid_batch <= '0;
      dc_act0_batch <= '0;
      dc_act1_batch <= '0;
      dc_psum_valid_north <= '0;
      dc_psum_north <= '0;
    end else begin
      mode_dc_req <= mode_dc_req_in;
      weight_load_we <= weight_load_we_in;
      weight_load_bank <= weight_load_bank_in;
      weight_load_row <= weight_load_row_in;
      weight0_row <= weight0_row_in;
      weight1_row <= weight1_row_in;
      weight_commit <= weight_commit_in;
      weight_commit_bank <= weight_commit_bank_in;
      pf_act_valid_west <= pf_act_valid_west_in;
      pf_act0_west <= pf_act0_west_in;
      pf_act1_west <= pf_act1_west_in;
      pf_psum_valid_north <= pf_psum_valid_north_in;
      pf_psum_north <= pf_psum_north_in;
      dc_act_valid_batch <= dc_act_valid_batch_in;
      dc_act0_batch <= dc_act0_batch_in;
      dc_act1_batch <= dc_act1_batch_in;
      dc_psum_valid_north <= dc_psum_valid_north_in;
      dc_psum_north <= dc_psum_north_in;
    end
  end

  matrix_int8_pe_array #(
    .TILE_ROWS(TILE_ROWS),
    .TILE_COLS(TILE_COLS),
    .PE_ROWS(PE_ROWS),
    .PE_COLS(PE_COLS),
    .ROW_ADDR_W(ROW_ADDR_W),
    .MODE_SWITCH_CYCLES(MODE_SWITCH_CYCLES)
  ) u_array (
    .clk(clk), .rst_n(rst_n),
    .mode_dc_req(mode_dc_req),
    .mode_dc_active(mode_dc_active_out),
    .mode_switch_busy(mode_switch_busy_out),
    .weight_load_we(weight_load_we),
    .weight_load_bank(weight_load_bank),
    .weight_load_row(weight_load_row),
    .weight0_row_in(weight0_row), .weight1_row_in(weight1_row),
    .weight_commit(weight_commit),
    .weight_commit_bank(weight_commit_bank),
    .active_weight_bank(active_weight_bank_out),
    .pf_act_valid_west(pf_act_valid_west),
    .pf_act0_west(pf_act0_west), .pf_act1_west(pf_act1_west),
    .pf_psum_valid_north(pf_psum_valid_north),
    .pf_psum_north(pf_psum_north),
    .dc_act_valid_batch(dc_act_valid_batch),
    .dc_act0_batch(dc_act0_batch), .dc_act1_batch(dc_act1_batch),
    .dc_psum_valid_north(dc_psum_valid_north),
    .dc_psum_north(dc_psum_north),
    .act_valid_east(act_valid_east_out),
    .act0_east(act0_east_out), .act1_east(act1_east_out),
    .psum_valid_south(psum_valid_south_out),
    .psum_south(psum_south_out)
  );
endmodule

`endif // MATRIX_INT8_PE_ARRAY_PROBE_SV
