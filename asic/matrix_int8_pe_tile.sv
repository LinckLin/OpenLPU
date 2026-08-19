// matrix_int8_pe_tile.sv - 16x16 structural dual-MAC PE tile.
//
// The tile is a physical hierarchy unit, not a reduced functional model.  A
// west activation wavefront and a north partial-sum wavefront are independently
// skewed at the boundary; at PE(row,col) they meet at launch_time+row+col.
// Each PE still advances one hop per cycle, so the tile is an exact 16x16
// sub-array of the frozen 128x128 weight-stationary geometry.
`ifndef MATRIX_INT8_PE_TILE_SV
`define MATRIX_INT8_PE_TILE_SV

`include "matrix_int8_pe.sv"

module matrix_int8_pe_tile #(
  parameter integer WEIGHT_BANKS = 1
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         weight_we,
  input  logic         weight_row_we,
  input  logic         weight_bank_sel,
  input  logic         active_bank_sel,
  input  logic [3:0]   weight_row,
  input  logic [3:0]   weight_col,
  input  logic [7:0]   weight0_in,
  input  logic [7:0]   weight1_in,
  input  logic [127:0] weight0_row_in,
  input  logic [127:0] weight1_row_in,
  input  logic [15:0]  act_valid_west,
  input  logic [127:0] act0_west,
  input  logic [127:0] act1_west,
  input  logic [15:0]  psum_valid_north,
  input  logic [511:0] psum_north,
  output logic [15:0]  act_valid_east,
  output logic [127:0] act0_east,
  output logic [127:0] act1_east,
  output logic [15:0]  psum_valid_south,
  output logic [511:0] psum_south
);
  localparam integer ROWS = 16;
  localparam integer COLS = 16;

  logic               act_valid_mesh [0:ROWS-1][0:COLS-1];
  logic [7:0]         act0_mesh      [0:ROWS-1][0:COLS-1];
  logic [7:0]         act1_mesh      [0:ROWS-1][0:COLS-1];
  logic               psum_valid_mesh[0:ROWS-1][0:COLS-1];
  logic [31:0]        psum_mesh      [0:ROWS-1][0:COLS-1];

  genvar r, c;
  generate
    for (r = 0; r < ROWS; r = r + 1) begin : gen_row
      for (c = 0; c < COLS; c = c + 1) begin : gen_col
        logic       pe_in_valid;
        logic       pe_psum_valid;
        logic [7:0] pe_act0_west;
        logic [7:0] pe_act1_west;
        logic [31:0] pe_psum_north;
        logic       pe_weight_we;
        logic [7:0] pe_weight0_in;
        logic [7:0] pe_weight1_in;
        logic       pe_act_valid_east;
        logic [7:0] pe_act0_east;
        logic [7:0] pe_act1_east;
        logic       pe_psum_valid_south;
        logic [31:0] pe_psum_south;

        if (c == 0) begin : gen_west_boundary
          assign pe_in_valid = act_valid_west[r];
          assign pe_act0_west = act0_west[r*8 +: 8];
          assign pe_act1_west = act1_west[r*8 +: 8];
        end else begin : gen_internal_west
          assign pe_in_valid = act_valid_mesh[r][c-1];
          assign pe_act0_west = act0_mesh[r][c-1];
          assign pe_act1_west = act1_mesh[r][c-1];
        end

        if (r == 0) begin : gen_north_boundary
          assign pe_psum_valid = psum_valid_north[c];
          assign pe_psum_north = psum_north[c*32 +: 32];
        end else begin : gen_internal_north
          assign pe_psum_valid = psum_valid_mesh[r-1][c];
          assign pe_psum_north = psum_mesh[r-1][c];
        end

        // A single addressed write fans out through the row/column decoder;
        // compute and weight-load phases remain mutually exclusive.
        assign pe_weight_we =
            (weight_we && (weight_row == r) && (weight_col == c)) ||
            (weight_row_we && (weight_row == r));
        assign pe_weight0_in = weight_row_we ?
            weight0_row_in[c*8 +: 8] : weight0_in;
        assign pe_weight1_in = weight_row_we ?
            weight1_row_in[c*8 +: 8] : weight1_in;

        matrix_int8_pe #(.WEIGHT_BANKS(WEIGHT_BANKS)) u_pe (
          .clk(clk),
          .rst_n(rst_n),
          .weight_we(pe_weight_we),
          .weight_bank_sel(weight_bank_sel),
          .active_bank_sel(active_bank_sel),
          .weight0_in(pe_weight0_in),
          .weight1_in(pe_weight1_in),
          .in_valid(pe_in_valid),
          .act0_west(pe_act0_west),
          .act1_west(pe_act1_west),
          .psum_north(pe_psum_north),
          .act_valid_east(pe_act_valid_east),
          .act0_east(pe_act0_east),
          .act1_east(pe_act1_east),
          .psum_valid_south(pe_psum_valid_south),
          .psum_south(pe_psum_south)
        );

        assign act_valid_mesh[r][c] = pe_act_valid_east;
        assign act0_mesh[r][c] = pe_act0_east;
        assign act1_mesh[r][c] = pe_act1_east;
        assign psum_valid_mesh[r][c] = pe_psum_valid_south;
        assign psum_mesh[r][c] = pe_psum_south;

        // The physical PE has one compute enable.  The external skew contract
        // requires the activation and partial-sum wavefronts to meet together.
        // Catch a malformed boundary schedule in simulation instead of silently
        // accumulating a stale north value.
        // synopsys translate_off
        always_ff @(posedge clk) begin
          if (rst_n && (pe_in_valid !== pe_psum_valid))
            $error("matrix_int8_pe_tile skew mismatch row=%0d col=%0d act=%b psum=%b",
                   r, c, pe_in_valid, pe_psum_valid);
        end
        // synopsys translate_on
      end
    end
  endgenerate

  generate
    for (r = 0; r < ROWS; r = r + 1) begin : gen_east_outputs
      assign act_valid_east[r] = act_valid_mesh[r][COLS-1];
      assign act0_east[r*8 +: 8] = act0_mesh[r][COLS-1];
      assign act1_east[r*8 +: 8] = act1_mesh[r][COLS-1];
    end
    for (c = 0; c < COLS; c = c + 1) begin : gen_south_outputs
      assign psum_valid_south[c] = psum_valid_mesh[ROWS-1][c];
      assign psum_south[c*32 +: 32] = psum_mesh[ROWS-1][c];
    end
  endgenerate

  // Weight loading is a separate phase for the whole tile.  This assertion
  // prevents a caller from changing one PE's weights under an active wavefront.
  // synopsys translate_off
  always_ff @(posedge clk) begin
    if (rst_n && (weight_we || weight_row_we) &&
        ((WEIGHT_BANKS == 1) || (weight_bank_sel == active_bank_sel)) &&
        (|act_valid_west || |psum_valid_north))
      $error("matrix_int8_pe_tile weight load overlaps boundary wavefront");
    if (rst_n && weight_we && weight_row_we)
      $error("matrix_int8_pe_tile scalar and row weight writes overlap");
  end
  // synopsys translate_on
endmodule

`endif // MATRIX_INT8_PE_TILE_SV
