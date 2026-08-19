// matrix_int8_pe_array.sv - structural 8x8 grid of 16x16 dual-MAC tiles.
//
// The default instance is the frozen 128x128 PE array (64 tiles).  Smaller
// TILE_ROWS/TILE_COLS values are intentionally supported for fast RTL checks;
// the tile-to-tile wiring and control semantics are unchanged.
`ifndef MATRIX_INT8_PE_ARRAY_SV
`define MATRIX_INT8_PE_ARRAY_SV

`include "matrix_int8_pe_tile.sv"

module matrix_int8_pe_array #(
  parameter integer TILE_ROWS = 8,
  parameter integer TILE_COLS = 8,
  parameter integer PE_ROWS = TILE_ROWS * 16,
  parameter integer PE_COLS = TILE_COLS * 16,
  parameter integer ROW_ADDR_W = 7,
  parameter integer MODE_SWITCH_CYCLES = 300
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // MODE is changed only after the current wavefront has drained.  A request
  // transition inserts MODE_SWITCH_CYCLES of an explicitly invalid output
  // window (the frozen contract is approximately 300 cycles).
  input  logic                    mode_dc_req,
  output logic                    mode_dc_active,
  output logic                    mode_switch_busy,

  // PF weight-stationary double buffer.  Loads may target the inactive bank
  // while the active bank is computing; commit selects the fully loaded bank.
  input  logic                    weight_load_we,
  input  logic                    weight_load_bank,
  input  logic [ROW_ADDR_W-1:0]  weight_load_row,
  input  logic [PE_COLS*8-1:0]    weight0_row_in,
  input  logic [PE_COLS*8-1:0]    weight1_row_in,
  input  logic                    weight_commit,
  input  logic                    weight_commit_bank,
  output logic                    active_weight_bank,

  // PF boundary: callers provide the usual row/column wavefront skew.
  input  logic [PE_ROWS-1:0]      pf_act_valid_west,
  input  logic [PE_ROWS*8-1:0]    pf_act0_west,
  input  logic [PE_ROWS*8-1:0]    pf_act1_west,
  input  logic [PE_COLS-1:0]      pf_psum_valid_north,
  input  logic [PE_COLS*32-1:0]   pf_psum_north,

  // DC boundary: one batch contains 16 lanes x 8 rows.  The reconstruction
  // pipeline delays global row g by g cycles, converting the lane batch into
  // the same 128-row wavefront consumed by the physical grid.
  input  logic [PE_ROWS-1:0]      dc_act_valid_batch,
  input  logic [PE_ROWS*8-1:0]    dc_act0_batch,
  input  logic [PE_ROWS*8-1:0]    dc_act1_batch,
  input  logic [PE_COLS-1:0]      dc_psum_valid_north,
  input  logic [PE_COLS*32-1:0]   dc_psum_north,

  // East/south boundaries are shared by PF and DC.  The data payloads hold
  // their last registered value when valid is low, matching matrix_int8_pe.
  output logic [PE_ROWS-1:0]      act_valid_east,
  output logic [PE_ROWS*8-1:0]    act0_east,
  output logic [PE_ROWS*8-1:0]    act1_east,
  output logic [PE_COLS-1:0]      psum_valid_south,
  output logic [PE_COLS*32-1:0]   psum_south
);
  localparam integer TILE_SIZE = 16;
  localparam integer TILE_ACT_W = TILE_SIZE * 8;
  localparam integer TILE_PSUM_W = TILE_SIZE * 32;
  localparam integer TILE_ROW_ADDR_W = (ROW_ADDR_W > 4) ? ROW_ADDR_W - 4 : 1;
  localparam integer MODE_COUNT_W =
      (MODE_SWITCH_CYCLES > 1) ? $clog2(MODE_SWITCH_CYCLES) : 1;
  localparam integer MODE_LAST_COUNT_INT = MODE_SWITCH_CYCLES - 1;
  localparam logic [MODE_COUNT_W-1:0] MODE_LAST_COUNT =
      MODE_LAST_COUNT_INT[MODE_COUNT_W-1:0];
  localparam integer ARRAY_DRAIN_CYCLES =
      (PE_ROWS > PE_COLS) ? PE_ROWS : PE_COLS;
  localparam integer ARRAY_DRAIN_COUNT_W =
      (ARRAY_DRAIN_CYCLES > 1) ? $clog2(ARRAY_DRAIN_CYCLES + 1) : 1;

  logic [MODE_COUNT_W-1:0] mode_switch_count;
  logic [ARRAY_DRAIN_COUNT_W-1:0] array_drain_count;
  wire mode_switch_pending = (mode_dc_req != mode_dc_active);

  // A full row-vector pipeline is deliberately used here.  It makes the
  // lane-to-global-row timing visible in RTL and supports one DC batch every
  // cycle after the initial 128-cycle reconstruction latency.  The packed
  // representation avoids variable-index nonblocking assignments, which are
  // rejected by the installed Verilator version and by older DC frontends.
  localparam integer DC_VALID_PIPE_W = PE_ROWS * PE_ROWS;
  localparam integer DC_DATA_PIPE_W = PE_ROWS * PE_ROWS * 8;
  logic [DC_VALID_PIPE_W-1:0] dc_valid_pipe;
  logic [DC_DATA_PIPE_W-1:0] dc_act0_pipe;
  logic [DC_DATA_PIPE_W-1:0] dc_act1_pipe;
  logic [PE_ROWS-1:0]    dc_act_valid_reconstructed;
  logic [PE_ROWS*8-1:0]  dc_act0_reconstructed;
  logic [PE_ROWS*8-1:0]  dc_act1_reconstructed;
  logic [PE_COLS-1:0]    dc_psum_valid_north_q;
  logic [PE_COLS*32-1:0] dc_psum_north_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      dc_valid_pipe <= '0;
      dc_act0_pipe <= '0;
      dc_act1_pipe <= '0;
      dc_psum_valid_north_q <= '0;
      dc_psum_north_q <= '0;
    end else begin
      // Flush stale DC data whenever PF is active or a mode transition is in
      // progress.  This also guarantees that the 300-cycle barrier drains
      // the longest (128-cycle) reconstruction pipeline.
      if (mode_dc_active && !mode_switch_busy && !mode_switch_pending) begin
        dc_valid_pipe <= {
          dc_valid_pipe[DC_VALID_PIPE_W-PE_ROWS-1:0],
          dc_act_valid_batch
        };
        dc_act0_pipe <= {
          dc_act0_pipe[DC_DATA_PIPE_W-PE_ROWS*8-1:0],
          dc_act0_batch
        };
        dc_act1_pipe <= {
          dc_act1_pipe[DC_DATA_PIPE_W-PE_ROWS*8-1:0],
          dc_act1_batch
        };
        // The north stream gets the same one-cycle boundary launch as the
        // reconstructed activation stream.  Without this register, row 0's
        // activation would meet column 0's partial sum one cycle early.
        dc_psum_valid_north_q <= dc_psum_valid_north;
        dc_psum_north_q <= dc_psum_north;
      end else begin
        dc_valid_pipe <= {
          dc_valid_pipe[DC_VALID_PIPE_W-PE_ROWS-1:0], {PE_ROWS{1'b0}}
        };
        dc_act0_pipe <= {
          dc_act0_pipe[DC_DATA_PIPE_W-PE_ROWS*8-1:0],
          {(PE_ROWS*8){1'b0}}
        };
        dc_act1_pipe <= {
          dc_act1_pipe[DC_DATA_PIPE_W-PE_ROWS*8-1:0],
          {(PE_ROWS*8){1'b0}}
        };
        dc_psum_valid_north_q <= '0;
        dc_psum_north_q <= '0;
      end
    end
  end

  genvar row;
  generate
    for (row = 0; row < PE_ROWS; row = row + 1) begin : gen_dc_rows
      // Row g is sourced from pipeline stage g.  The input vector layout is
      // lane-major with eight rows per lane, i.e. global row = lane*8+row.
      assign dc_act_valid_reconstructed[row] =
          dc_valid_pipe[row*PE_ROWS + row];
      assign dc_act0_reconstructed[row*8 +: 8] =
          dc_act0_pipe[row*PE_ROWS*8 + row*8 +: 8];
      assign dc_act1_reconstructed[row*8 +: 8] =
          dc_act1_pipe[row*PE_ROWS*8 + row*8 +: 8];
    end
  endgenerate

  logic [PE_ROWS-1:0]   selected_act_valid_west;
  logic [PE_ROWS*8-1:0] selected_act0_west;
  logic [PE_ROWS*8-1:0] selected_act1_west;
  logic [PE_COLS-1:0]   selected_psum_valid_north;
  logic [PE_COLS*32-1:0] selected_psum_north;

  always_comb begin
    selected_act_valid_west = '0;
    selected_act0_west = '0;
    selected_act1_west = '0;
    selected_psum_valid_north = '0;
    selected_psum_north = '0;
    if (!mode_switch_busy) begin
      if (mode_dc_active) begin
        selected_act_valid_west = dc_act_valid_reconstructed;
        selected_act0_west = dc_act0_reconstructed;
        selected_act1_west = dc_act1_reconstructed;
        selected_psum_valid_north = dc_psum_valid_north_q;
        selected_psum_north = dc_psum_north_q;
      end else begin
        selected_act_valid_west = pf_act_valid_west;
        selected_act0_west = pf_act0_west;
        selected_act1_west = pf_act1_west;
        selected_psum_valid_north = pf_psum_valid_north;
        selected_psum_north = pf_psum_north;
      end
    end
  end

  // Keep weight-bank commits behind the last in-flight PE hop.  Boundary
  // vectors are already skewed, so after the final launch the longest
  // remaining path is max(rows, columns), not rows+columns.
  wire boundary_launch_active =
      (|selected_act_valid_west) || (|selected_psum_valid_north);
  wire array_wave_active = boundary_launch_active || (array_drain_count != 0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      array_drain_count <= '0;
    end else if (boundary_launch_active) begin
      array_drain_count <= ARRAY_DRAIN_CYCLES;
    end else if (array_drain_count != 0) begin
      array_drain_count <= array_drain_count - 1'b1;
    end
  end

  // MODE and PF weight-bank commit state.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      mode_dc_active <= 1'b0;
      mode_switch_busy <= 1'b0;
      mode_switch_count <= '0;
      active_weight_bank <= 1'b0;
    end else begin
      if (mode_switch_busy) begin
        if (mode_switch_count == MODE_LAST_COUNT) begin
          mode_switch_busy <= 1'b0;
          mode_switch_count <= '0;
          mode_dc_active <= mode_dc_req;
        end else begin
          mode_switch_count <= mode_switch_count + 1'b1;
        end
      end else if (mode_switch_pending) begin
        mode_switch_busy <= 1'b1;
        mode_switch_count <= '0;
      end

      if (weight_commit && !mode_switch_busy && !array_wave_active)
        active_weight_bank <= weight_commit_bank;
    end
  end

  logic [TILE_SIZE-1:0] tile_act_valid_west [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_ACT_W-1:0] tile_act0_west [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_ACT_W-1:0] tile_act1_west [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_SIZE-1:0] tile_act_valid_east [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_ACT_W-1:0] tile_act0_east [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_ACT_W-1:0] tile_act1_east [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_SIZE-1:0] tile_psum_valid_north [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_PSUM_W-1:0] tile_psum_north [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_SIZE-1:0] tile_psum_valid_south [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic [TILE_PSUM_W-1:0] tile_psum_south [0:TILE_ROWS-1][0:TILE_COLS-1];
  logic tile_weight_row_we [0:TILE_ROWS-1][0:TILE_COLS-1];

  logic [TILE_ROW_ADDR_W-1:0] tile_row_addr;
  generate
    if (ROW_ADDR_W > 4) begin : gen_tile_row_addr
      assign tile_row_addr = weight_load_row[ROW_ADDR_W-1:4];
    end else begin : gen_single_tile_row_addr
      assign tile_row_addr = '0;
    end
  endgenerate

  genvar tr, tc;
  generate
    for (tr = 0; tr < TILE_ROWS; tr = tr + 1) begin : gen_tile_row
      for (tc = 0; tc < TILE_COLS; tc = tc + 1) begin : gen_tile_col
        if (tc == 0) begin : gen_array_west
          assign tile_act_valid_west[tr][tc] =
              selected_act_valid_west[tr*TILE_SIZE +: TILE_SIZE];
          assign tile_act0_west[tr][tc] =
              selected_act0_west[tr*TILE_ACT_W +: TILE_ACT_W];
          assign tile_act1_west[tr][tc] =
              selected_act1_west[tr*TILE_ACT_W +: TILE_ACT_W];
        end else begin : gen_tile_west
          assign tile_act_valid_west[tr][tc] = tile_act_valid_east[tr][tc-1];
          assign tile_act0_west[tr][tc] = tile_act0_east[tr][tc-1];
          assign tile_act1_west[tr][tc] = tile_act1_east[tr][tc-1];
        end

        if (tr == 0) begin : gen_array_north
          assign tile_psum_valid_north[tr][tc] =
              selected_psum_valid_north[tc*TILE_SIZE +: TILE_SIZE];
          assign tile_psum_north[tr][tc] =
              selected_psum_north[tc*TILE_PSUM_W +: TILE_PSUM_W];
        end else begin : gen_tile_north
          assign tile_psum_valid_north[tr][tc] =
              tile_psum_valid_south[tr-1][tc];
          assign tile_psum_north[tr][tc] = tile_psum_south[tr-1][tc];
        end

        // One global row write updates 128 PEs (two bytes per PE) in one
        // cycle: 256 B/cycle, exactly the PF overlap-refresh contract.
        assign tile_weight_row_we[tr][tc] = weight_load_we &&
            (tile_row_addr == tr);

        matrix_int8_pe_tile #(.WEIGHT_BANKS(2)) u_tile (
          .clk(clk),
          .rst_n(rst_n),
          .weight_we(1'b0),
          .weight_row_we(tile_weight_row_we[tr][tc]),
          .weight_bank_sel(weight_load_bank),
          .active_bank_sel(active_weight_bank),
          .weight_row(weight_load_row[3:0]),
          .weight_col(4'b0),
          .weight0_in(8'b0),
          .weight1_in(8'b0),
          .weight0_row_in(weight0_row_in[tc*TILE_ACT_W +: TILE_ACT_W]),
          .weight1_row_in(weight1_row_in[tc*TILE_ACT_W +: TILE_ACT_W]),
          .act_valid_west(tile_act_valid_west[tr][tc]),
          .act0_west(tile_act0_west[tr][tc]),
          .act1_west(tile_act1_west[tr][tc]),
          .psum_valid_north(tile_psum_valid_north[tr][tc]),
          .psum_north(tile_psum_north[tr][tc]),
          .act_valid_east(tile_act_valid_east[tr][tc]),
          .act0_east(tile_act0_east[tr][tc]),
          .act1_east(tile_act1_east[tr][tc]),
          .psum_valid_south(tile_psum_valid_south[tr][tc]),
          .psum_south(tile_psum_south[tr][tc])
        );
      end
    end
  endgenerate

  // Flatten the final tile's per-row/per-column payloads into global vectors.
  // The explicit generate assignments avoid relying on streaming-concat
  // support in older DC/Verilog frontends.  Each output bit/slice has one
  // driver, including the valid gate during a MODE barrier.
  generate
    for (tr = 0; tr < TILE_ROWS; tr = tr + 1) begin : gen_all_final_rows
      for (row = 0; row < TILE_SIZE; row = row + 1) begin : gen_row_slice
        assign act_valid_east[tr*TILE_SIZE + row] = mode_switch_busy ? 1'b0 :
            tile_act_valid_east[tr][TILE_COLS-1][row];
        assign act0_east[(tr*TILE_SIZE + row)*8 +: 8] =
            tile_act0_east[tr][TILE_COLS-1][row*8 +: 8];
        assign act1_east[(tr*TILE_SIZE + row)*8 +: 8] =
            tile_act1_east[tr][TILE_COLS-1][row*8 +: 8];
      end
    end
    for (tc = 0; tc < TILE_COLS; tc = tc + 1) begin : gen_all_final_cols
      for (row = 0; row < TILE_SIZE; row = row + 1) begin : gen_col_slice
        assign psum_valid_south[tc*TILE_SIZE + row] = mode_switch_busy ? 1'b0 :
            tile_psum_valid_south[TILE_ROWS-1][tc][row];
        assign psum_south[(tc*TILE_SIZE + row)*32 +: 32] =
            tile_psum_south[TILE_ROWS-1][tc][row*32 +: 32];
      end
    end
  endgenerate

  // Commit and active-bank overlap are protocol errors.  Loading the
  // inactive bank during a wave is intentionally legal and is checked in the
  // PE/tile assertions using weight_load_bank vs active_weight_bank.
  // synopsys translate_off
  always_ff @(posedge clk) begin
    if (rst_n && weight_commit && array_wave_active)
      $error("matrix_int8_pe_array weight commit overlaps in-flight wavefront");
    if (rst_n && weight_load_we && weight_commit)
      $error("matrix_int8_pe_array load and commit share a cycle");
  end
  // synopsys translate_on
endmodule

`endif // MATRIX_INT8_PE_ARRAY_SV
