// Focused TT mapped-gate check for the 16x16 dual-MAC PE tile.
`timescale 1ns/1ps

module matrix_int8_pe_tile_gate_tb;
  localparam integer ROWS = 16;
  localparam integer COLS = 16;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic weight_we_in = 1'b0;
  logic [3:0] weight_row_in = 4'd0;
  logic [3:0] weight_col_in = 4'd0;
  logic [7:0] weight0_in = 8'd0;
  logic [7:0] weight1_in = 8'd0;
  logic [15:0] act_valid_west_in = 16'd0;
  logic [127:0] act0_west_in = 128'd0;
  logic [127:0] act1_west_in = 128'd0;
  logic [15:0] psum_valid_north_in = 16'd0;
  logic [511:0] psum_north_in = 512'd0;
  wire [15:0] act_valid_east_out;
  wire [127:0] act0_east_out;
  wire [127:0] act1_east_out;
  wire [15:0] psum_valid_south_out;
  wire [511:0] psum_south_out;

  matrix_int8_pe_tile_probe dut (
    .clk(clk), .rst_n(rst_n),
    .weight_we_in(weight_we_in), .weight_row_in(weight_row_in),
    .weight_col_in(weight_col_in), .weight0_in(weight0_in),
    .weight1_in(weight1_in),
    .act_valid_west_in(act_valid_west_in),
    .act0_west_in(act0_west_in), .act1_west_in(act1_west_in),
    .psum_valid_north_in(psum_valid_north_in),
    .psum_north_in(psum_north_in),
    .act_valid_east_out(act_valid_east_out),
    .act0_east_out(act0_east_out), .act1_east_out(act1_east_out),
    .psum_valid_south_out(psum_valid_south_out),
    .psum_south_out(psum_south_out)
  );

  always #5 clk = ~clk;

  logic signed [7:0] weight0 [0:ROWS-1][0:COLS-1];
  logic signed [7:0] weight1 [0:ROWS-1][0:COLS-1];
  logic signed [7:0] act0 [0:ROWS-1];
  logic signed [7:0] act1 [0:ROWS-1];
  logic [31:0] psum [0:COLS-1];
  integer failures = 0;
  integer checks = 0;

  function automatic [31:0] ref_psum(input integer col);
    integer row;
    reg signed [63:0] sum;
    begin
      sum = $signed(psum[col]);
      for (row = 0; row < ROWS; row = row + 1) begin
        sum = sum + act0[row] * weight0[row][col];
        sum = sum + act1[row] * weight1[row][col];
      end
      ref_psum = sum[31:0];
    end
  endfunction

  task automatic clear_boundary;
    begin
      act_valid_west_in = 16'd0;
      act0_west_in = 128'd0;
      act1_west_in = 128'd0;
      psum_valid_north_in = 16'd0;
      psum_north_in = 512'd0;
    end
  endtask

  integer row, col, cycle;
  initial begin
    for (row = 0; row < ROWS; row = row + 1) begin
      act0[row] = row[7:0] ^ 8'h81;
      act1[row] = (row * 13 + 7) & 8'hff;
      for (col = 0; col < COLS; col = col + 1) begin
        weight0[row][col] = (row * 17 + col * 3) & 8'hff;
        weight1[row][col] = (row * 5 - col * 19) & 8'hff;
      end
    end
    weight0[0][0] = -128;
    weight1[15][15] = -128;
    for (col = 0; col < COLS; col = col + 1)
      psum[col] = 32'h80000000 ^ (col * 32'h01020304);

    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    // Serial addressed weight phase.
    for (row = 0; row < ROWS; row = row + 1) begin
      for (col = 0; col < COLS; col = col + 1) begin
        @(negedge clk);
        weight_we_in = 1'b1;
        weight_row_in = row;
        weight_col_in = col;
        weight0_in = weight0[row][col];
        weight1_in = weight1[row][col];
        clear_boundary();
        @(posedge clk);
      end
    end
    @(negedge clk);
    weight_we_in = 1'b0;
    clear_boundary();

    // One complete wavefront.  West row r launches at t=r; north column c
    // launches at t=c, so each PE sees both operands at t=r+c.
    for (cycle = 0; cycle <= ROWS + COLS + 2; cycle = cycle + 1) begin
      @(negedge clk);
      clear_boundary();
      for (row = 0; row < ROWS; row = row + 1) begin
        if (cycle == row) begin
          act_valid_west_in[row] = 1'b1;
          act0_west_in[row*8 +: 8] = act0[row];
          act1_west_in[row*8 +: 8] = act1[row];
        end
      end
      for (col = 0; col < COLS; col = col + 1) begin
        if (cycle == col) begin
          psum_valid_north_in[col] = 1'b1;
          psum_north_in[col*32 +: 32] = psum[col];
        end
      end
      @(posedge clk);
      #1;
      for (row = 0; row < ROWS; row = row + 1) begin
        checks = checks + 1;
        if (((act_valid_east_out >> row) & 1'b1) !== (cycle == ROWS + row)) begin
          $display("FAIL tile gate east valid cycle=%0d row=%0d", cycle, row);
          failures = failures + 1;
        end
        if (cycle == ROWS + row) begin
          checks = checks + 2;
          if (act0_east_out[row*8 +: 8] !== act0[row] ||
              act1_east_out[row*8 +: 8] !== act1[row]) begin
            $display("FAIL tile gate east payload cycle=%0d row=%0d", cycle, row);
            failures = failures + 1;
          end
        end
      end
      for (col = 0; col < COLS; col = col + 1) begin
        checks = checks + 1;
        if (((psum_valid_south_out >> col) & 1'b1) !== (cycle == ROWS + col)) begin
          $display("FAIL tile gate south valid cycle=%0d col=%0d", cycle, col);
          failures = failures + 1;
        end
        if (cycle == ROWS + col) begin
          checks = checks + 1;
          if (psum_south_out[col*32 +: 32] !== ref_psum(col)) begin
            $display("FAIL tile gate south payload cycle=%0d col=%0d got=%h exp=%h",
                     cycle, col, psum_south_out[col*32 +: 32], ref_psum(col));
            failures = failures + 1;
          end
        end
      end
    end

    $display("matrix_int8_pe_tile_gate_check: %s checks=%0d fail_count=%0d",
             failures == 0 ? "PASS" : "FAIL", checks, failures);
    if (failures != 0) $fatal(1, "mapped tile mismatch");
    $finish;
  end
endmodule
