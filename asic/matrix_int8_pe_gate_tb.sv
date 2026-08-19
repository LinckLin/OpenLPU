// Independent mapped-gate scoreboard for matrix_int8_pe_probe.
`timescale 1ns/1ps

module matrix_int8_pe_gate_tb;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic weight_we = 1'b0;
  logic [7:0] weight0_in = 8'b0;
  logic [7:0] weight1_in = 8'b0;
  logic in_valid = 1'b0;
  logic [7:0] act0_in = 8'b0;
  logic [7:0] act1_in = 8'b0;
  logic [31:0] psum_in = 32'b0;
  wire act_valid_out;
  wire [7:0] act0_out;
  wire [7:0] act1_out;
  wire psum_valid_out;
  wire [31:0] psum_out;

  matrix_int8_pe_probe dut (
    .clk(clk), .rst_n(rst_n), .weight_we(weight_we),
    .weight0_in(weight0_in), .weight1_in(weight1_in),
    .in_valid(in_valid), .act0_in(act0_in), .act1_in(act1_in),
    .psum_in(psum_in), .act_valid_out(act_valid_out),
    .act0_out(act0_out), .act1_out(act1_out),
    .psum_valid_out(psum_valid_out), .psum_out(psum_out)
  );

  always #5 clk = ~clk;

  function automatic [15:0] ref_mul8(
    input logic [7:0] lhs,
    input logic [7:0] rhs
  );
    logic signed [7:0] lhs_s, rhs_s;
    logic signed [15:0] product_s;
    begin
      lhs_s = lhs;
      rhs_s = rhs;
      product_s = lhs_s * rhs_s;
      ref_mul8 = product_s;
    end
  endfunction

  function automatic [31:0] ref_psum(
    input logic [7:0] a0,
    input logic [7:0] a1,
    input logic [7:0] w0,
    input logic [7:0] w1,
    input logic [31:0] psum
  );
    logic [15:0] p0, p1;
    begin
      p0 = ref_mul8(a0, w0);
      p1 = ref_mul8(a1, w1);
      ref_psum = psum + {{16{p0[15]}}, p0} + {{16{p1[15]}}, p1};
    end
  endfunction

  function automatic [31:0] random_next(input logic [31:0] value);
    logic [31:0] x;
    begin
      x = value;
      x = x ^ (x << 13);
      x = x ^ (x >> 17);
      x = x ^ (x << 5);
      random_next = x;
    end
  endfunction

  logic expected_valid = 1'b0;
  logic [7:0] expected_a0, expected_a1;
  logic [31:0] expected_psum;
  integer failures = 0;
  integer checks = 0;

  task automatic drive_cycle(
    input logic valid,
    input logic [7:0] a0,
    input logic [7:0] a1,
    input logic [31:0] psum
  );
    begin
      @(negedge clk);
      in_valid = valid;
      act0_in = a0;
      act1_in = a1;
      psum_in = psum;
      @(posedge clk);
      #1;
      if ((act_valid_out !== expected_valid) ||
          (psum_valid_out !== expected_valid)) begin
        $display("FAIL gate valid got=%b/%b expected=%b",
                 act_valid_out, psum_valid_out, expected_valid);
        failures = failures + 1;
      end else if (expected_valid) begin
        if ((act0_out !== expected_a0) || (act1_out !== expected_a1) ||
            (psum_out !== expected_psum)) begin
          $display("FAIL gate got=(%h,%h,%h) expected=(%h,%h,%h)",
                   act0_out, act1_out, psum_out,
                   expected_a0, expected_a1, expected_psum);
          failures = failures + 1;
        end
      end
      expected_valid = valid;
      if (valid) begin
        expected_a0 = a0;
        expected_a1 = a1;
        expected_psum = ref_psum(a0, a1, weight0_in, weight1_in, psum);
        checks = checks + 1;
      end
    end
  endtask

  integer i;
  logic [31:0] random_state;
  logic [7:0] next_a0, next_a1;
  logic [31:0] next_psum;
  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    weight0_in = 8'h8b;  // -117
    weight1_in = 8'h67;  // +103
    weight_we = 1'b1;
    @(posedge clk);
    #1;
    @(negedge clk);
    weight_we = 1'b0;

    random_state = 32'h50454741;
    for (i = 0; i < 2048; i = i + 1) begin
      random_state = random_next(random_state);
      next_a0 = random_state[7:0];
      random_state = random_next(random_state);
      next_a1 = random_state[7:0];
      random_state = random_next(random_state);
      next_psum = random_state;
      drive_cycle((i % 17) != 8, next_a0, next_a1, next_psum);
    end
    drive_cycle(1'b0, 8'b0, 8'b0, 32'b0);
    drive_cycle(1'b0, 8'b0, 8'b0, 32'b0);

    $display("matrix_int8_pe_gate_check: %s checks=%0d fail_count=%0d",
             failures == 0 ? "PASS" : "FAIL", checks, failures);
    if (failures != 0) $fatal(1, "mapped PE mismatch");
    $finish;
  end
endmodule
