// synth_mac.sv — QCore per-MAC representative units (P10/M9, pipelined P10b).
//
// P10b change (plans/m8-wait-plan.md §3): the MAC units are pipelined into
// clocked stages.  The BF16 MAC = fp32_mul (3 stages) + fp32_add (2 stages) =
// 5 stages; the INT8 MAC = 8x8 multiply + 32-bit accumulate = 2 stages.
// Throughput stays 1 MAC/cycle (fully pipelined).
//
// Relationship to the frozen cycle model: the matrix engine accumulates over
// the K dimension as a *tree* (log-depth reduction), not a serial chain, so
// the MAC pipeline latency does not multiply into matrix_pf_cycles()
// (ceil(K/256)*M + 256 — the "+256" already charges fill/drain).  The +1-cycle
// MAC accumulator is folded into that fill/drain.  This representative top
// measures the per-MAC datapath delay; the tree depth is reported separately.
`ifndef SYNTH_MAC_SV
`define SYNTH_MAC_SV

`include "synth_datapath.sv"

module mac_bf16 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] a,
  input  logic [31:0] b,
  input  logic [31:0] acc,
  output logic [31:0] y
);
  // 3-stage multiply + 2-stage add = 5-stage fused MAC datapath.
  logic [31:0] prod;
  fp32_mul3 u_mul (.clk(clk), .rst_n(rst_n), .a(a), .b(b), .y(prod));
  fp32_add2 u_add (.clk(clk), .rst_n(rst_n), .a(prod), .b(acc), .y(y));
endmodule

module mac_int8 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [7:0]  a,
  input  logic [7:0]  b,
  input  logic [31:0] acc,
  output logic [31:0] y
);
  // stage 0: 8x8 multiply; stage 1: 32-bit accumulate.
  logic signed [15:0] prod;
  always_ff @(posedge clk) begin
    if (!rst_n) prod <= 0;
    else        prod <= $signed(a) * $signed(b);
  end

  logic [31:0] y_c;
  assign y_c = acc + {{16{prod[15]}}, prod};

  always_ff @(posedge clk) begin
    if (!rst_n) y <= 0;
    else        y <= y_c;
  end
endmodule

`endif // SYNTH_MAC_SV
