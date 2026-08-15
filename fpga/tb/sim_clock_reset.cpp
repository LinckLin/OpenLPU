// clock_reset unit test driver — checks async assert + synchronous release
// (RST_STAGES=3) of fpga/clock_reset.sv.
#include "Vclock_reset_tb.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vclock_reset_tb* tb = new Vclock_reset_tb;
  int errors = 0;

  // settle with async reset deasserted
  tb->clk_i = 0; tb->async_rst_n_i = 1; tb->eval();

  // -- async assert: rst_n must drop without a clock edge -----------------
  tb->async_rst_n_i = 0; tb->eval();
  if (tb->rst_n != 0) { printf("FAIL: rst_n not asserted asynchronously\n"); errors++; }

  // -- release: rst_n must stay low for RST_STAGES-1 posedges ------------
  tb->async_rst_n_i = 1; tb->eval();
  for (int i = 0; i < 2; i++) {           // 2 posedges (RST_STAGES=3)
    tb->clk_i = 1; tb->eval();
    tb->clk_i = 0; tb->eval();
  }
  if (tb->rst_n != 0) { printf("FAIL: rst_n released too early (after 2 posedges)\n"); errors++; }

  // 3rd posedge -> rst_n must be high
  tb->clk_i = 1; tb->eval();
  tb->clk_i = 0; tb->eval();
  if (tb->rst_n != 1) { printf("FAIL: rst_n not released after 3 posedges\n"); errors++; }

  // clk passthrough
  if (tb->clk != tb->clk_i) { printf("FAIL: clk passthrough broken\n"); errors++; }

  if (errors == 0) printf("PASS: clock_reset async assert + sync release (3 stages)\n");
  else             printf("FAIL: %d error(s)\n", errors);

  tb->final();
  delete tb;
  return errors == 0 ? 0 : 1;
}
