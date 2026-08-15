// ddr_axi4_master unit test driver — toggles the clock until the self-checking
// FSM in ddr_axi4_tb.sv calls $finish, then reports its error count.
#include "Vddr_axi4_tb.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vddr_axi4_tb* tb = new Vddr_axi4_tb;

  // reset
  tb->clk = 0; tb->rst_n = 0;
  for (int i = 0; i < 4; i++) { tb->clk = 0; tb->eval(); tb->clk = 1; tb->eval(); }
  tb->rst_n = 1;

  long n = 0;
  while (!Verilated::gotFinish() && n < 200000) {
    tb->clk = 0; tb->eval();
    tb->clk = 1; tb->eval();
    n++;
  }
  if (n >= 200000 && !Verilated::gotFinish()) {
    printf("FAIL: ddr_axi4_tb did not finish (timeout)\n");
    tb->final(); delete tb; return 1;
  }
  uint32_t err = tb->err_count;
  tb->final();
  delete tb;
  return err == 0 ? 0 : 1;
}
