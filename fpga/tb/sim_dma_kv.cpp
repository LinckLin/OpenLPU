// dma_kv_stage unit test driver — toggles the clock until the self-checking
// FSM in dma_kv_tb.sv calls $finish, then reports its error count.
#include "Vdma_kv_tb.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vdma_kv_tb* tb = new Vdma_kv_tb;

  // reset
  tb->clk = 0; tb->rst_n = 0;
  for (int i = 0; i < 4; i++) { tb->clk = 0; tb->eval(); tb->clk = 1; tb->eval(); }
  tb->rst_n = 1;

  long n = 0;
  while (!Verilated::gotFinish() && n < 2000000) {
    tb->clk = 0; tb->eval();
    tb->clk = 1; tb->eval();
    n++;
  }
  if (n >= 2000000 && !Verilated::gotFinish()) {
    printf("FAIL: dma_kv_tb did not finish (timeout)\n");
    tb->final(); delete tb; return 1;
  }
  uint32_t err = tb->err_count;
  tb->final();
  delete tb;
  return err == 0 ? 0 : 1;
}
