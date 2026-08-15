// sim_sram_shrink.cpp — Verilator harness for fpga/tb/sram_shrink_tb.sv.
// Clocks the self-checking testbench until `done`, then reports pass/fail.
#include "Vsram_shrink_tb.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vsram_shrink_tb* top = new Vsram_shrink_tb;

  top->clk = 0; top->rst_n = 0; top->eval();
  for (int i = 0; i < 4; i++) { top->clk = 0; top->eval(); top->clk = 1; top->eval(); }
  top->rst_n = 1; top->eval();

  int c = 0;
  const int maxc = 1000000;
  for (; c < maxc && !top->done; c++) {
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
  }

  bool pass = top->pass;
  int fail = top->fail_count;
  std::printf("sram_shrink_tb: done=%d pass=%d fail_count=%d (cycles=%d)\n",
              (int)top->done, (int)pass, fail, c);

  top->final();
  delete top;
  return pass ? 0 : 1;
}
