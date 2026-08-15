// sram_check_main.cpp — Verilator harness for asic/sram_check.sv (P10b).
// Clocks the self-checking testbench until `done`, then reports pass/fail.
#include "Vsram_check.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vsram_check* top = new Vsram_check;

  top->clk = 0; top->rst_n = 0; top->eval();
  for (int i = 0; i < 4; i++) { top->clk = 0; top->eval(); top->clk = 1; top->eval(); }
  top->rst_n = 1; top->eval();

  int c = 0;
  const int maxc = 100000;
  for (; c < maxc && !top->done; c++) {
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
  }

  bool pass = top->pass;
  int fail = top->fail_count;
  std::printf("sram_check: done=%d pass=%d fail_count=%d (cycles=%d)\n",
              (int)top->done, (int)pass, fail, c);

  top->final();
  delete top;
  return pass ? 0 : 1;
}
