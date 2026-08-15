// fp_equiv_main.cpp — Verilator harness for asic/fp_equiv.sv (P10b).
#include "Vfp_equiv.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vfp_equiv* top = new Vfp_equiv;
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
  std::printf("fp_equiv: done=%d pass=%d fail_count=%d (cycles=%d)\n",
              (int)top->done, (int)pass, fail, c);
  top->final();
  delete top;
  return pass ? 0 : 1;
}
