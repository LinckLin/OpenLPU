// sim_softfloat.cpp — drive softfloat_check with generated inputs, dump results.
#include "Vsoftfloat_check.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

// xorshift32 for deterministic inputs.
static uint32_t xs = 0x9E3779B9u;
static uint32_t rnd() {
  xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs;
}

static uint32_t rand_fp32() {
  // bias toward normal-range values (no denormals / inf), like v0 activations
  uint32_t r = rnd();
  uint32_t e = ((r >> 23) & 0x7F) + 0x38;   // exp in [56,183] -> 2^-71 .. 2^56
  uint32_t m = r & 0x7FFFFF;
  uint32_t s = (r >> 31) & 1;
  if (e > 0xFE) e = 0xFE;
  return (s << 31) | (e << 23) | m;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vsoftfloat_check* dut = new Vsoftfloat_check;
  int n = 200000;
  for (int i = 0; i < n; i++) {
    uint32_t a = rand_fp32();
    uint32_t b = rand_fp32();
    // make b normal-range too for div/recip denominator safety
    int op = (int)(rnd() % 9);
    // restrict exp to avoid overflow/underflow noise: renormalize a,b
    dut->a = a; dut->b = b; dut->op = op;
    dut->eval();
    printf("%d %08x %08x %08x\n", op, a, b, dut->out);
  }
  delete dut;
  return 0;
}
