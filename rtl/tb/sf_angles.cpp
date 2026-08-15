// sf_angles.cpp — evaluate fp32_sin/fp32_cos over a list of fp32 angles.
// Usage: sf_angles < input.hex  (one 8-hex-digit angle per line)
// Output: one line per input: "sin_hex cos_hex"
#include "Vsoftfloat_check.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vsoftfloat_check* d = new Vsoftfloat_check;
  d->clk = 0; d->rst_n = 1; d->b = 0;
  char line[32];
  while (fgets(line, sizeof line, stdin)) {
    uint32_t a = (uint32_t)strtoul(line, nullptr, 16);
    d->a = a;
    d->op = 11; d->eval(); uint32_t s = d->out;
    d->op = 12; d->eval(); uint32_t c = d->out;
    printf("%08x %08x\n", s, c);
  }
  delete d;
  return 0;
}
