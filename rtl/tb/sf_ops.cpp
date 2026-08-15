// sf_ops.cpp — evaluate softfloat ops (log2, exp2, mul, sin, cos) for ROPE angle analysis.
// stdin lines: "op hex_a [hex_b]" ; output one hex per line.
#include "Vsoftfloat_check.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vsoftfloat_check* d = new Vsoftfloat_check;
  d->clk = 0; d->rst_n = 1;
  char line[64];
  while (fgets(line, sizeof line, stdin)) {
    uint32_t op = 0, a = 0, b = 0;
    int n = sscanf(line, "%u %x %x", &op, &a, &b);
    d->a = a; d->b = b; d->op = (op & 0xF);
    d->eval();
    printf("%08x\n", d->out);
  }
  delete d;
  return 0;
}
