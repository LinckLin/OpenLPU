// Quick check of exp2/log2/sin/cos against a few fixed inputs.
#include "Vsoftfloat_check.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

static uint32_t f(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float u(uint32_t x) { float v; memcpy(&v, &x, 4); return v; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vsoftfloat_check* d = new Vsoftfloat_check;
  d->clk = 0; d->rst_n = 1; d->b = 0;
  struct { int op; float x; } cases[] = {
    {9, 0.0f}, {9, 1.0f}, {9, -0.3114f}, {9, 0.3114f}, {9, 2.0f}, {9, -3.0f},
    {10, 1000000.0f}, {10, 2.0f}, {10, 0.5f}, {10, 1.0f},
    {11, 42.0f}, {11, 33.9f}, {11, 0.0f}, {11, 1.5707963f},
    {12, 42.0f}, {12, 33.9f}, {12, 0.0f},
  };
  for (auto& c : cases) {
    d->a = f(c.x); d->op = c.op; d->eval();
    printf("op=%d x=%.9g -> 0x%08x (%.9g)\n", c.op, c.x, d->out, u(d->out));
  }
  delete d;
  return 0;
}
