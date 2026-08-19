// Randomized and directed checks for the physical dual-INT8-MAC PE.
#include "Vmatrix_int8_pe.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>

double sc_time_stamp() { return 0.0; }

namespace {

void tick(Vmatrix_int8_pe* top) {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

uint32_t expected_psum(int8_t a0, int8_t a1, int8_t w0, int8_t w1,
                       uint32_t psum) {
  const int64_t signed_psum = (psum & 0x80000000u)
      ? static_cast<int64_t>(psum) - (int64_t{1} << 32)
      : static_cast<int64_t>(psum);
  const int64_t sum = signed_psum +
                      static_cast<int32_t>(a0) * static_cast<int32_t>(w0) +
                      static_cast<int32_t>(a1) * static_cast<int32_t>(w1);
  return static_cast<uint32_t>(sum);
}

uint32_t next_random(uint32_t* state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

int check_cycle(Vmatrix_int8_pe* top, int8_t a0, int8_t a1, int8_t w0,
                int8_t w1, uint32_t psum, int index) {
  top->in_valid = 1;
  top->act0_west = static_cast<uint8_t>(a0);
  top->act1_west = static_cast<uint8_t>(a1);
  top->psum_north = psum;
  tick(top);

  const uint32_t expected = expected_psum(a0, a1, w0, w1, psum);
  if (!top->act_valid_east || !top->psum_valid_south ||
      static_cast<uint8_t>(top->act0_east) != static_cast<uint8_t>(a0) ||
      static_cast<uint8_t>(top->act1_east) != static_cast<uint8_t>(a1) ||
      static_cast<uint32_t>(top->psum_south) != expected) {
    std::printf("FAIL cycle=%d a=(%d,%d) w=(%d,%d) psum=%08x got=%08x exp=%08x\n",
                index, a0, a1, w0, w1, psum,
                static_cast<uint32_t>(top->psum_south), expected);
    return 1;
  }
  return 0;
}

void load_weights(Vmatrix_int8_pe* top, int8_t w0, int8_t w1) {
  top->in_valid = 0;
  top->weight_we = 1;
  top->weight0_in = static_cast<uint8_t>(w0);
  top->weight1_in = static_cast<uint8_t>(w1);
  tick(top);
  top->weight_we = 0;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* top = new Vmatrix_int8_pe;
  top->clk = 0;
  top->rst_n = 0;
  top->weight_we = 0;
  top->in_valid = 0;
  tick(top);
  tick(top);
  top->rst_n = 1;

  int failures = 0;
  int checks = 0;

  load_weights(top, -128, 127);
  const int8_t edge_a0[] = {-128, -128, -1, 0, 1, 127, 127, 64};
  const int8_t edge_a1[] = {-128, 127, -1, 0, 1, -128, 127, -64};
  const uint32_t edge_psum[] = {
      0x00000000u, 0x7fffffffu, 0x80000000u, 0xffffffffu,
      0x00000001u, 0x7fffff00u, 0xffffff00u, 0x12345678u};
  for (int i = 0; i < 8; ++i) {
    failures += check_cycle(top, edge_a0[i], edge_a1[i], -128, 127,
                            edge_psum[i], checks++);
  }

  // A bubble must invalidate both channels without changing the held payload.
  const uint32_t held_psum = top->psum_south;
  top->in_valid = 0;
  tick(top);
  if (top->act_valid_east || top->psum_valid_south ||
      static_cast<uint32_t>(top->psum_south) != held_psum) {
    std::printf("FAIL bubble valid/payload contract\n");
    ++failures;
  }

  // Exhaust every signed 8-bit value for each activation/weight multiplier.
  // The lane-1 mappings are permutations, so it covers the same full domain.
  uint32_t random_state = 0x51434f52u;
  for (int block = 0; block < 256; ++block) {
    const int8_t w0 = static_cast<int8_t>(block);
    const int8_t w1 = static_cast<int8_t>(block ^ 0xa5);
    load_weights(top, w0, w1);
    for (int i = 0; i < 256; ++i) {
      const int8_t a0 = static_cast<int8_t>(i);
      const int8_t a1 = static_cast<int8_t>((i * 73 + block) & 0xff);
      const uint32_t psum = next_random(&random_state);
      failures += check_cycle(top, a0, a1, w0, w1, psum, checks++);
    }
  }

  std::printf("matrix_int8_pe_check: %s checks=%d fail_count=%d\n",
              failures == 0 ? "PASS" : "FAIL", checks, failures);
  top->final();
  delete top;
  return failures == 0 ? 0 : 1;
}
