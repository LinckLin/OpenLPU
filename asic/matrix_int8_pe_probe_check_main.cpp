// Scoreboard shared by RTL-probe and mapped-gate PE verification.
#include "Vmatrix_int8_pe_probe.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <deque>

double sc_time_stamp() { return 0.0; }

namespace {

struct Expected {
  uint8_t a0;
  uint8_t a1;
  uint32_t psum;
};

void tick(Vmatrix_int8_pe_probe* top) {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

uint32_t result(int8_t a0, int8_t a1, int8_t w0, int8_t w1,
                uint32_t psum) {
  const int64_t signed_psum = (psum & 0x80000000u)
      ? static_cast<int64_t>(psum) - (int64_t{1} << 32)
      : static_cast<int64_t>(psum);
  return static_cast<uint32_t>(
      signed_psum + static_cast<int32_t>(a0) * static_cast<int32_t>(w0) +
      static_cast<int32_t>(a1) * static_cast<int32_t>(w1));
}

uint32_t random_word(uint32_t* state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

int check_output(Vmatrix_int8_pe_probe* top, std::deque<Expected>* queue,
                 int cycle) {
  if (!top->psum_valid_out) return 0;
  if (queue->empty()) {
    std::printf("FAIL probe cycle=%d unexpected valid\n", cycle);
    return 1;
  }
  const Expected expected = queue->front();
  queue->pop_front();
  if (!top->act_valid_out ||
      static_cast<uint8_t>(top->act0_out) != expected.a0 ||
      static_cast<uint8_t>(top->act1_out) != expected.a1 ||
      static_cast<uint32_t>(top->psum_out) != expected.psum) {
    std::printf("FAIL probe cycle=%d got=(%02x,%02x,%08x) exp=(%02x,%02x,%08x)\n",
                cycle, static_cast<uint8_t>(top->act0_out),
                static_cast<uint8_t>(top->act1_out),
                static_cast<uint32_t>(top->psum_out), expected.a0, expected.a1,
                expected.psum);
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* top = new Vmatrix_int8_pe_probe;
  top->clk = 0;
  top->rst_n = 0;
  top->weight_we = 0;
  top->in_valid = 0;
  tick(top);
  tick(top);
  top->rst_n = 1;

  constexpr int8_t w0 = -117;
  constexpr int8_t w1 = 103;
  top->weight_we = 1;
  top->weight0_in = static_cast<uint8_t>(w0);
  top->weight1_in = static_cast<uint8_t>(w1);
  tick(top);
  top->weight_we = 0;

  int failures = 0;
  int checks = 0;
  int cycle = 0;
  uint32_t state = 0x50454741u;
  std::deque<Expected> queue;

  for (int i = 0; i < 1024; ++i) {
    const int8_t a0 = static_cast<int8_t>(random_word(&state));
    const int8_t a1 = static_cast<int8_t>(random_word(&state));
    const uint32_t psum = random_word(&state);
    top->in_valid = (i % 17) != 8;
    top->act0_in = static_cast<uint8_t>(a0);
    top->act1_in = static_cast<uint8_t>(a1);
    top->psum_in = psum;
    if (top->in_valid) {
      queue.push_back(Expected{static_cast<uint8_t>(a0),
                               static_cast<uint8_t>(a1),
                               result(a0, a1, w0, w1, psum)});
      ++checks;
    }
    tick(top);
    failures += check_output(top, &queue, cycle++);
  }

  top->in_valid = 0;
  for (int i = 0; i < 4; ++i) {
    tick(top);
    failures += check_output(top, &queue, cycle++);
  }
  if (!queue.empty()) {
    std::printf("FAIL probe undrained expected=%zu\n", queue.size());
    ++failures;
  }

  std::printf("matrix_int8_pe_probe_check: %s checks=%d fail_count=%d\n",
              failures == 0 ? "PASS" : "FAIL", checks, failures);
  top->final();
  delete top;
  return failures == 0 ? 0 : 1;
}
