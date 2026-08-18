// matrix_sram_check_main.cpp - directed verification for matrix_engine_sram.
#include "Vmatrix_engine.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

double sc_time_stamp() { return 0.0; }

namespace {

constexpr uint8_t DT_BF16 = 0;
constexpr uint8_t DT_INT8 = 2;
constexpr uint8_t DT_INT4 = 3;
constexpr uint32_t F_ONE = 0x3f800000u;
constexpr uint32_t F_THREE = 0x40400000u;

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void tick(Vmatrix_engine* top) {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

void idle_inputs(Vmatrix_engine* top) {
  top->start = 0;
  top->step = 0;
  top->cin_we = 0;
  top->scale_we = 0;
  top->cin_waddr = 0;
  top->cin_wdata = 0;
  top->scale_waddr = 0;
  top->scale_wdata = 0;
  top->c_raddr = 0;
  for (int i = 0; i < 128; ++i) {
    top->a_slice[i] = 0;
    top->b_slice[i] = 0;
  }
}

void write_cin(Vmatrix_engine* top, uint16_t addr, uint32_t data) {
  top->cin_waddr = addr;
  top->cin_wdata = data;
  top->cin_we = 1;
  tick(top);
  top->cin_we = 0;
}

void write_scale(Vmatrix_engine* top, uint16_t addr, uint32_t data) {
  top->scale_waddr = addr;
  top->scale_wdata = data;
  top->scale_we = 1;
  tick(top);
  top->scale_we = 0;
}

void start_case(Vmatrix_engine* top, int m, int n, int k, uint8_t src_a,
                uint8_t src_b, bool acc_init, bool dequant) {
  top->M = m;
  top->N = n;
  top->K = k;
  top->srcA = src_a;
  top->srcB = src_b;
  top->acc_init = acc_init;
  top->dequant = dequant;
  top->start = 1;
  tick(top);
  top->start = 0;
}

bool wait_done(Vmatrix_engine* top, const char* name) {
  // Model the CP's stale rd_ptr while it waits for done.  The matrix shell must
  // use the WAIT->RDOUT transition cycle to prefetch result element zero.
  top->c_raddr = 127;
  for (int cycle = 0; cycle < 64; ++cycle) {
    if (top->done) return true;
    tick(top);
  }
  std::printf("FAIL %-18s timeout waiting for done\n", name);
  return false;
}

uint32_t read_result(Vmatrix_engine* top, uint16_t addr) {
  top->c_raddr = addr;
  tick(top);
  tick(top);
  return top->c_rdata;
}

int check_results(Vmatrix_engine* top, const char* name,
                  const std::vector<uint32_t>& expected) {
  int failures = 0;
  tick(top);
  top->c_raddr = 0;
  top->eval();
  if (top->c_rdata != expected[0]) {
    ++failures;
    std::printf("FAIL %-18s first-prefetch got=%08x expected=%08x\n",
                name, top->c_rdata, expected[0]);
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    const uint32_t got = read_result(top, static_cast<uint16_t>(i));
    if (got != expected[i]) {
      ++failures;
      std::printf("FAIL %-18s idx=%zu got=%08x expected=%08x\n",
                  name, i, got, expected[i]);
    }
  }
  if (failures == 0)
    std::printf("PASS %-18s elements=%zu\n", name, expected.size());
  return failures;
}

int run_int8_accumulate(Vmatrix_engine* top) {
  constexpr int m = 2, n = 8, k = 5;
  std::vector<int32_t> expected(m * n, 0);
  start_case(top, m, n, k, DT_INT8, DT_INT8, true, false);
  for (int kk = 0; kk < k; ++kk) {
    for (int mm = 0; mm < m; ++mm) {
      const int32_t a = ((kk + mm * 2) % 7) - 3;
      top->a_slice[mm] = static_cast<uint32_t>(a);
    }
    for (int nn = 0; nn < n; ++nn) {
      const int32_t b = ((kk * 3 + nn) % 9) - 4;
      top->b_slice[nn] = static_cast<uint32_t>(b);
    }
    for (int idx = 0; idx < m * n; ++idx) {
      top->step = 1;
      tick(top);
      const int mm = idx / n;
      const int nn = idx % n;
      const int32_t a = ((kk + mm * 2) % 7) - 3;
      const int32_t b = ((kk * 3 + nn) % 9) - 4;
      expected[idx] += a * b;
    }
    top->step = 0;
  }
  if (!wait_done(top, "int8-accumulate")) return 1;
  std::vector<uint32_t> bits(expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    bits[i] = static_cast<uint32_t>(expected[i]);
  return check_results(top, "int8-accumulate", bits);
}

int run_bf16_seed(Vmatrix_engine* top) {
  constexpr int m = 1, n = 8, k = 2;
  for (int i = 0; i < m * n; ++i) write_cin(top, i, F_THREE);
  start_case(top, m, n, k, DT_BF16, DT_BF16, false, false);
  for (int kk = 0; kk < k; ++kk) {
    top->a_slice[0] = (kk == 0) ? F_ONE : 0x40000000u;
    for (int nn = 0; nn < n; ++nn) top->b_slice[nn] = F_ONE;
    for (int idx = 0; idx < m * n; ++idx) {
      top->step = 1;
      tick(top);
    }
    top->step = 0;
  }
  if (!wait_done(top, "bf16-c-seed")) return 1;
  return check_results(top, "bf16-c-seed",
                       std::vector<uint32_t>(m * n, 0x40c00000u));  // 6.0
}

int run_int8_dequant(Vmatrix_engine* top) {
  constexpr int m = 1, n = 8, k = 128;
  for (int nn = 0; nn < n; ++nn) write_scale(top, nn, F_ONE);
  start_case(top, m, n, k, DT_INT8, DT_INT8, true, true);
  for (int kk = 0; kk < k; ++kk) {
    top->a_slice[0] = 1;
    for (int nn = 0; nn < n; ++nn) top->b_slice[nn] = nn + 1;
    for (int idx = 0; idx < m * n; ++idx) {
      top->step = 1;
      tick(top);
    }
    top->step = 0;
  }
  if (!wait_done(top, "int8-dequant")) return 1;
  std::vector<uint32_t> expected(n);
  for (int nn = 0; nn < n; ++nn)
    expected[nn] = float_bits(static_cast<float>(128 * (nn + 1)));
  return check_results(top, "int8-dequant", expected);
}

int run_int4_dequant(Vmatrix_engine* top) {
  constexpr int m = 1, n = 8, k = 128;
  const int32_t weight[n] = {1, 2, 3, 4, 5, 6, 7, -1};
  for (int nn = 0; nn < n; ++nn) write_scale(top, nn, F_ONE);
  start_case(top, m, n, k, DT_BF16, DT_INT4, true, true);
  for (int kk = 0; kk < k; ++kk) {
    top->a_slice[0] = F_ONE;
    for (int nn = 0; nn < n; ++nn)
      top->b_slice[nn] = static_cast<uint32_t>(weight[nn]);
    for (int idx = 0; idx < m * n; ++idx) {
      top->step = 1;
      tick(top);
    }
    top->step = 0;
  }
  if (!wait_done(top, "int4-dequant")) return 1;
  std::vector<uint32_t> expected(n);
  for (int nn = 0; nn < n; ++nn)
    expected[nn] = float_bits(static_cast<float>(128 * weight[nn]));
  return check_results(top, "int4-dequant", expected);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* top = new Vmatrix_engine;
  idle_inputs(top);
  top->rst_n = 0;
  for (int i = 0; i < 4; ++i) tick(top);
  top->rst_n = 1;
  tick(top);

  int failures = 0;
  failures += run_int8_accumulate(top);
  failures += run_bf16_seed(top);
  failures += run_int8_dequant(top);
  failures += run_int4_dequant(top);

  std::printf("matrix_sram_check: %s fail_count=%d\n",
              failures == 0 ? "PASS" : "FAIL", failures);
  top->final();
  delete top;
  return failures == 0 ? 0 : 1;
}
