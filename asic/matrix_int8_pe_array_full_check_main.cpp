// Full 8x8-tile (128x128 PE) PF wavefront scoreboard.
#include "Vmatrix_int8_pe_array_probe.h"
#include "verilated.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

double sc_time_stamp() { return 0.0; }

namespace {

constexpr int ROWS = 128;
constexpr int COLS = 128;
constexpr int WAVES = 256;

void tick(Vmatrix_int8_pe_array_probe* top) {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

void put_bit(WData* bus, int bit) {
  bus[bit / 32] |= uint32_t{1} << (bit % 32);
}

bool get_bit(const WData* bus, int bit) {
  return ((bus[bit / 32] >> (bit % 32)) & 1u) != 0;
}

void put_byte(WData* bus, int byte_index, uint8_t value) {
  const int word = byte_index / 4;
  const int shift = (byte_index % 4) * 8;
  const uint32_t mask = uint32_t{0xff} << shift;
  bus[word] = (bus[word] & ~mask) | (uint32_t{value} << shift);
}

uint8_t get_byte(const WData* bus, int byte_index) {
  const int word = byte_index / 4;
  const int shift = (byte_index % 4) * 8;
  return static_cast<uint8_t>((bus[word] >> shift) & 0xffu);
}

int8_t weight0(int row, int col) {
  return static_cast<int8_t>(((row + 3 * col) % 7) - 3);
}

int8_t weight1(int row, int col) {
  return static_cast<int8_t>(((5 * row + col) % 9) - 4);
}

int8_t act0(int wave, int row) {
  return static_cast<int8_t>(((3 * wave + 2 * row) % 17) - 8);
}

int8_t act1(int wave, int row) {
  return static_cast<int8_t>(((5 * wave + row) % 19) - 9);
}

uint32_t seed(int wave, int col) {
  return 0x10203040u + static_cast<uint32_t>(wave * 257 + col * 13);
}

uint32_t expected_sum(int wave, int col) {
  uint32_t sum = seed(wave, col);
  for (int row = 0; row < ROWS; ++row) {
    const int32_t term = static_cast<int32_t>(act0(wave, row)) *
                             weight0(row, col) +
                         static_cast<int32_t>(act1(wave, row)) *
                             weight1(row, col);
    sum += static_cast<uint32_t>(term);
  }
  return sum;
}

void clear_inputs(Vmatrix_int8_pe_array_probe* top) {
  top->mode_dc_req_in = 0;
  top->weight_load_we_in = 0;
  top->weight_load_bank_in = 0;
  top->weight_load_row_in = 0;
  top->weight_commit_in = 0;
  top->weight_commit_bank_in = 0;
  std::memset(top->weight0_row_in, 0, sizeof(top->weight0_row_in));
  std::memset(top->weight1_row_in, 0, sizeof(top->weight1_row_in));
  std::memset(top->pf_act_valid_west_in, 0,
              sizeof(top->pf_act_valid_west_in));
  std::memset(top->pf_act0_west_in, 0, sizeof(top->pf_act0_west_in));
  std::memset(top->pf_act1_west_in, 0, sizeof(top->pf_act1_west_in));
  std::memset(top->pf_psum_valid_north_in, 0,
              sizeof(top->pf_psum_valid_north_in));
  std::memset(top->pf_psum_north_in, 0, sizeof(top->pf_psum_north_in));
  std::memset(top->dc_act_valid_batch_in, 0,
              sizeof(top->dc_act_valid_batch_in));
  std::memset(top->dc_act0_batch_in, 0, sizeof(top->dc_act0_batch_in));
  std::memset(top->dc_act1_batch_in, 0, sizeof(top->dc_act1_batch_in));
  std::memset(top->dc_psum_valid_north_in, 0,
              sizeof(top->dc_psum_valid_north_in));
  std::memset(top->dc_psum_north_in, 0, sizeof(top->dc_psum_north_in));
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* top = new Vmatrix_int8_pe_array_probe;
  top->clk = 0;
  top->rst_n = 0;
  clear_inputs(top);
  tick(top);
  tick(top);
  top->rst_n = 1;

  // One 256-byte row per cycle loads all 32 KiB of a bank in 128 cycles.
  for (int row = 0; row < ROWS; ++row) {
    clear_inputs(top);
    top->weight_load_we_in = 1;
    top->weight_load_bank_in = 0;
    top->weight_load_row_in = row;
    for (int col = 0; col < COLS; ++col) {
      put_byte(top->weight0_row_in, col,
               static_cast<uint8_t>(weight0(row, col)));
      put_byte(top->weight1_row_in, col,
               static_cast<uint8_t>(weight1(row, col)));
    }
    tick(top);
  }
  clear_inputs(top);
  tick(top);  // forwards the final registered row write into u_array
  tick(top);

  int failures = 0;
  int checks = 0;
  int max_active_pes = 0;
  int wave0_last_output_cycle = -1;
  const int max_cycle = WAVES + ROWS + COLS + 4;

  for (int cycle = 0; cycle < max_cycle; ++cycle) {
    clear_inputs(top);
    for (int row = 0; row < ROWS; ++row) {
      const int wave = cycle - row;
      if (wave >= 0 && wave < WAVES) {
        put_bit(top->pf_act_valid_west_in, row);
        put_byte(top->pf_act0_west_in, row,
                 static_cast<uint8_t>(act0(wave, row)));
        put_byte(top->pf_act1_west_in, row,
                 static_cast<uint8_t>(act1(wave, row)));
      }
    }
    for (int col = 0; col < COLS; ++col) {
      const int wave = cycle - col;
      if (wave >= 0 && wave < WAVES) {
        put_bit(top->pf_psum_valid_north_in, col);
        top->pf_psum_north_in[col] = seed(wave, col);
      }
    }
    tick(top);

    int active_pes = 0;
    for (int row = 0; row < ROWS; ++row) {
      for (int col = 0; col < COLS; ++col) {
        const int wave = cycle - 1 - row - col;
        if (wave >= 0 && wave < WAVES)
          ++active_pes;
      }
    }
    max_active_pes = std::max(max_active_pes, active_pes);

    for (int row = 0; row < ROWS; ++row) {
      const int wave = cycle - row - COLS;
      const bool expected_valid = wave >= 0 && wave < WAVES;
      const bool got_valid = get_bit(top->act_valid_east_out, row);
      ++checks;
      if (got_valid != expected_valid) {
        std::printf("FAIL full east valid cycle=%d row=%d got=%d exp=%d\n",
                    cycle, row, got_valid, expected_valid);
        ++failures;
      }
      if (expected_valid) {
        ++checks;
        if (get_byte(top->act0_east_out, row) !=
                static_cast<uint8_t>(act0(wave, row)) ||
            get_byte(top->act1_east_out, row) !=
                static_cast<uint8_t>(act1(wave, row))) {
          std::printf("FAIL full east payload cycle=%d row=%d wave=%d\n",
                      cycle, row, wave);
          ++failures;
        }
      }
    }

    for (int col = 0; col < COLS; ++col) {
      const int wave = cycle - col - ROWS;
      const bool expected_valid = wave >= 0 && wave < WAVES;
      const bool got_valid = get_bit(top->psum_valid_south_out, col);
      ++checks;
      if (got_valid != expected_valid) {
        std::printf("FAIL full south valid cycle=%d col=%d got=%d exp=%d\n",
                    cycle, col, got_valid, expected_valid);
        ++failures;
      }
      if (expected_valid) {
        const uint32_t expected = expected_sum(wave, col);
        ++checks;
        if (top->psum_south_out[col] != expected) {
          std::printf(
              "FAIL full south cycle=%d col=%d wave=%d got=%08x exp=%08x\n",
              cycle, col, wave, top->psum_south_out[col], expected);
          ++failures;
        }
        if (wave == 0 && col == COLS - 1)
          wave0_last_output_cycle = cycle;
      }
    }
  }

  checks += 3;
  if (max_active_pes != ROWS * COLS) {
    std::printf("FAIL max active PEs=%d expected=%d\n",
                max_active_pes, ROWS * COLS);
    ++failures;
  }
  if (2 * max_active_pes != 32768) {
    std::printf("FAIL max active MACs=%d expected=32768\n", 2 * max_active_pes);
    ++failures;
  }
  if (wave0_last_output_cycle + 1 != 256) {
    std::printf("FAIL fill/drain slots=%d expected=256\n",
                wave0_last_output_cycle + 1);
    ++failures;
  }

  std::printf(
      "matrix_int8_pe_array_full_check: %s checks=%d fail_count=%d "
      "max_active_pes=%d max_macs_per_cycle=%d fill_drain_cycles=%d\n",
      failures == 0 ? "PASS" : "FAIL", checks, failures, max_active_pes,
      2 * max_active_pes, wave0_last_output_cycle + 1);
  top->final();
  delete top;
  return failures == 0 ? 0 : 1;
}
