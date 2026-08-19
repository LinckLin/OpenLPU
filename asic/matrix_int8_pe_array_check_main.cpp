// Parameterized 2x2-tile scoreboard for the full-array control contract.
// The shell sets TILE_ROWS/TILE_COLS=2 so this test compiles quickly while
// exercising the same 8x8-grid wiring, double-bank weights, and DC skew logic.
#include "Vmatrix_int8_pe_array.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

double sc_time_stamp() { return 0.0; }

namespace {

constexpr int ROWS = 32;
constexpr int COLS = 32;

void tick(Vmatrix_int8_pe_array* top) {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

void clear_wide_inputs(Vmatrix_int8_pe_array* top) {
  std::memset(top->pf_act0_west, 0, sizeof(top->pf_act0_west));
  std::memset(top->pf_act1_west, 0, sizeof(top->pf_act1_west));
  std::memset(top->pf_psum_north, 0, sizeof(top->pf_psum_north));
  std::memset(top->dc_act0_batch, 0, sizeof(top->dc_act0_batch));
  std::memset(top->dc_act1_batch, 0, sizeof(top->dc_act1_batch));
  std::memset(top->dc_psum_north, 0, sizeof(top->dc_psum_north));
  std::memset(top->weight0_row_in, 0, sizeof(top->weight0_row_in));
  std::memset(top->weight1_row_in, 0, sizeof(top->weight1_row_in));
  top->pf_act_valid_west = 0;
  top->pf_psum_valid_north = 0;
  top->dc_act_valid_batch = 0;
  top->dc_psum_valid_north = 0;
}

void clear_inputs(Vmatrix_int8_pe_array* top) {
  clear_wide_inputs(top);
  top->weight_load_we = 0;
  top->weight_load_bank = 0;
  top->weight_load_row = 0;
  top->weight_commit = 0;
  top->weight_commit_bank = 0;
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

void put_u32(WData* bus, int word_index, uint32_t value) {
  bus[word_index] = value;
}

uint32_t get_u32(const WData* bus, int word_index) {
  return bus[word_index];
}

int8_t as_i8(uint8_t value) {
  return static_cast<int8_t>(value);
}

uint32_t expected_sum(const int8_t (&a0)[ROWS], const int8_t (&a1)[ROWS],
                      const int8_t (&w0)[ROWS][COLS],
                      const int8_t (&w1)[ROWS][COLS], int col) {
  uint32_t sum = 0;
  for (int row = 0; row < ROWS; ++row) {
    const int32_t term = static_cast<int32_t>(a0[row]) * w0[row][col] +
                         static_cast<int32_t>(a1[row]) * w1[row][col];
    sum += static_cast<uint32_t>(term);
  }
  return sum;
}

void load_bank(Vmatrix_int8_pe_array* top, int bank,
               const int8_t (&w0)[ROWS][COLS],
               const int8_t (&w1)[ROWS][COLS]) {
  clear_wide_inputs(top);
  for (int row = 0; row < ROWS; ++row) {
    for (int col = 0; col < COLS; ++col) {
      put_byte(top->weight0_row_in, col, static_cast<uint8_t>(w0[row][col]));
      put_byte(top->weight1_row_in, col, static_cast<uint8_t>(w1[row][col]));
    }
    top->weight_load_we = 1;
    top->weight_load_bank = bank;
    top->weight_load_row = row;
    tick(top);
  }
  clear_inputs(top);
  tick(top);
}

void commit_bank(Vmatrix_int8_pe_array* top, int bank) {
  clear_inputs(top);
  top->weight_commit = 1;
  top->weight_commit_bank = bank;
  tick(top);
  clear_inputs(top);
  tick(top);
}

int check_pf_wave(Vmatrix_int8_pe_array* top, int8_t (&a0)[ROWS],
                  int8_t (&a1)[ROWS], const int8_t (&w0)[ROWS][COLS],
                  const int8_t (&w1)[ROWS][COLS], int* checks) {
  int failures = 0;
  const int max_cycle = ROWS + COLS + 4;
  for (int cycle = 0; cycle < max_cycle; ++cycle) {
    clear_wide_inputs(top);
    if (cycle < ROWS) {
      top->pf_act_valid_west |= uint32_t{1} << cycle;
      put_byte(top->pf_act0_west, cycle, static_cast<uint8_t>(a0[cycle]));
      put_byte(top->pf_act1_west, cycle, static_cast<uint8_t>(a1[cycle]));
    }
    if (cycle < COLS) {
      top->pf_psum_valid_north |= uint32_t{1} << cycle;
      put_u32(top->pf_psum_north, cycle, 0);
    }
    tick(top);

    for (int row = 0; row < ROWS; ++row) {
      const bool expected_valid = (cycle == row + COLS - 1);
      const bool got_valid = ((top->act_valid_east >> row) & 1u) != 0;
      ++*checks;
      if (got_valid != expected_valid) {
        std::printf("FAIL PF east valid cycle=%d row=%d got=%d exp=%d\n",
                    cycle, row, got_valid, expected_valid);
        ++failures;
      }
      if (expected_valid) {
        ++*checks;
        if (get_byte(top->act0_east, row) != static_cast<uint8_t>(a0[row]) ||
            get_byte(top->act1_east, row) != static_cast<uint8_t>(a1[row])) {
          std::printf("FAIL PF east payload cycle=%d row=%d\n", cycle, row);
          ++failures;
        }
      }
    }
    for (int col = 0; col < COLS; ++col) {
      const bool expected_valid = (cycle == col + ROWS - 1);
      const bool got_valid = ((top->psum_valid_south >> col) & 1u) != 0;
      ++*checks;
      if (got_valid != expected_valid) {
        std::printf("FAIL PF south valid cycle=%d col=%d got=%d exp=%d\n",
                    cycle, col, got_valid, expected_valid);
        ++failures;
      }
      if (expected_valid) {
        const uint32_t expected = expected_sum(a0, a1, w0, w1, col);
        ++*checks;
        if (get_u32(top->psum_south, col) != expected) {
          std::printf("FAIL PF south cycle=%d col=%d got=%08x exp=%08x\n",
                      cycle, col, get_u32(top->psum_south, col), expected);
          ++failures;
        }
      }
    }
  }
  return failures;
}

int check_dc_wave(Vmatrix_int8_pe_array* top, int8_t (&a0)[ROWS],
                  int8_t (&a1)[ROWS], const int8_t (&w0)[ROWS][COLS],
                  const int8_t (&w1)[ROWS][COLS], int* checks) {
  int failures = 0;
  const int max_cycle = ROWS + COLS + 6;
  for (int cycle = 0; cycle < max_cycle; ++cycle) {
    clear_wide_inputs(top);
    if (cycle == 0) {
      top->dc_act_valid_batch = ~uint32_t{0};
      for (int row = 0; row < ROWS; ++row) {
        put_byte(top->dc_act0_batch, row, static_cast<uint8_t>(a0[row]));
        put_byte(top->dc_act1_batch, row, static_cast<uint8_t>(a1[row]));
      }
    }
    if (cycle < COLS) {
      top->dc_psum_valid_north |= uint32_t{1} << cycle;
      put_u32(top->dc_psum_north, cycle, 0);
    }
    tick(top);

    for (int row = 0; row < ROWS; ++row) {
      const bool expected_valid = (cycle == row + COLS);
      const bool got_valid = ((top->act_valid_east >> row) & 1u) != 0;
      ++*checks;
      if (got_valid != expected_valid) {
        std::printf("FAIL DC east valid cycle=%d row=%d got=%d exp=%d\n",
                    cycle, row, got_valid, expected_valid);
        ++failures;
      }
      if (expected_valid) {
        ++*checks;
        if (get_byte(top->act0_east, row) != static_cast<uint8_t>(a0[row]) ||
            get_byte(top->act1_east, row) != static_cast<uint8_t>(a1[row])) {
          std::printf("FAIL DC east payload cycle=%d row=%d\n", cycle, row);
          ++failures;
        }
      }
    }
    for (int col = 0; col < COLS; ++col) {
      const bool expected_valid = (cycle == col + ROWS);
      const bool got_valid = ((top->psum_valid_south >> col) & 1u) != 0;
      ++*checks;
      if (got_valid != expected_valid) {
        std::printf("FAIL DC south valid cycle=%d col=%d got=%d exp=%d\n",
                    cycle, col, got_valid, expected_valid);
        ++failures;
      }
      if (expected_valid) {
        const uint32_t expected = expected_sum(a0, a1, w0, w1, col);
        ++*checks;
        if (get_u32(top->psum_south, col) != expected) {
          std::printf("FAIL DC south cycle=%d col=%d got=%08x exp=%08x\n",
                      cycle, col, get_u32(top->psum_south, col), expected);
          ++failures;
        }
      }
    }
  }
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* top = new Vmatrix_int8_pe_array;
  top->clk = 0;
  top->rst_n = 0;
  top->mode_dc_req = 0;
  clear_inputs(top);
  tick(top);
  tick(top);
  top->rst_n = 1;

  static int8_t w0_bank0[ROWS][COLS];
  static int8_t w1_bank0[ROWS][COLS];
  static int8_t w0_bank1[ROWS][COLS];
  static int8_t w1_bank1[ROWS][COLS];
  static int8_t a0[ROWS];
  static int8_t a1[ROWS];
  for (int row = 0; row < ROWS; ++row) {
    a0[row] = static_cast<int8_t>(row - 13);
    a1[row] = static_cast<int8_t>(7 - row);
    for (int col = 0; col < COLS; ++col) {
      w0_bank0[row][col] = static_cast<int8_t>(1 + ((row + col) & 3));
      w1_bank0[row][col] = static_cast<int8_t>(-2 + ((row + 2 * col) & 3));
      w0_bank1[row][col] = static_cast<int8_t>(2 + ((3 * row + col) & 5));
      w1_bank1[row][col] = static_cast<int8_t>(-3 + ((row + col) & 5));
    }
  }

  load_bank(top, 0, w0_bank0, w1_bank0);
  load_bank(top, 1, w0_bank1, w1_bank1);
  commit_bank(top, 1);

  int failures = 0;
  int checks = 0;
  failures += check_pf_wave(top, a0, a1, w0_bank1, w1_bank1, &checks);

  // Loading an inactive bank is legal during an active wave.  One complete
  // row is refreshed per cycle, so this 32-row instance reloads in 32 cycles.
  for (int cycle = 0; cycle < ROWS; ++cycle) {
    clear_wide_inputs(top);
    top->weight_load_we = 1;
    top->weight_load_bank = 0;
    top->weight_load_row = cycle;
    for (int col = 0; col < COLS; ++col) {
      put_byte(top->weight0_row_in, col,
               static_cast<uint8_t>(w0_bank0[cycle][col]));
      put_byte(top->weight1_row_in, col,
               static_cast<uint8_t>(w1_bank0[cycle][col]));
    }
    if (cycle < ROWS) {
      top->pf_act_valid_west |= uint32_t{1} << cycle;
      put_byte(top->pf_act0_west, cycle, static_cast<uint8_t>(a0[cycle]));
      put_byte(top->pf_act1_west, cycle, static_cast<uint8_t>(a1[cycle]));
      top->pf_psum_valid_north |= uint32_t{1} << cycle;
    }
    tick(top);
  }
  clear_inputs(top);
  tick(top);

  top->mode_dc_req = 1;
  clear_inputs(top);
  tick(top);
  int switch_cycles = 0;
  while (!top->mode_dc_active || top->mode_switch_busy) {
    clear_inputs(top);
    tick(top);
    if (++switch_cycles > 360) {
      std::printf("FAIL MODE switch did not complete\n");
      ++failures;
      break;
    }
  }
  if (switch_cycles != 300) {
    std::printf("FAIL MODE switch cycles=%d expected=300\n", switch_cycles);
    ++failures;
  }

  if (top->mode_dc_active && !top->mode_switch_busy)
    failures += check_dc_wave(top, a0, a1, w0_bank1, w1_bank1, &checks);

  std::printf(
      "matrix_int8_pe_array_check: %s checks=%d fail_count=%d "
      "mode_switch_cycles=%d\n",
      failures == 0 ? "PASS" : "FAIL", checks, failures, switch_cycles);
  top->final();
  delete top;
  return failures == 0 ? 0 : 1;
}
