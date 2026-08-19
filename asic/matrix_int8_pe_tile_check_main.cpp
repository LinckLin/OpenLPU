// Wavefront scoreboard for the registered-boundary 16x16 PE tile.
#include "Vmatrix_int8_pe_tile_probe.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>

double sc_time_stamp() { return 0.0; }

namespace {

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kWaves = 24;
constexpr int kActWords = 4;
constexpr int kPsumWords = 16;

struct Wave {
  int launch;
  int8_t act0[kRows];
  int8_t act1[kRows];
  uint32_t psum[kCols];
};

void tick(Vmatrix_int8_pe_tile_probe* top) {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

uint32_t next_random(uint32_t* state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

void clear_words(WData* words, int count) {
  for (int i = 0; i < count; ++i) words[i] = 0;
}

void put_u8(WData* words, int index, uint8_t value) {
  const int bit = index * 8;
  const int word = bit / 32;
  const int shift = bit % 32;
  const uint32_t mask = 0xffu << shift;
  words[word] = (words[word] & ~mask) | (uint32_t(value) << shift);
}

uint8_t get_u8(const WData* words, int index) {
  const int bit = index * 8;
  return static_cast<uint8_t>(words[bit / 32] >> (bit % 32));
}

void put_u32(WData* words, int index, uint32_t value) {
  words[index] = value;
}

uint32_t get_u32(const WData* words, int index) {
  return words[index];
}

int find_wave_at(const Wave* waves, int cycle, int offset) {
  for (int n = 0; n < kWaves; ++n) {
    if (waves[n].launch + offset == cycle) return n;
  }
  return -1;
}

int32_t signed_word(uint32_t value) {
  return (value & 0x80000000u)
      ? static_cast<int32_t>(value - 0x100000000ull)
      : static_cast<int32_t>(value);
}

uint32_t expected_psum(const Wave& wave, const int8_t w0[kRows][kCols],
                       const int8_t w1[kRows][kCols], int col) {
  int64_t sum = signed_word(wave.psum[col]);
  for (int row = 0; row < kRows; ++row) {
    sum += static_cast<int32_t>(wave.act0[row]) * w0[row][col];
    sum += static_cast<int32_t>(wave.act1[row]) * w1[row][col];
  }
  return static_cast<uint32_t>(sum);
}

void clear_inputs(Vmatrix_int8_pe_tile_probe* top) {
  top->weight_we_in = 0;
  top->weight_row_in = 0;
  top->weight_col_in = 0;
  top->weight0_in = 0;
  top->weight1_in = 0;
  top->act_valid_west_in = 0;
  top->psum_valid_north_in = 0;
  clear_words(top->act0_west_in, kActWords);
  clear_words(top->act1_west_in, kActWords);
  clear_words(top->psum_north_in, kPsumWords);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* top = new Vmatrix_int8_pe_tile_probe;
  top->clk = 0;
  top->rst_n = 0;
  clear_inputs(top);
  tick(top);
  tick(top);
  top->rst_n = 1;

  int failures = 0;
  int checks = 0;
  uint32_t state = 0x54494c45u;
  int8_t weights0[kRows][kCols];
  int8_t weights1[kRows][kCols];

  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      weights0[row][col] = static_cast<int8_t>(next_random(&state));
      weights1[row][col] = static_cast<int8_t>(next_random(&state));
    }
  }
  weights0[0][0] = -128;
  weights1[0][0] = 127;
  weights0[15][15] = 127;
  weights1[15][15] = -128;

  // Address every PE once.  The probe adds one launch cycle, so leave one
  // all-idle edge after the final request for the last write to enter u_tile.
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      clear_inputs(top);
      top->weight_we_in = 1;
      top->weight_row_in = static_cast<uint8_t>(row);
      top->weight_col_in = static_cast<uint8_t>(col);
      top->weight0_in = static_cast<uint8_t>(weights0[row][col]);
      top->weight1_in = static_cast<uint8_t>(weights1[row][col]);
      tick(top);
    }
  }
  clear_inputs(top);
  tick(top);

  Wave waves[kWaves];
  for (int n = 0; n < kWaves; ++n) {
    // Two empty launch slots after wave 7 exercise a real bubble in both
    // boundary streams while preserving row/column skew.
    waves[n].launch = n + (n >= 8 ? 2 : 0);
    for (int row = 0; row < kRows; ++row) {
      waves[n].act0[row] = static_cast<int8_t>(next_random(&state));
      waves[n].act1[row] = static_cast<int8_t>(next_random(&state));
    }
    for (int col = 0; col < kCols; ++col)
      waves[n].psum[col] = next_random(&state);
  }

  const int last_cycle = waves[kWaves - 1].launch + kRows + kCols + 3;
  for (int cycle = 0; cycle <= last_cycle; ++cycle) {
    clear_inputs(top);

    // West activation launch for row r occurs at launch+r.  North partial
    // sum launch for column c occurs at launch+c.  Thus both arrive at PE(r,c)
    // on the same internal edge: launch+r+c.
    for (int row = 0; row < kRows; ++row) {
      const int wave = find_wave_at(waves, cycle, row);
      if (wave >= 0) {
        top->act_valid_west_in |= static_cast<uint16_t>(1u << row);
        put_u8(top->act0_west_in, row, static_cast<uint8_t>(waves[wave].act0[row]));
        put_u8(top->act1_west_in, row, static_cast<uint8_t>(waves[wave].act1[row]));
      }
    }
    for (int col = 0; col < kCols; ++col) {
      const int wave = find_wave_at(waves, cycle, col);
      if (wave >= 0) {
        top->psum_valid_north_in |= static_cast<uint16_t>(1u << col);
        put_u32(top->psum_north_in, col, waves[wave].psum[col]);
      }
    }

    tick(top);

    // The registered probe delays the boundary launch by one edge.  The
    // bottom output therefore appears at launch+ROWS+col, and the east output
    // at launch+row+COLS.
    for (int col = 0; col < kCols; ++col) {
      const int wave = find_wave_at(waves, cycle, kRows + col);
      const bool expected_valid = wave >= 0;
      const bool actual_valid = ((top->psum_valid_south_out >> col) & 1u) != 0;
      ++checks;
      if (actual_valid != expected_valid) {
        std::printf("FAIL south cycle=%d col=%d valid=%d expected=%d\n",
                    cycle, col, actual_valid, expected_valid);
        ++failures;
      } else if (expected_valid) {
        const uint32_t got = get_u32(top->psum_south_out, col);
        const uint32_t expected = expected_psum(waves[wave], weights0,
                                                weights1, col);
        ++checks;
        if (got != expected) {
          std::printf("FAIL south cycle=%d col=%d wave=%d got=%08x exp=%08x\n",
                      cycle, col, wave, got, expected);
          ++failures;
        }
      }
    }
    for (int row = 0; row < kRows; ++row) {
      const int wave = find_wave_at(waves, cycle, row + kCols);
      const bool expected_valid = wave >= 0;
      const bool actual_valid = ((top->act_valid_east_out >> row) & 1u) != 0;
      ++checks;
      if (actual_valid != expected_valid) {
        std::printf("FAIL east cycle=%d row=%d valid=%d expected=%d\n",
                    cycle, row, actual_valid, expected_valid);
        ++failures;
      } else if (expected_valid) {
        const uint8_t got0 = get_u8(top->act0_east_out, row);
        const uint8_t got1 = get_u8(top->act1_east_out, row);
        const uint8_t expected0 = static_cast<uint8_t>(waves[wave].act0[row]);
        const uint8_t expected1 = static_cast<uint8_t>(waves[wave].act1[row]);
        ++checks;
        if (got0 != expected0 || got1 != expected1) {
          std::printf("FAIL east cycle=%d row=%d wave=%d got=(%02x,%02x) exp=(%02x,%02x)\n",
                      cycle, row, wave, got0, got1, expected0, expected1);
          ++failures;
        }
      }
    }
  }

  std::printf("matrix_int8_pe_tile_check: %s checks=%d fail_count=%d\n",
              failures == 0 ? "PASS" : "FAIL", checks, failures);
  top->final();
  delete top;
  return failures == 0 ? 0 : 1;
}
