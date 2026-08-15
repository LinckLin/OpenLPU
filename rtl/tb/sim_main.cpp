// co-sim testbench main — drives qcore_top, preloads program/memory, dumps
// trace + memory regions.  See docs/p7/rtl-report.md for the file formats.
#include "Vqcore_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

static Vqcore_top* top = nullptr;
static uint64_t sim_time = 0;

static std::vector<uint8_t> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

static void tick() {
  top->clk = 0; top->eval(); sim_time++;
  top->clk = 1; top->eval(); sim_time++;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  if (argc < 8) {
    fprintf(stderr, "usage: %s prog.bin preload.bin dump_req.bin trace.bin total.bin dump.bin [max_cycles]\n", argv[0]);
    return 2;
  }
  const std::string prog_path   = argv[1];
  const std::string preload_path= argv[2];
  const std::string dump_req    = argv[3];
  const std::string trace_path  = argv[4];
  const std::string total_path  = argv[5];
  const std::string dump_path   = argv[6];
  uint64_t max_cycles = (argc > 7) ? strtoull(argv[7], nullptr, 10) : 100000000ULL;

  top = new Vqcore_top;

  // -- reset ------------------------------------------------
  top->clk = 0; top->rst_n = 0; top->start = 0;
  top->imem_we = 0; top->imem_waddr = 0;
  for (int k = 0; k < 4; k++) top->imem_wdata[k] = 0;
  top->prog_len = 0;
  top->bd_en = 0; top->bd_sel = 0; top->bd_addr = 0; top->bd_wdata = 0;
  for (int i = 0; i < 4; i++) tick();          // reset cycles
  top->rst_n = 1; top->eval();

  // -- load program (128-bit LE instructions) --------------
  auto prog = read_file(prog_path);
  int ninst = (int)(prog.size() / 16);
  top->prog_len = (uint16_t)ninst;
  for (int i = 0; i < ninst; i++) {
    top->imem_we = 1;
    top->imem_waddr = (uint16_t)i;
    for (int k = 0; k < 4; k++) {
      const uint8_t* b = &prog[i * 16 + k * 4];
      top->imem_wdata[k] = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
                           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    }
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
  }
  top->imem_we = 0;

  // -- preload memory (records: sel[1] addr[8] nbytes[4] data[n]) ----
  {
    auto pre = read_file(preload_path);
    size_t off = 0;
    while (off + 13 <= pre.size()) {
      uint8_t sel = pre[off];
      uint64_t addr = 0;
      for (int b = 0; b < 8; b++) addr |= ((uint64_t)pre[off+1+b]) << (8*b);
      uint32_t n = (uint32_t)pre[off+9] | ((uint32_t)pre[off+10]<<8) |
                   ((uint32_t)pre[off+11]<<16) | ((uint32_t)pre[off+12]<<24);
      off += 13;
      for (uint32_t i = 0; i < n; i++) {
        top->bd_en = 1; top->bd_sel = sel; top->bd_addr = addr + i;
        top->bd_wdata = pre[off + i];
        top->clk = 0; top->eval();
        top->clk = 1; top->eval();
      }
      off += n;
    }
    // Deassert the backdoor write port before execution: leaving bd_en=1
    // would re-write the last preload byte into SRAM/HBM every cycle and
    // corrupt engine writes to that byte (golden3 attn_softmax head-15 last
    // element read 0x41E0 instead of 0x3EE0).
    top->bd_en = 0;
    top->eval();
  }

  // -- run -------------------------------------------------
  top->start = 1;
  top->clk = 0; top->eval();
  top->clk = 1; top->eval();
  top->start = 0;

  std::vector<uint8_t> trace_bytes;
  uint64_t ncycles = 0;
  bool finished = false;
  for (uint64_t c = 0; c < max_cycles && !finished; c++) {
    top->clk = 0; top->eval(); ncycles++;
    top->clk = 1; top->eval(); ncycles++;
    if (top->trace_valid) {
      uint16_t idx = (uint16_t)top->trace_index;
      uint32_t cyc = (uint32_t)top->trace_cycles;
      trace_bytes.push_back((uint8_t)(idx & 0xFF));
      trace_bytes.push_back((uint8_t)(idx >> 8));
      trace_bytes.push_back((uint8_t)(cyc & 0xFF));
      trace_bytes.push_back((uint8_t)((cyc>>8) & 0xFF));
      trace_bytes.push_back((uint8_t)((cyc>>16) & 0xFF));
      trace_bytes.push_back((uint8_t)((cyc>>24) & 0xFF));
    }
    if (top->done) finished = true;
  }
  if (!finished) { fprintf(stderr, "TIMEOUT after %llu cycles\n", (unsigned long long)max_cycles); return 3; }

  uint64_t total = top->total_cycles;

  // -- write trace (binary: 6-byte records) -----------------
  {
    std::ofstream f(trace_path, std::ios::binary);
    f.write((const char*)trace_bytes.data(), trace_bytes.size());
  }
  {
    std::ofstream f(total_path, std::ios::binary);
    f.write((const char*)&total, 8);
  }

  // -- dump requested regions (records: sel[1] addr[8] nbytes[4]) --
  {
    auto req = read_file(dump_req);
    std::ofstream out(dump_path, std::ios::binary);
    size_t off = 0;
    while (off + 13 <= req.size()) {
      uint8_t sel = req[off];
      uint64_t addr = 0;
      for (int b = 0; b < 8; b++) addr |= ((uint64_t)req[off+1+b]) << (8*b);
      uint32_t n = (uint32_t)req[off+9] | ((uint32_t)req[off+10]<<8) |
                   ((uint32_t)req[off+11]<<16) | ((uint32_t)req[off+12]<<24);
      off += 13;
      for (uint32_t i = 0; i < n; i++) {
        top->bd_en = 0; top->bd_sel = sel; top->bd_addr = addr + i;
        top->eval();
        uint8_t b = (uint8_t)top->bd_rdata;
        out.write((const char*)&b, 1);
      }
    }
  }

  top->final();
  delete top;
  return 0;
}
