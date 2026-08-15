// FPGA integration-smoke testbench — drives qcore_fpga_top entirely through
// the host_if register interface (config / cmd-queue load / qbin load /
// logits readback), and samples the observability trace pins (mirroring
// rtl/tb/sim_main.cpp) so the per-instruction cycle trace can be compared
// against the qsim baseline with the M6 co-sim criterion.
//
// File formats are identical to rtl/tb/sim_main.cpp (see docs/p7/rtl-report.md):
//   prog.bin     : 128-bit LE instructions (16 B each)
//   preload.bin  : records sel[1] addr[8] nbytes[4] data[n]
//   dump_req.bin : records sel[1] addr[8] nbytes[4]
//   trace.bin    : 6-byte records (index:u16, cycles:u32)
//   total.bin    : u64 total_cycles
//   dump.bin     : concatenated requested bytes
#include "Vqcore_fpga_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

static Vqcore_fpga_top* top = nullptr;

// host_if register word addresses (see fpga/host_if.sv header)
enum : uint32_t {
  R_CTRL        = 0x000, R_STATUS  = 0x001,
  R_TOTAL_LO    = 0x002, R_TOTAL_HI= 0x003,
  R_TRACE       = 0x004, R_TRACE_CYC = 0x005,
  R_PROG_LEN    = 0x006, R_CMDQ_ADDR = 0x007,
  R_CMDQ_W0     = 0x008, R_CMDQ_W1   = 0x009,
  R_CMDQ_W2     = 0x00A, R_CMDQ_W3   = 0x00B,
  R_CMDQ_GO     = 0x00C,
  R_MEM_ADDR_LO = 0x010, R_MEM_ADDR_HI = 0x011,
  R_MEM_SEL     = 0x012, R_MEM_WDATA  = 0x013,
  R_MEM_RDATA   = 0x014, R_MEM_ADV    = 0x015,
};

static std::vector<uint8_t> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

static uint64_t rd_u64_le(const uint8_t* p) {
  uint64_t v = 0;
  for (int b = 0; b < 8; b++) v |= ((uint64_t)p[b]) << (8 * b);
  return v;
}
static uint32_t rd_u32_le(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void host_write(uint32_t addr, uint32_t data) {
  top->host_addr = addr;
  top->host_wdata = data;
  top->host_wen = 1;
  top->clk_i = 0; top->eval();
  top->clk_i = 1; top->eval();      // posedge: host_if samples
  top->host_wen = 0;
}

static uint32_t host_read(uint32_t addr) {
  top->host_addr = addr;
  top->host_ren = 1;
  top->eval();                    // combinational read
  uint32_t v = top->host_rdata;
  top->host_ren = 0;
  return v;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  if (argc < 8) {
    fprintf(stderr, "usage: %s prog.bin preload.bin dump_req.bin trace.bin total.bin dump.bin [max_cycles]\n", argv[0]);
    return 2;
  }
  const std::string prog_path    = argv[1];
  const std::string preload_path = argv[2];
  const std::string dump_req     = argv[3];
  const std::string trace_path   = argv[4];
  const std::string total_path   = argv[5];
  const std::string dump_path    = argv[6];
  uint64_t max_cycles = (argc > 7) ? strtoull(argv[7], nullptr, 10) : 100000000ULL;

  top = new Vqcore_fpga_top;

  // -- reset (async, active-low, sync release via clock_reset) ------------
  top->clk_i = 0; top->async_rst_n_i = 0;
  top->host_addr = 0; top->host_wdata = 0; top->host_wen = 0; top->host_ren = 0;
  // AXI4 slave inputs: no DDR controller attached in co-sim
  top->m_axi_arready = 0; top->m_axi_awready = 0; top->m_axi_wready = 0;
  top->m_axi_rvalid = 0; top->m_axi_rlast = 0; top->m_axi_rdata = 0;
  top->m_axi_rresp = 0; top->m_axi_bvalid = 0; top->m_axi_bresp = 0;

  for (int i = 0; i < 4; i++) { top->clk_i = 0; top->eval(); top->clk_i = 1; top->eval(); }
  top->async_rst_n_i = 1;
  for (int i = 0; i < 4; i++) { top->clk_i = 0; top->eval(); top->clk_i = 1; top->eval(); } // sync release

  // -- load program via cmd queue (128-bit LE instructions) ---------------
  auto prog = read_file(prog_path);
  int ninst = (int)(prog.size() / 16);
  host_write(R_PROG_LEN, (uint32_t)ninst);
  host_write(R_CMDQ_ADDR, 0);
  for (int i = 0; i < ninst; i++) {
    const uint8_t* b = &prog[i * 16];
    host_write(R_CMDQ_W0, rd_u32_le(b + 0));
    host_write(R_CMDQ_W1, rd_u32_le(b + 4));
    host_write(R_CMDQ_W2, rd_u32_le(b + 8));
    host_write(R_CMDQ_W3, rd_u32_le(b + 12));
    host_write(R_CMDQ_GO, 0);
  }

  // -- preload memory via qbin-load byte stream ---------------------------
  {
    auto pre = read_file(preload_path);
    size_t off = 0;
    while (off + 13 <= pre.size()) {
      uint8_t sel = pre[off];
      uint64_t addr = rd_u64_le(&pre[off + 1]);
      uint32_t n = rd_u32_le(&pre[off + 9]);
      off += 13;
      host_write(R_MEM_SEL, sel);
      host_write(R_MEM_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFFu));
      host_write(R_MEM_ADDR_HI, (uint32_t)(addr >> 32));
      for (uint32_t i = 0; i < n; i++) {
        host_write(R_MEM_WDATA, (1u << 8) | pre[off + i]);  // commit byte
      }
      off += n;
    }
  }

  // -- run ---------------------------------------------------------------
  host_write(R_CTRL, 0x1);       // start pulse

  std::vector<uint8_t> trace_bytes;
  uint64_t ncycles = 0;
  bool finished = false;
  for (uint64_t c = 0; c < max_cycles && !finished; c++) {
    top->clk_i = 0; top->eval(); ncycles++;
    top->clk_i = 1; top->eval(); ncycles++;
    if (top->trace_valid) {
      uint16_t idx = (uint16_t)top->trace_index;
      uint32_t cyc = (uint32_t)top->trace_cycles;
      trace_bytes.push_back((uint8_t)(idx & 0xFF));
      trace_bytes.push_back((uint8_t)(idx >> 8));
      trace_bytes.push_back((uint8_t)(cyc & 0xFF));
      trace_bytes.push_back((uint8_t)((cyc >> 8) & 0xFF));
      trace_bytes.push_back((uint8_t)((cyc >> 16) & 0xFF));
      trace_bytes.push_back((uint8_t)((cyc >> 24) & 0xFF));
    }
    if (top->done) finished = true;
  }
  if (!finished) { fprintf(stderr, "TIMEOUT after %llu cycles\n", (unsigned long long)max_cycles); return 3; }

  // total_cycles read back through host_if (logits-readback status path)
  uint64_t total = (uint64_t)host_read(R_TOTAL_LO) | ((uint64_t)host_read(R_TOTAL_HI) << 32);
  if (total != (uint64_t)top->total_cycles) {
    fprintf(stderr, "host_if TOTAL mismatch: reg=%llu pin=%llu\n",
            (unsigned long long)total, (unsigned long long)top->total_cycles);
    return 4;
  }
  // STATUS.done must also be set through the register path
  if (!(host_read(R_STATUS) & 0x1)) {
    fprintf(stderr, "host_if STATUS.done not set\n");
    return 5;
  }

  // -- write trace (6-byte records) --------------------------------------
  {
    std::ofstream f(trace_path, std::ios::binary);
    f.write((const char*)trace_bytes.data(), trace_bytes.size());
  }
  {
    std::ofstream f(total_path, std::ios::binary);
    f.write((const char*)&total, 8);
  }

  // -- dump requested regions via logits-readback byte stream ------------
  {
    auto req = read_file(dump_req);
    std::ofstream out(dump_path, std::ios::binary);
    size_t off = 0;
    while (off + 13 <= req.size()) {
      uint8_t sel = req[off];
      uint64_t addr = rd_u64_le(&req[off + 1]);
      uint32_t n = rd_u32_le(&req[off + 9]);
      off += 13;
      host_write(R_MEM_SEL, sel);
      host_write(R_MEM_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFFu));
      host_write(R_MEM_ADDR_HI, (uint32_t)(addr >> 32));
      for (uint32_t i = 0; i < n; i++) {
        uint32_t v = host_read(R_MEM_RDATA);
        uint8_t b = (uint8_t)(v & 0xFF);
        out.write((const char*)&b, 1);
        host_write(R_MEM_ADV, 0);   // advance
      }
    }
  }

  top->final();
  delete top;
  return 0;
}
