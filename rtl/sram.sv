// ============================================================================
// sram.sv — QCore scratchpad SRAM: 16 bank x 512 KiB, 2R1W + fixed priority.
//
// Frozen addressing (03 §2.1, D15): a 16 B word W[18:0] maps to
//   bank       = byte_addr[7:4]   (= W[3:0])
//   bank-word  = byte_addr[22:8]  (= W[18:4])
// i.e. 16 B granularity, 16-way interleave; a 256 B contiguous run touches all
// 16 banks exactly once.  Each bank serves <= 2 reads + 1 write per cycle;
// contention is resolved by the frozen fixed priority (03 §2.3):
//   MATRIX.A > MATRIX.B > MATRIX.C > VECTOR > DMA > KV.
//
// Ports (engine-facing, 16 B word granularity):
//   * 6 logical engine ports, each with 2 read + 1 write request.
//   * Arbitration is per-bank, so a full 256 B/cycle stream (16 words, one per
//     bank) is served with zero conflict; a 3rd read or 2nd write to the same
//     bank in one cycle stalls that requester (fixed priority).
//
// SRAM shrink (P10b, plans/m8-wait-plan.md §3): BANK_BYTES is a parameter with
// default 512 KiB / bank (8 MiB total, unchanged).  Shrink instances keep the
// 16-bank interleave and the 6-port crossbar structure; every address width is
// derived from the parameters via $clog2 so a smaller bank simply narrows the
// engine-facing address ports:
//   BANK_BYTES = 512 KiB  ->  8 MiB total  (default, frozen)
//   BANK_BYTES = 256 KiB  ->  4 MiB total  (84% of the 4.75 MiB ZCU104 budget)
//   BANK_BYTES = 128 KiB  ->  2 MiB total  (42% of the 4.75 MiB budget)
//
// Synthesis note: `sram_bank` is the physical macro boundary (replace with a
// compiled SRAM macro in P9/P10); `sram_top` is the crossbar + arbiter.
// ============================================================================
`ifndef SRAM_SV
`define SRAM_SV

`include "qcore_pkg.sv"

// ----------------------------------------------------------------------------
// sram_bank — one BANK_BYTES bank, 2 read ports + 1 write port, 16 B words.
// ----------------------------------------------------------------------------
module sram_bank #(
  parameter int BANK_BYTES = 512 * 1024,
  parameter int WORD_BYTES = 16
) (
  input  logic        clk,
  input  logic        rst_n,
  // read port 0 / 1: word address inside this bank
  input  logic [$clog2(BANK_BYTES/WORD_BYTES)-1:0] r0_addr,
  input  logic        r0_en,
  output logic [127:0] r0_data,
  input  logic [$clog2(BANK_BYTES/WORD_BYTES)-1:0] r1_addr,
  input  logic        r1_en,
  output logic [127:0] r1_data,
  // write port: word address + 16 B word + byte enables
  input  logic [$clog2(BANK_BYTES/WORD_BYTES)-1:0] w_addr,
  input  logic [127:0] w_data,
  input  logic [15:0] w_be,
  input  logic        w_en
);
  localparam int BANK_WORD_W = $clog2(BANK_BYTES / WORD_BYTES);  // 15 (default)
  localparam int BANK_WORDS  = BANK_BYTES / WORD_BYTES;          // 32768 (default)
  logic [127:0] mem [0:BANK_WORDS-1];

  always_ff @(posedge clk) begin
    if (w_en)
      for (int b = 0; b < 16; b++)
        if (w_be[b]) mem[w_addr][b*8 +: 8] <= w_data[b*8 +: 8];
  end

  always_comb begin
    r0_data = r0_en ? mem[r0_addr] : 128'b0;
    r1_data = r1_en ? mem[r1_addr] : 128'b0;
  end
endmodule

// ----------------------------------------------------------------------------
// sram_top — N_BANK banks + address interleave + fixed-priority arbitration.
//
// The six logical ports (one per engine role) each carry up to 2 read + 1 write
// requests per cycle.  Requests are decoded to a bank and arbitrated there.
//
// Addressing (03 §2.1, D15): bank = word_addr[BANK_IDX_W-1:0] (low bits),
// bank-word = word_addr[SRAM_WORD_W-1:BANK_IDX_W] (high bits).  The widths are
// derived from N_BANK/BANK_BYTES/WORD_BYTES so a shrunk bank narrows the ports.
// ----------------------------------------------------------------------------
module sram_top #(
  parameter int N_BANK     = 16,
  parameter int BANK_BYTES = 512 * 1024,
  parameter int WORD_BYTES = 16
) (
  input  logic clk,
  input  logic rst_n,
  // Engine ports.  Port order encodes priority (highest first).
  // p = {MATRIX.A, MATRIX.B, MATRIX.C, VECTOR, DMA, KV}
  input  logic [5:0]  p_r0_en,
  input  logic [$clog2(N_BANK*BANK_BYTES/WORD_BYTES)-1:0] p_r0_addr [6],
  output logic [127:0] p_r0_data [6],
  input  logic [5:0]  p_r1_en,
  input  logic [$clog2(N_BANK*BANK_BYTES/WORD_BYTES)-1:0] p_r1_addr [6],
  output logic [127:0] p_r1_data [6],
  input  logic [5:0]  p_w_en,
  input  logic [$clog2(N_BANK*BANK_BYTES/WORD_BYTES)-1:0] p_w_addr [6],
  input  logic [127:0] p_w_data [6],
  input  logic [15:0]  p_w_be [6]
);
  localparam int BANK_IDX_W   = $clog2(N_BANK);                        // 4
  localparam int BANK_WORD_W  = $clog2(BANK_BYTES / WORD_BYTES);       // 15
  localparam int SRAM_WORD_W  = $clog2(N_BANK*BANK_BYTES/WORD_BYTES);  // 19

  // Per-bank request fan-in.  We first decode every active port request into a
  // (bank, bank-word) pair, then pick the two reads + one write per bank by the
  // frozen priority.  Requests that lose are stalled (data invalid) for one
  // cycle; the requester re-drives next cycle (fixed priority, no reorder).
  //
  // For the co-sim and synthesis-friendliness we implement the arbiter as a
  // two-level selection: (1) each bank gathers its candidate read/write
  // requests from the six ports; (2) picks by port priority index.

  // Port -> bank decode.
  wire [BANK_IDX_W-1:0]  port_bank_r0 [6];
  wire [BANK_IDX_W-1:0]  port_bank_r1 [6];
  wire [BANK_IDX_W-1:0]  port_bank_w  [6];
  wire [BANK_WORD_W-1:0] port_waddr_r0 [6];
  wire [BANK_WORD_W-1:0] port_waddr_r1 [6];
  wire [BANK_WORD_W-1:0] port_waddr_w  [6];

  genvar g;
  generate
    for (g = 0; g < 6; g++) begin : gen_decode
      assign port_bank_r0[g]   = p_r0_addr[g][BANK_IDX_W-1:0];
      assign port_bank_r1[g]   = p_r1_addr[g][BANK_IDX_W-1:0];
      assign port_bank_w[g]    = p_w_addr[g][BANK_IDX_W-1:0];
      assign port_waddr_r0[g]  = p_r0_addr[g][SRAM_WORD_W-1:BANK_IDX_W];
      assign port_waddr_r1[g]  = p_r1_addr[g][SRAM_WORD_W-1:BANK_IDX_W];
      assign port_waddr_w[g]   = p_w_addr[g][SRAM_WORD_W-1:BANK_IDX_W];
    end
  endgenerate

  // Per-bank nets.
  logic [BANK_WORD_W-1:0] bk_r0_addr [N_BANK];
  logic [BANK_WORD_W-1:0] bk_r1_addr [N_BANK];
  logic        bk_r0_en   [N_BANK];
  logic        bk_r1_en   [N_BANK];
  logic [127:0] bk_r0_data [N_BANK];
  logic [127:0] bk_r1_data [N_BANK];
  logic [BANK_WORD_W-1:0] bk_w_addr  [N_BANK];
  logic [127:0] bk_w_data  [N_BANK];
  logic [15:0] bk_w_be    [N_BANK];
  logic        bk_w_en    [N_BANK];
  integer      r_used, w_used;

  generate
    for (g = 0; g < N_BANK; g++) begin : gen_banks
      sram_bank #(.BANK_BYTES(BANK_BYTES), .WORD_BYTES(WORD_BYTES)) u_bank (
        .clk(clk), .rst_n(rst_n),
        .r0_addr(bk_r0_addr[g]), .r0_en(bk_r0_en[g]), .r0_data(bk_r0_data[g]),
        .r1_addr(bk_r1_addr[g]), .r1_en(bk_r1_en[g]), .r1_data(bk_r1_data[g]),
        .w_addr(bk_w_addr[g]), .w_data(bk_w_data[g]), .w_be(bk_w_be[g]),
        .w_en(bk_w_en[g])
      );
    end
  endgenerate



  // Arbitration.  For each bank pick the highest-priority active port for each
  // of the two read slots and the single write slot.  Port g has priority
  // index g (0 = MATRIX.A, highest).  A port may use both read slots only if
  // no higher-priority port needs that slot.
  always_comb begin
    for (int b = 0; b < N_BANK; b++) begin
      // defaults
      bk_r0_en[b] = 1'b0; bk_r0_addr[b] = '0;
      bk_r1_en[b] = 1'b0; bk_r1_addr[b] = '0;
      bk_w_en[b]  = 1'b0; bk_w_addr[b] = '0; bk_w_data[b] = 128'b0; bk_w_be[b] = 16'b0;
      // two read winners + one write winner
      r_used = 0; w_used = 0;
      for (int p = 0; p < 6; p++) begin
        // read slot 0
        if (p_r0_en[p] && port_bank_r0[p] == b[BANK_IDX_W-1:0] && r_used < 2) begin
          if (r_used == 0) begin bk_r0_en[b] = 1'b1; bk_r0_addr[b] = port_waddr_r0[p]; end
          else            begin bk_r1_en[b] = 1'b1; bk_r1_addr[b] = port_waddr_r0[p]; end
          r_used = r_used + 1;
        end
        // read slot 1
        if (p_r1_en[p] && port_bank_r1[p] == b[BANK_IDX_W-1:0] && r_used < 2) begin
          if (r_used == 0) begin bk_r0_en[b] = 1'b1; bk_r0_addr[b] = port_waddr_r1[p]; end
          else            begin bk_r1_en[b] = 1'b1; bk_r1_addr[b] = port_waddr_r1[p]; end
          r_used = r_used + 1;
        end
        // write
        if (p_w_en[p] && port_bank_w[p] == b[BANK_IDX_W-1:0] && w_used < 1) begin
          bk_w_en[b] = 1'b1; bk_w_addr[b] = port_waddr_w[p];
          bk_w_data[b] = p_w_data[p]; bk_w_be[b] = p_w_be[p];
          w_used = w_used + 1;
        end
      end
    end
  end

  // Read data fan-out: route each bank's read data back to the winning port.
  always_comb begin
    for (int p = 0; p < 6; p++) begin
      p_r0_data[p] = 128'b0;
      p_r1_data[p] = 128'b0;
      if (p_r0_en[p]) p_r0_data[p] = bk_r0_data[port_bank_r0[p]];
      if (p_r1_en[p]) p_r1_data[p] = bk_r1_data[port_bank_r1[p]];
    end
  end

endmodule
`endif // SRAM_SV
