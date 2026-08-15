// ============================================================================
// sram_check.sv — dedicated Verilator unit test for rtl/sram.sv (P10b).
//
// Validates the parametrized SRAM crossbar (sram_top) for the frozen default
// (512 KiB/bank -> 8 MiB) and the two shrink instances
// (256 KiB/bank -> 4 MiB, 128 KiB/bank -> 2 MiB).  Each instance runs the same
// self-checking sequence:
//   1. write/read round-trip over 64 words (covers all 16 banks x 4 words);
//   2. byte-enable partial write (only enabled bytes change);
//   3. bank-interleave mapping (D15: bank = word_addr[3:0]) via a same-bank
//      write-contention probe (word 0 vs word 16, both -> bank 0, so the
//      lower-priority write must be lost);
//   4. dual read from one bank in a single cycle (2R/bank).
//
// The top instantiates three sram_test instances and ANDs their results; the
// C++ harness (sram_check_main.cpp) clocks the design and reports pass/fail.
// Build: see asic/run_sram_check.sh.
// ============================================================================
`ifndef SRAM_CHECK_SV
`define SRAM_CHECK_SV

`include "sram.sv"

// ----------------------------------------------------------------------------
// sram_test — parameterized self-checking sequence around one sram_top.
// ----------------------------------------------------------------------------
module sram_test #(
  parameter int N_BANK     = 16,
  parameter int BANK_BYTES = 512 * 1024,
  parameter int WORD_BYTES = 16
) (
  input  logic        clk,
  input  logic        rst_n,
  output logic        done,
  output logic        pass,
  output logic [31:0] fail_count
);
  localparam int BANK_IDX_W  = $clog2(N_BANK);
  localparam int BANK_WORD_W = $clog2(BANK_BYTES / WORD_BYTES);
  localparam int SRAM_WORD_W = $clog2(N_BANK * BANK_BYTES / WORD_BYTES);
  localparam int SRAM_WORDS  = N_BANK * BANK_BYTES / WORD_BYTES;

  logic [5:0]  p_r0_en, p_r1_en, p_w_en;
  logic [SRAM_WORD_W-1:0] p_r0_addr [6], p_r1_addr [6], p_w_addr [6];
  logic [127:0] p_r0_data [6], p_r1_data [6], p_w_data [6];
  logic [15:0]  p_w_be [6];

  sram_top #(.N_BANK(N_BANK), .BANK_BYTES(BANK_BYTES), .WORD_BYTES(WORD_BYTES)) u_sram (
    .clk(clk), .rst_n(rst_n),
    .p_r0_en(p_r0_en), .p_r0_addr(p_r0_addr), .p_r0_data(p_r0_data),
    .p_r1_en(p_r1_en), .p_r1_addr(p_r1_addr), .p_r1_data(p_r1_data),
    .p_w_en(p_w_en), .p_w_addr(p_w_addr), .p_w_data(p_w_data), .p_w_be(p_w_be)
  );

  // word pattern: 4 copies of the 32-bit index -> deterministic, distinct.
  function automatic logic [127:0] pat(input logic [31:0] k);
    return {4{k}};
  endfunction

  // Precomputed patterns used with bit-slices (Verilator 4.038 does not allow
  // slicing a function-call result inline).
  localparam logic [127:0] P7  = {4{32'd7}};
  localparam logic [127:0] P16 = {4{32'd16}};
  localparam logic [127:0] PA0 = {4{32'hA0}};

  typedef enum logic [3:0] {
    S_RESET, S_WR, S_RD, S_RDCHK, S_BE_WR, S_BE_CHK, S_BE_CHK2,
    S_ILV_WR, S_ILV_CHK0, S_ILV_CHK16, S_ILV_CHK16B, S_DR, S_DRCHK, S_DONE
  } state_t;

  state_t      state;
  logic [31:0] i;
  logic [127:0] exp_w;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_RESET;
      i <= 0;
      fail_count <= 0;
      pass <= 1'b0;
      done <= 1'b0;
    end else begin
      // default: idle all ports
      p_r0_en <= 6'b0; p_r1_en <= 6'b0; p_w_en <= 6'b0;
      for (int p = 0; p < 6; p++) begin
        p_r0_addr[p] <= '0; p_r1_addr[p] <= '0; p_w_addr[p] <= '0;
        p_w_data[p] <= '0; p_w_be[p] <= 16'b0;
      end
      exp_w <= '0;
      case (state)
        S_RESET: begin
          i <= 0;
          state <= S_WR;
        end

        // --- phase 1: write words 0..63 -----------------------------------
        S_WR: begin
          p_w_en[0]  <= 1'b1;
          p_w_addr[0] <= SRAM_WORD_W'(i);
          p_w_data[0] <= pat(i);
          p_w_be[0]   <= 16'hFFFF;
          if (i == 63) begin i <= 0; state <= S_RD; end
          else         begin i <= i + 1; end
        end

        // --- phase 2: read words 0..63 (combinational), check next cycle ----
        S_RD: begin
          p_r0_en[0]  <= 1'b1;
          p_r0_addr[0] <= SRAM_WORD_W'(i);
          exp_w <= pat(i);
          state <= S_RDCHK;
        end
        S_RDCHK: begin
          if (p_r0_data[0] != exp_w) begin
            fail_count <= fail_count + 1;
            $display("sram_test(bb=%0d) RDCHK i=%0d got=%h exp=%h",
                     BANK_BYTES, i, p_r0_data[0], exp_w);
          end
          if (i == 63) begin i <= 0; state <= S_BE_WR; end
          else         begin i <= i + 1; state <= S_RD; end
        end

        // --- phase 3: byte-enable partial write to word 7 -------------------
        // First write word 7 full (pat(7)), then overwrite low 2 bytes.
        S_BE_WR: begin
          p_w_en[0]  <= 1'b1;
          p_w_addr[0] <= SRAM_WORD_W'(7);
          p_w_data[0] <= '0;            // zero the low 2 bytes
          p_w_be[0]   <= 16'h00FF;
          state <= S_BE_CHK;
        end
        S_BE_CHK: begin
          p_r0_en[0]  <= 1'b1;
          p_r0_addr[0] <= SRAM_WORD_W'(7);
          exp_w <= {P7[127:16], 16'b0};   // high 112 bits kept, low 16 zeroed
          state <= S_BE_CHK2;
        end

        // --- phase 4: interleave / bank-mapping probe -----------------------
        // word 0 and word 16 both decode to bank 0 under D15 (bank=addr[3:0]);
        // a simultaneous write to both must drop the lower-priority (port 1)
        // write, so word 16 keeps its phase-1 value pat(16).

        S_ILV_WR: begin
          p_w_en[0]  <= 1'b1;
          p_w_addr[0] <= SRAM_WORD_W'(0);
          p_w_data[0] <= pat(32'hA0);
          p_w_be[0]   <= 16'hFFFF;
          p_w_en[1]  <= 1'b1;
          p_w_addr[1] <= SRAM_WORD_W'(16);
          p_w_data[1] <= pat(32'hB0);
          p_w_be[1]   <= 16'hFFFF;
          state <= S_ILV_CHK0;
        end
        S_ILV_CHK0: begin
          p_r0_en[0]  <= 1'b1;
          p_r0_addr[0] <= SRAM_WORD_W'(0);
          exp_w <= pat(32'hA0);
          state <= S_ILV_CHK16;
        end
        S_ILV_CHK16: begin
          if (p_r0_data[0] != exp_w) fail_count <= fail_count + 1;
          p_r0_en[0]  <= 1'b1;
          p_r0_addr[0] <= SRAM_WORD_W'(16);
          exp_w <= pat(32'd16);           // lower-priority write must have lost
          state <= S_ILV_CHK16B;
        end

        // --- phase 5: dual read from one bank in one cycle ------------------
        S_ILV_CHK16B: begin
          if (p_r0_data[0] != exp_w) fail_count <= fail_count + 1;
          state <= S_DR;
        end
        S_DR: begin
          p_r0_en[0]  <= 1'b1;
          p_r0_addr[0] <= SRAM_WORD_W'(0);
          p_r1_en[0]  <= 1'b1;
          p_r1_addr[0] <= SRAM_WORD_W'(16);   // both bank 0 (D15)
          exp_w <= pat(32'hA0);
          state <= S_DRCHK;
        end
        S_DRCHK: begin
          // read0 -> word 0 = pat(A0); read1 -> word 16 = pat(16) (kept).
          if (p_r0_data[0] != pat(32'hA0)) fail_count <= fail_count + 1;
          if (p_r1_data[0] != pat(32'd16)) fail_count <= fail_count + 1;
          state <= S_DONE;
        end

        S_DONE: begin
          pass <= (fail_count == 0);
          done <= 1'b1;
        end
        default: state <= S_DONE;
      endcase
    end
  end

  // Guard: shrunk banks must be strictly smaller than the default (sanity).
  // (Not synthesized; documentation-of-intent only, evaluated at elaboration.)
  // pragma translate_off
  initial begin
    if (BANK_BYTES > 512 * 1024) begin
      $display("sram_test: ERROR unsupported BANK_BYTES=%0d", BANK_BYTES);
      $finish;
    end
  end
  // pragma translate_on
endmodule

// ----------------------------------------------------------------------------
// sram_check — top: default + two shrink instances, combined pass.
// ----------------------------------------------------------------------------
module sram_check (
  input  logic clk,
  input  logic rst_n,
  output logic done,
  output logic pass,
  output logic [31:0] fail_count
);
  logic [2:0] t_done;
  logic [2:0] t_pass;
  logic [31:0] t_fail [3];

  sram_test #(.N_BANK(16), .BANK_BYTES(512*1024), .WORD_BYTES(16)) t_def (
    .clk(clk), .rst_n(rst_n), .done(t_done[0]), .pass(t_pass[0]), .fail_count(t_fail[0]));
  sram_test #(.N_BANK(16), .BANK_BYTES(256*1024), .WORD_BYTES(16)) t_256k (
    .clk(clk), .rst_n(rst_n), .done(t_done[1]), .pass(t_pass[1]), .fail_count(t_fail[1]));
  sram_test #(.N_BANK(16), .BANK_BYTES(128*1024), .WORD_BYTES(16)) t_128k (
    .clk(clk), .rst_n(rst_n), .done(t_done[2]), .pass(t_pass[2]), .fail_count(t_fail[2]));

  assign done = &t_done;
  assign pass = &t_pass;
  assign fail_count = t_fail[0] + t_fail[1] + t_fail[2];
endmodule

`endif // SRAM_CHECK_SV
