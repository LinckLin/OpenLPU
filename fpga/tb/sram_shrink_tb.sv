// ============================================================================
// sram_shrink_tb.sv — ddr_if SRAM_BYTES shrink-instance smoke (P9b follow-up).
//
// Uses the P10b SRAM parametrisation (ddr_if.SRAM_BYTES) to instantiate the
// on-chip scratchpad at the two shrink sizes from plans/m8-wait-plan.md §3:
//   256 KiB/bank -> 4 MiB total  (84% of the 4.75 MiB ZCU104 budget)
//   128 KiB/bank -> 2 MiB total  (42% of the 4.75 MiB budget)
//
// Each instance runs the same self-checking sequence (Verilator 4.038 is
// cycle-based, no `#` timing):
//   1. engine byte write/read round-trip over a 64-point sweep spanning the
//      full shrunk address space [0, SRAM_BYTES);
//   2. engine byte round-trip at the top boundary SRAM_BYTES-3 .. -1 (proves
//      the $clog2(SRAM_BYTES) address width is correct for the shrunk size);
//   3. DDR (rd_sel=1) sparse round-trip — the shrink must not touch the DDR
//      associative array;
//   4. host-port byte round-trip (qbin load / logits readback path).
//
// The M6 co-sim criterion (per-instruction trace identical + bf16 <= 1 ULP)
// is preserved because the byte-level read/write semantics are unchanged for
// every in-range address; the full integration smoke (run_fpga_smoke.py, with
// the shrunk qcore_fpga_top.SRAM_BYTES) is the end-to-end M6-criterion proof.
// Build: see fpga/tb/run_sram_shrink.sh.
// ============================================================================
`ifndef SRAM_SHRINK_TB_SV
`define SRAM_SHRINK_TB_SV

`include "ddr_if.sv"

// ----------------------------------------------------------------------------
// ddr_shrink_test — parameterized self-checking sequence around one ddr_if.
// ----------------------------------------------------------------------------
module ddr_shrink_test #(
  parameter int SRAM_BYTES = 4 * 1024 * 1024   // 256 KiB/bank x 16
) (
  input  logic        clk,
  input  logic        rst_n,
  output logic        done,
  output logic        pass,
  output logic [31:0] fail_count
);
  localparam int SRAM_ADDR_BITS = $clog2(SRAM_BYTES);
  localparam int N_SWEEP        = 64;
  localparam logic [39:0] SWEEP_STRIDE = SRAM_BYTES / N_SWEEP;

  // ---- ddr_if ports --------------------------------------------------------
  logic        rd_sel, wr_en, wr_sel;
  logic [39:0] rd_addr, wr_addr;
  logic [7:0]  rd_data, wr_data;
  logic        hd_en, hd_sel;
  logic [39:0] hd_addr;
  logic [7:0]  hd_wdata, hd_rdata;

  logic [39:0] m_axi_araddr, m_axi_awaddr;
  logic [7:0]  m_axi_arlen, m_axi_awlen;
  logic [2:0]  m_axi_arsize, m_axi_awsize;
  logic [1:0]  m_axi_arburst, m_axi_awburst;
  logic        m_axi_arvalid, m_axi_awvalid, m_axi_wlast, m_axi_wvalid;
  logic        m_axi_rready, m_axi_bready;
  logic [63:0] m_axi_rdata, m_axi_wdata;
  logic [7:0]  m_axi_wstrb;
  logic [1:0]  m_axi_rresp, m_axi_bresp;
  logic        m_axi_rlast, m_axi_rvalid, m_axi_bvalid;

  ddr_if #(.SRAM_BYTES(SRAM_BYTES)) dut (
    .clk(clk), .rst_n(rst_n),
    .rd_sel(rd_sel), .rd_addr(rd_addr), .rd_data(rd_data),
    .wr_en(wr_en), .wr_sel(wr_sel), .wr_addr(wr_addr), .wr_data(wr_data),
    .hd_en(hd_en), .hd_sel(hd_sel), .hd_addr(hd_addr),
    .hd_wdata(hd_wdata), .hd_rdata(hd_rdata),
    .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
    .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(1'b0),
    .m_axi_rdata(64'b0), .m_axi_rresp(2'b0), .m_axi_rlast(1'b0),
    .m_axi_rvalid(1'b0), .m_axi_rready(m_axi_rready),
    .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
    .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
    .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(1'b0),
    .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
    .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
    .m_axi_wready(1'b0),
    .m_axi_bresp(2'b0), .m_axi_bvalid(1'b0), .m_axi_bready(m_axi_bready)
  );

  // deterministic byte pattern
  function automatic logic [7:0] pat(input logic [31:0] k);
    return 8'((k * 37 + 90) & 32'hFF);
  endfunction

  typedef enum logic [3:0] {
    S_RESET, S_SWEEP_WR, S_SWEEP_RD, S_SWEEP_CHK,
    S_TOP_WR, S_TOP_RD, S_TOP_CHK,
    S_DDR_WR, S_DDR_RD, S_DDR_CHK,
    S_HOST_WR, S_HOST_RD, S_HOST_CHK, S_DONE
  } state_t;

  state_t      state;
  logic [31:0] i;
  logic [7:0]  exp;
  logic [39:0] addr;

  // sweep address: i * SWEEP_STRIDE (covers 0 .. SRAM_BYTES-SWEEP_STRIDE)
  function automatic logic [39:0] sweep_addr(input logic [31:0] k);
    return 40'(k) * SWEEP_STRIDE;
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_RESET;
      i <= 0;
      exp <= 8'b0;
      addr <= 40'b0;
      fail_count <= 0;
      pass <= 1'b0;
      done <= 1'b0;
      rd_sel <= 0; rd_addr <= 40'b0;
      wr_en <= 0; wr_sel <= 0; wr_addr <= 40'b0; wr_data <= 8'b0;
      hd_en <= 0; hd_sel <= 0; hd_addr <= 40'b0; hd_wdata <= 8'b0;
    end else begin
      // defaults: idle engine/host ports
      rd_sel <= 0; rd_addr <= 40'b0;
      wr_en <= 0; wr_sel <= 0; wr_addr <= 40'b0; wr_data <= 8'b0;
      hd_en <= 0; hd_sel <= 0; hd_addr <= 40'b0; hd_wdata <= 8'b0;
      exp <= 8'b0;
      case (state)
        S_RESET: begin
          i <= 0;
          state <= S_SWEEP_WR;
        end

        // ---- phase 1: engine SRAM sweep round-trip ------------------------
        S_SWEEP_WR: begin
          wr_en <= 1'b1; wr_sel <= 1'b0;
          wr_addr <= sweep_addr(i);
          wr_data <= pat(i);
          state <= S_SWEEP_RD;
        end
        S_SWEEP_RD: begin
          rd_sel <= 1'b0;
          rd_addr <= sweep_addr(i);
          exp <= pat(i);
          state <= S_SWEEP_CHK;
        end
        S_SWEEP_CHK: begin
          if (rd_data != exp) begin
            fail_count <= fail_count + 1;
            $display("sram_shrink(bb=%0d) SWEEP i=%0d addr=%0d got=%02x exp=%02x",
                     SRAM_BYTES, i, sweep_addr(i), rd_data, exp);
          end
          if (i == N_SWEEP - 1) begin i <= 0; state <= S_TOP_WR; end
          else                  begin i <= i + 1; state <= S_SWEEP_WR; end
        end

        // ---- phase 2: top-boundary round-trip (SRAM_BYTES-3 .. -1) --------
        S_TOP_WR: begin
          wr_en <= 1'b1; wr_sel <= 1'b0;
          wr_addr <= 40'(SRAM_BYTES - 1) - 40'(i);   // i=0..2 -> top-1, top-2, top-3
          wr_data <= pat(32'hCAFE0000 + i);
          state <= S_TOP_RD;
        end
        S_TOP_RD: begin
          rd_sel <= 1'b0;
          rd_addr <= 40'(SRAM_BYTES - 1) - 40'(i);
          exp <= pat(32'hCAFE0000 + i);
          state <= S_TOP_CHK;
        end
        S_TOP_CHK: begin
          if (rd_data != exp) begin
            fail_count <= fail_count + 1;
            $display("sram_shrink(bb=%0d) TOP i=%0d addr=%0d got=%02x exp=%02x",
                     SRAM_BYTES, i, 40'(SRAM_BYTES - 1) - 40'(i), rd_data, exp);
          end
          if (i == 2) begin i <= 0; state <= S_DDR_WR; end
          else        begin i <= i + 1; state <= S_TOP_WR; end
        end

        // ---- phase 3: DDR (sel=1) unaffected by the shrink ----------------
        S_DDR_WR: begin
          wr_en <= 1'b1; wr_sel <= 1'b1;
          wr_addr <= {40'b0} | (40'(i) << 12);      // 0x000, 0x1000, 0x2000
          wr_data <= pat(32'h5A5A0000 + i);
          state <= S_DDR_RD;
        end
        S_DDR_RD: begin
          rd_sel <= 1'b1;
          rd_addr <= {40'b0} | (40'(i) << 12);
          exp <= pat(32'h5A5A0000 + i);
          state <= S_DDR_CHK;
        end
        S_DDR_CHK: begin
          if (rd_data != exp) begin
            fail_count <= fail_count + 1;
            $display("sram_shrink(bb=%0d) DDR i=%0d addr=%0d got=%02x exp=%02x",
                     SRAM_BYTES, i, 40'(i) << 12, rd_data, exp);
          end
          if (i == 2) begin i <= 0; state <= S_HOST_WR; end
          else        begin i <= i + 1; state <= S_DDR_WR; end
        end

        // ---- phase 4: host-port round-trip (qbin load / logits readback) --
        S_HOST_WR: begin
          hd_en <= 1'b1; hd_sel <= 1'b0;
          hd_addr <= sweep_addr(i);
          hd_wdata <= pat(32'hA5A50000 + i);
          state <= S_HOST_RD;
        end
        S_HOST_RD: begin
          hd_en <= 1'b0; hd_sel <= 1'b0;
          hd_addr <= sweep_addr(i);
          exp <= pat(32'hA5A50000 + i);
          state <= S_HOST_CHK;
        end
        S_HOST_CHK: begin
          if (hd_rdata != exp) begin
            fail_count <= fail_count + 1;
            $display("sram_shrink(bb=%0d) HOST i=%0d addr=%0d got=%02x exp=%02x",
                     SRAM_BYTES, i, sweep_addr(i), hd_rdata, exp);
          end
          if (i == N_SWEEP - 1) begin i <= 0; state <= S_DONE; end
          else                  begin i <= i + 1; state <= S_HOST_WR; end
        end

        S_DONE: begin
          pass <= (fail_count == 0);
          done <= 1'b1;
        end
        default: state <= S_DONE;
      endcase
    end
  end

  // Sanity guard (not synthesized; evaluated at elaboration).
  // pragma translate_off
  initial begin
    if (SRAM_BYTES > 8 * 1024 * 1024) begin
      $display("sram_shrink_tb: ERROR unsupported SRAM_BYTES=%0d", SRAM_BYTES);
      $finish;
    end
    if (SRAM_BYTES <= 0) begin
      $display("sram_shrink_tb: ERROR non-positive SRAM_BYTES=%0d", SRAM_BYTES);
      $finish;
    end
  end
  // pragma translate_on
endmodule

// ----------------------------------------------------------------------------
// sram_shrink_tb — top: two shrink instances (4 MiB + 2 MiB), combined pass.
// ----------------------------------------------------------------------------
module sram_shrink_tb (
  input  logic        clk,
  input  logic        rst_n,
  output logic        done,
  output logic        pass,
  output logic [31:0] fail_count
);
  logic [1:0] t_done;
  logic [1:0] t_pass;
  logic [31:0] t_fail [2];

  ddr_shrink_test #(.SRAM_BYTES(256 * 1024 * 16)) t_256k (   // 256 KiB/bank -> 4 MiB
    .clk(clk), .rst_n(rst_n), .done(t_done[0]), .pass(t_pass[0]), .fail_count(t_fail[0]));
  ddr_shrink_test #(.SRAM_BYTES(128 * 1024 * 16)) t_128k (   // 128 KiB/bank -> 2 MiB
    .clk(clk), .rst_n(rst_n), .done(t_done[1]), .pass(t_pass[1]), .fail_count(t_fail[1]));

  assign done = &t_done;
  assign pass = &t_pass;
  assign fail_count = t_fail[0] + t_fail[1];
endmodule

`endif // SRAM_SHRINK_TB_SV
