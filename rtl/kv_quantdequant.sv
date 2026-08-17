// ============================================================================
// kv_quantdequant.sv — B' KV quantize-on-write datapath (APPEND side only).
//
// One invocation quantizes ONE head (128 dims) of ONE tensor (K or V) from the
// BF16 SRAM staging into the HBM data + scale slabs.
//
// The LOAD-side staged dequant (S_RDS/S_RD) is RETIRED: quantized KV is now
// consumed inline by the matrix B-feed (rtl/kv_bfeed.sv), which dequantizes +
// rotates on the fly — no BF16 staging write.  This module keeps only the
// APPEND quantize path (INT8-K QK-norm folded / INT4-V).
//
// Scheme (DECISION §5 / fold-verify, hardware-faithful BF16 rounding):
//   quant K:  s_q = bf16(max|k_unit|/127);  q = round(k_unit/s_q) clip[-127,127]
//   quant V:  s_v = bf16(max|v|/7);  q4 = round(v/s_v) clip[-7,7] (packed 2/byte)
// Reuses the softfloat fp32 core (i32_to_f32 / f32_to_i32_rne / fp32_mul /
// fp32_div / fp32_to_bf16) — the same W4A16/INT8 mantissa path.
//
// Memory model (05 §1.3 + B' scale slabs):
//   K data slab (INT8):  data_base   (128 B/token, pos<<7)
//   V data slab (INT4):  data_base   (64 B/token,  pos<<6)
//   scale record:        scale_base -> [s_q (2B @ +0)][s_v (2B @ +2)]  (HBM)
//   staging:             sram_base (read 256B BF16 source)
//
// NOTE: s_q/s_v clamp min to 1e-6 in the executor (degenerate all-zero
// activations); for unit-norm K / normal V this never binds, so RTL omits it.
// ============================================================================
`ifndef KV_QUANTDEQUANT_SV
`define KV_QUANTDEQUANT_SV

`include "qcore_pkg.sv"
`include "softfloat.sv"

module kv_quantdequant #(
  parameter int MAX_ELEMS = 128     // one head = 128 dims
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic        kv,           // 0 = K (INT8 fold), 1 = V (INT4)
  input  logic [39:0] data_base,    // HBM data slab byte base (this head+token)
  input  logic [39:0] scale_base,   // HBM scale record byte base ([s_q][s_v])
  input  logic [39:0] sram_base,    // SRAM staging byte addr (BF16 source)
  output logic        rd_sel,
  output logic [39:0] rd_addr,
  input  logic [7:0]  rd_data,
  output logic        wr_en,
  output logic        wr_sel,
  output logic [39:0] wr_addr,
  output logic [7:0]  wr_data,
  output logic        done
);
  import qcore_pkg::*;
  import softfloat_pkg::*;

  logic [31:0] qbuf [0:MAX_ELEMS-1];   // fp32 working buffer (one head)

  // quant_rne: round-to-nearest-even of f to an integer.  The frozen
  // softfloat f32_to_i32_rne returns 0 for |f| in [0.5, 1.0) (biased exp 126),
  // which is wrong for RNE (0.6 -> 1).  The B' executor reference uses
  // numpy `round` (RNE), so this local fix keeps the module bit-consistent
  // without touching the frozen softfloat (default-regression contract).
  function automatic logic [31:0] quant_rne(input logic [31:0] f);
    logic s;
    logic [7:0] e;
    logic [22:0] m;
    s = f[31]; e = f[30:23]; m = f[22:0];
    if (e == 8'd126) begin
      // |f| in [0.5, 1.0): exactly 0.5 -> 0 (even), else -> +/-1
      if (m == 23'b0) quant_rne = 32'b0;
      else            quant_rne = s ? 32'hFFFFFFFF : 32'd1;
    end else begin
      quant_rne = f32_to_i32_rne(f);
    end
  endfunction

  localparam logic [2:0]
    S_IDLE   = 3'd0,
    S_LD     = 3'd1,   // read BF16 source, track max (quant)
    S_SCL    = 3'd2,   // write scale (2B)
    S_WR     = 3'd3,   // quantize -> data bytes
    S_DONE   = 3'd4;

  logic [2:0]  state;
  logic [7:0]  idx;             // element/channel index 0..127
  logic [2:0]  sub;             // micro-step
  logic [7:0]  accb;            // low byte of a 2-byte read
  logic [31:0] amax;            // running max magnitude (fp32)
  logic [15:0] s_bits;          // s_q or s_v (BF16)
  logic [3:0]  lo_nib;          // quant V: pending low nibble

  // ---- combinational read address -------------------------------------
  always_comb begin
    rd_sel = 1'b0; rd_addr = 40'b0;
    case (state)
      S_LD: begin rd_sel = 1'b0; rd_addr = sram_base + idx*2 + sub[0]; end
      default: begin end
    endcase
  end

  // ---- combinational write ---------------------------------------------
  always_comb begin
    wr_en = 1'b0; wr_sel = 1'b0; wr_addr = 40'b0; wr_data = 8'b0;
    case (state)
      S_SCL: begin
        logic [15:0] s_tmp;
        // s_bits is registered at sub=0; the combinational write must not read
        // the stale value, so recompute the scale here (amax/kv stable in S_SCL).
        s_tmp = fp32_to_bf16(fp32_div(amax, i32_to_f32(kv ? 32'd7 : 32'd127)));
        wr_en = 1'b1; wr_sel = 1'b1; wr_addr = scale_base + (kv ? 2 : 0) + sub[0];
        wr_data = sub[0] ? s_tmp[15:8] : s_tmp[7:0];
      end
      S_WR: begin
        logic [31:0] qv;
        qv = quant_rne(fp32_div(qbuf[idx], bf16_to_fp32(s_bits)));
        if (!kv) begin
          if ($signed(qv) < -32'sd127) qv = 32'shFFFFFF81;
          if ($signed(qv) >  32'sd127) qv = 32'sd127;
          wr_en = 1'b1; wr_sel = 1'b1; wr_addr = data_base + idx;
          wr_data = qv[7:0];
        end else begin
          logic [31:0] qve;
          if ($signed(qv) < -7) qv = 32'shFFFFFFF9;
          if ($signed(qv) >  7) qv = 32'sd7;
          // even element -> low nibble; odd -> combine with the freshly
          // recomputed (and clipped) even nibble (avoid the stale lo_nib reg).
          qve = (idx[0]) ? quant_rne(fp32_div(qbuf[idx-1], bf16_to_fp32(s_bits))) : 32'b0;
          if ($signed(qve) < -7) qve = 32'shFFFFFFF9;
          if ($signed(qve) >  7) qve = 32'sd7;
          wr_en = 1'b1; wr_sel = 1'b1; wr_addr = data_base + (idx >> 1);
          wr_data = idx[0] ? {qv[3:0], qve[3:0]} : qv[3:0];
        end
      end
      default: begin end
    endcase
  end

  // ---- FSM --------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE; idx <= 8'b0; sub <= 3'b0; accb <= 8'b0;
      amax <= 32'b0; s_bits <= 16'b0; lo_nib <= 4'b0; done <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            idx <= 8'b0; sub <= 3'b0; amax <= 32'b0; accb <= 8'b0; lo_nib <= 4'b0;
            state <= S_LD;
          end
        end

        // ---------- quantize (write) ----------
        S_LD: begin
          if (sub == 3'd0) begin
            accb <= rd_data; sub <= 3'd1;
          end else begin
            logic [31:0] val, mag, valf;
            val = {rd_data, accb};
            valf = bf16_to_fp32(val[15:0]);
            mag = {1'b0, valf[30:0]};
            qbuf[idx] <= valf;
            amax <= (idx == 0) ? mag : ((mag > amax) ? mag : amax);
            if (idx == 127) begin idx <= 8'b0; sub <= 3'b0; state <= S_SCL; end
            else begin idx <= idx + 8'd1; sub <= 3'b0; end
          end
        end

        S_SCL: begin
          if (sub == 3'd0) begin
            s_bits <= fp32_to_bf16(fp32_div(amax,
              i32_to_f32(kv ? 32'd7 : 32'd127)));
            sub <= 3'd1;
          end else begin
            sub <= 3'd0; idx <= 8'b0; state <= S_WR;
          end
        end

        S_WR: begin
          if (idx[0] && kv) begin
            logic [31:0] qvprev;
            qvprev = quant_rne(fp32_div(qbuf[idx-1], bf16_to_fp32(s_bits)));
            lo_nib <= qvprev[3:0];
          end
          if (idx == 127) begin state <= S_DONE; done <= 1'b1; end
          else idx <= idx + 8'd1;
        end

        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
`endif // KV_QUANTDEQUANT_SV
