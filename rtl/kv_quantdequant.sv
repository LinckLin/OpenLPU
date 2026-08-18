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
// s_q/s_v clamp to float32 1e-6 before BF16 rounding, matching the executor for
// degenerate all-zero or very small activations.
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

  function automatic logic [31:0] quant_clip(
    input logic [31:0] f,
    input logic        is_v
  );
    logic [31:0] q;
    q = quant_rne(f);
    if (is_v) begin
      if ($signed(q) < -32'sd7) q = 32'hFFFFFFF9;
      if ($signed(q) >  32'sd7) q = 32'd7;
    end else begin
      if ($signed(q) < -32'sd127) q = 32'hFFFFFF81;
      if ($signed(q) >  32'sd127) q = 32'd127;
    end
    quant_clip = q;
  endfunction

  // This is the seed from fp32_recip.  S_INV preserves the original iteration
  // order, but registers each mul/sub so no Newton-Raphson cone spans a full
  // clock cycle.
  function automatic logic [31:0] recip_seed(input logic [31:0] b);
    logic [7:0]  eb;
    logic [22:0] mb;
    eb = b[30:23]; mb = b[22:0];
    if (eb == 8'hFF || eb == 0)
      recip_seed = b;
    else
      recip_seed = {b[31], 8'd253 - eb, 23'h7FFFFF - (mb >> 1)};
  endfunction

  localparam logic [2:0]
    S_IDLE   = 3'd0,
    S_LD     = 3'd1,   // read BF16 source, track max (quant)
    S_SCL    = 3'd2,   // write scale (2B)
    S_INV    = 3'd3,   // reciprocal, 4 iterations x (mul/sub/mul)
    S_WR     = 3'd4,   // pipelined quantize -> data bytes
    S_DRAIN  = 3'd5,   // retire the final two pipeline entries
    S_DONE   = 3'd6;

  logic [2:0]  state;
  logic [7:0]  idx;             // element/channel index 0..127
  logic [2:0]  sub;             // micro-step
  logic [7:0]  accb;            // low byte of a 2-byte read
  logic [31:0] amax;            // running max magnitude (fp32)
  logic [15:0] s_bits;          // s_q or s_v (BF16)
  logic [3:0]  lo_nib;          // quant V: pending low nibble
  logic [31:0] scale_f;         // BF16-exact scale in fp32 form
  logic [31:0] inv_scale;       // reciprocal after four NR iterations
  logic [31:0] recip_r;         // NR iteration input carried to final mul
  logic [31:0] recip_prod;      // scale_f * recip_r
  logic [31:0] recip_err;       // 2.0 - recip_prod
  logic [1:0]  inv_iter;        // Newton-Raphson iteration 0..3
  logic [31:0] qmul_pipe;       // qbuf[idx] * inv_scale
  logic [7:0]  qmul_idx;
  logic        qmul_valid;
  logic [7:0]  qout_data;       // clipped two's-complement integer
  logic [7:0]  qout_idx;
  logic        qout_valid;

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
        // sub=0 computes s_bits.  Write the registered bytes at sub=1/2 so
        // the scale divider is not part of an output path.
        if (sub == 3'd1 || sub == 3'd2) begin
          wr_en = 1'b1; wr_sel = 1'b1;
          wr_addr = scale_base + (kv ? 2 : 0) + (sub - 3'd1);
          wr_data = (sub == 3'd1) ? s_bits[7:0] : s_bits[15:8];
        end
      end
      S_WR, S_DRAIN: begin
        if (qout_valid) begin
          wr_en = 1'b1; wr_sel = 1'b1;
          if (!kv) begin
            wr_addr = data_base + qout_idx;
            wr_data = qout_data;
          end else begin
            wr_addr = data_base + (qout_idx >> 1);
            wr_data = qout_idx[0] ? {qout_data[3:0], lo_nib} : qout_data[3:0];
          end
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
      scale_f <= 32'b0; inv_scale <= 32'b0; recip_r <= 32'b0;
      recip_prod <= 32'b0; recip_err <= 32'b0; inv_iter <= 2'b0;
      qmul_pipe <= 32'b0; qmul_idx <= 8'b0; qmul_valid <= 1'b0;
      qout_data <= 8'b0; qout_idx <= 8'b0; qout_valid <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            idx <= 8'b0; sub <= 3'b0; amax <= 32'b0; accb <= 8'b0; lo_nib <= 4'b0;
            qmul_valid <= 1'b0; qout_valid <= 1'b0;
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
            logic [31:0] sf;
            sf = fp32_div(amax, i32_to_f32(kv ? 32'd7 : 32'd127));
            if (sf < 32'h358637BD) sf = 32'h358637BD;  // float32(1e-6)
            s_bits <= fp32_to_bf16(sf);
            sub <= 3'd1;
          end else if (sub == 3'd1) begin
            logic [31:0] sf;
            sf = bf16_to_fp32(s_bits);
            scale_f <= sf;
            inv_scale <= recip_seed(sf);
            sub <= 3'd2;
          end else begin
            sub <= 3'd0; inv_iter <= 2'b0; state <= S_INV;
          end
        end

        // One soft-float operation per register slice.  Four iterations match
        // fp32_recip exactly: r <- r * (2 - scale_f * r).
        S_INV: begin
          if (scale_f[30:23] == 8'b0 || scale_f[30:23] == 8'hFF) begin
            // fp32_recip returns zero/Inf/NaN inputs without NR iterations.
            sub <= 3'd0;
            if (inv_iter == 2'd3) begin
              idx <= 8'b0; qmul_valid <= 1'b0; qout_valid <= 1'b0;
              state <= S_WR;
            end else begin
              inv_iter <= inv_iter + 2'd1;
            end
          end else if (sub == 3'd0) begin
            recip_r <= inv_scale;
            recip_prod <= fp32_mul(scale_f, inv_scale);
            sub <= 3'd1;
          end else if (sub == 3'd1) begin
            recip_err <= fp32_sub(F_TWO, recip_prod);
            sub <= 3'd2;
          end else begin
            inv_scale <= fp32_mul(recip_r, recip_err);
            sub <= 3'd0;
            if (inv_iter == 2'd3) begin
              idx <= 8'b0; qmul_valid <= 1'b0; qout_valid <= 1'b0;
              state <= S_WR;
            end else begin
              inv_iter <= inv_iter + 2'd1;
            end
          end
        end

        S_WR: begin
          logic [31:0] qv;
          if (qout_valid && kv && !qout_idx[0]) begin
            lo_nib <= qout_data[3:0];
          end
          qv = quant_clip(qmul_pipe, kv);
          qout_data <= qv[7:0];
          qout_idx <= qmul_idx;
          qout_valid <= qmul_valid;
          qmul_pipe <= fp32_mul(qbuf[idx], inv_scale);
          qmul_idx <= idx;
          qmul_valid <= 1'b1;
          if (idx == 127) state <= S_DRAIN;
          else idx <= idx + 8'd1;
        end

        S_DRAIN: begin
          logic [31:0] qv;
          if (qout_valid && kv && !qout_idx[0]) begin
            lo_nib <= qout_data[3:0];
          end
          qv = quant_clip(qmul_pipe, kv);
          qout_data <= qv[7:0];
          qout_idx <= qmul_idx;
          qout_valid <= qmul_valid;
          qmul_valid <= 1'b0;
          if (qout_valid && qout_idx == 8'd127) begin
            qout_valid <= 1'b0;
            done <= 1'b1;
            state <= S_DONE;
          end
        end

        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
`endif // KV_QUANTDEQUANT_SV
