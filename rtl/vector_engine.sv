// ============================================================================
// vector_engine.sv — Vector Engine numeric core (co-sim model, 128-lane).
//
// All 18 VECTOR instructions (02 §3 / 04 §3.2), executed element-wise on
// 32-bit working values.  The Command Processor resolves addresses, reads
// operands from qmem, dtype-decodes them (bf16/fp16 -> fp32 bit pattern,
// int8/int16/int4 -> sign-extended int32) into a_vec/b_vec, and encodes the
// result (bf16/fp16 RNE writeback, int8/int4 packing) back to qmem.  This
// module is a pure combinational numeric core: op = f(a, b, cval, imm).
//
// cval carries the fp32 C-register operand (VSCALE scalar, RMSNorm eps, ROPE
// theta) or the reduction group count (VREDUCE_*).  imm is the raw 32-bit
// immediate.  b_vec carries the second operand, the RMSNorm gamma vector, or
// the QUANT/DEQUANT scale vector (broadcast per-128-group / per-tensor by the
// CP).  `go` gates the combinational compute (held low during operand
// marshalling).  The CP charges vector_latency(op) * ceil(len/128) cycles
// (qcore_pkg).
// ============================================================================
`ifndef VECTOR_ENGINE_SV
`define VECTOR_ENGINE_SV

`include "qcore_pkg.sv"
`include "softfloat.sv"
`include "rope_lut.sv"

module vector_engine #(
  parameter int MAX_VEC = 4096    // 2^13 working elements (covers INT=3072 SiLU, RoPE 2048)
) (
  input  logic [7:0]  op,
  input  logic [2:0]  srcA,
  input  logic [2:0]  srcB,
  input  logic [1:0]  acc,
  input  logic [31:0] len,        // element count for this instruction
  input  logic [31:0] cval,       // fp32 scalar / reduction group count
  input  logic        bcast,      // binary-op scalar broadcast (cv field != 0)
  input  logic [31:0] imm,        // raw 32-bit immediate
  input  logic        go,         // compute enable (held high for 1 cycle)
  input  logic [31:0] a_vec  [MAX_VEC],
  input  logic [31:0] b_vec  [MAX_VEC],
  output logic [31:0] out_vec [MAX_VEC],
  output logic [31:0] out_len
);
  import qcore_pkg::*;
  import softfloat_pkg::*;
  import rope_lut_pkg::*;

  integer i, j, g, ngrp, d, blk;
  logic is_fp;               // fp32 datapath vs int32 datapath
  logic [31:0] s;            // scalar (VSCALE)
  logic [31:0] neg_a;
  logic [31:0] rope_c [0:63];   // RoPE cos table (bf16-in-fp32), module level for Verilator
  logic [31:0] rope_s [0:63];   // RoPE sin table (bf16-in-fp32)

  assign is_fp = (srcA == DT_BF16) || (srcA == DT_FP16) ||
                 (srcB == DT_BF16) || (srcB == DT_FP16);

  always_comb begin
    // default: zero output (go=0 holds this), out_len = len
    for (i = 0; i < MAX_VEC; i++) out_vec[i] = 32'b0;
    for (d = 0; d < 64; d++) begin rope_c[d] = 32'b0; rope_s[d] = 32'b0; end
    out_len = len;

    if (go) begin
      case (op)
        OP_VADD, OP_VSUB, OP_VMUL, OP_VDIV, OP_VMAX: begin
          for (i = 0; i < len; i++) begin
            logic [31:0] av, bv, r;
            av = a_vec[i];
            // bcast -> scalar b[0] broadcast across all lanes
            bv = bcast ? b_vec[0] : b_vec[i];
            if (is_fp) begin
              case (op)
                OP_VADD: r = fp32_add(av, bv);
                OP_VSUB: r = fp32_sub(av, bv);
                OP_VMUL: r = fp32_mul(av, bv);
                OP_VDIV: r = fp32_div(av, bv);
                default: r = fp32_max(av, bv);
              endcase
            end else begin
              case (op)
                OP_VADD: r = $signed(av) + $signed(bv);
                OP_VSUB: r = $signed(av) - $signed(bv);
                OP_VMUL: r = $signed(av) * $signed(bv);
                default: r = ($signed(av) >= $signed(bv)) ? av : bv;  // VMAX
              endcase
            end
            out_vec[i] = r;
          end
        end

        OP_VRECIP: for (i = 0; i < len; i++) out_vec[i] = fp32_recip(a_vec[i]);

        OP_VEXP: for (i = 0; i < len; i++) out_vec[i] = fp32_exp(a_vec[i]);

        OP_VRSQRT: for (i = 0; i < len; i++) out_vec[i] = fp32_rsqrt(a_vec[i]);

        OP_VSILU: for (i = 0; i < len; i++) begin
          // silu(a) = a * sigmoid(a) = a / (1 + exp(-a))
          neg_a = {~a_vec[i][31], a_vec[i][30:0]};
          out_vec[i] = fp32_div(a_vec[i],
                                fp32_add(F_ONE, fp32_exp(neg_a)));
        end

        OP_VMOV: for (i = 0; i < len; i++) out_vec[i] = a_vec[i];

        OP_VSCALE: begin
          s = (acc == ACC_FP32) ? cval : bf16_to_fp32(imm[15:0]);
          for (i = 0; i < len; i++) out_vec[i] = fp32_mul(a_vec[i], s);
        end

        OP_VMASK: begin
          // rows = imm[31:16], cols = imm[15:0]; col_base = cval[31:16],
          // row_base = cval[15:0].  tile[r][c] = 0 if col_base+c <= row_base+r
          // else -inf (fp32 bit patterns; CP writes bf16).
          logic [15:0] rows, cols, row_base, col_base;
          integer r, c;
          rows = imm[31:16]; cols = imm[15:0];
          row_base = cval[15:0]; col_base = cval[31:16];
          for (r = 0; r < rows; r++) begin
            for (c = 0; c < cols; c++) begin
              out_vec[r * cols + c] =
                ((col_base + c[15:0]) <= (row_base + r[15:0])) ? 32'b0 : F_NEG_INF;
            end
          end
          out_len = rows * cols;
        end

        OP_VREDUCE_SUM, OP_VREDUCE_MAX: begin
          // ngroups from cval (>=1); a is [ngroups x len]
          ngrp = (cval == 32'b0) ? 1 : cval;
          for (g = 0; g < ngrp; g++) begin
            logic [31:0] accv;
            if (op == OP_VREDUCE_MAX) begin
              accv = a_vec[g * len];
              for (i = 1; i < len; i++) accv = fp32_max(accv, a_vec[g * len + i]);
            end else begin
              accv = 32'b0;
              for (i = 0; i < len; i++) accv = fp32_add(accv, a_vec[g * len + i]);
            end
            out_vec[g] = accv;
          end
          out_len = ngrp;
        end

        OP_ROPE: begin
          // theta = rope_theta = 1e6 (spec 02 §7.2, frozen v0); pos = imm[15:0].
          // inv_freq[d] comes from the ROPE_INVF LUT (bit-exact vs numpy
          // 1/powf(1e6, d/64)), so ang = pos*inv_freq matches the executor and
          // cos/sin stay 0 ULP at pos up to 8192 (and beyond).
          logic [31:0] posf;
          integer nblk, dd, bb;
          posf       = i32_to_f32({16'b0, imm[15:0]});
          for (dd = 0; dd < 64; dd++) begin
            logic [31:0] ang;
            logic [63:0] sc;
            ang = fp32_mul(posf, ROPE_INVF[dd]);
            // sin/cos LUT (spec 04 §3.2): fp32 lerp, then round to bf16 (HF bit-match)
            sc       = rope_sincos(ang);
            rope_c[dd] = bf16_to_fp32(fp32_to_bf16(sc[63:32]));
            rope_s[dd] = bf16_to_fp32(fp32_to_bf16(sc[31:0]));
          end
          nblk = len / 128;
          for (bb = 0; bb < nblk; bb++) begin
            for (dd = 0; dd < 64; dd++) begin
              // HF rotate_half (bit-match): out[:, :half] = bf16(x1*cos) + bf16(-x2*sin),
              // out[:, half:] = bf16(x2*cos) + bf16(x1*sin).  cos/sin are already bf16,
              // and the rotation runs in bf16 arithmetic with per-op rounding — the
              // products are rounded to bf16 BEFORE the add/sub (executor _rope_apply).
              logic [31:0] xr, xi, t1, t2, t3, t4, or_, oi;
              xr = a_vec[bb * 128 + dd];        // x1
              xi = a_vec[bb * 128 + 64 + dd];   // x2
              t1 = bf16_to_fp32(fp32_to_bf16(fp32_mul(xr, rope_c[dd])));    // bf16(x1*cos)
              t2 = bf16_to_fp32(fp32_to_bf16(fp32_mul(xi, rope_s[dd])));    // bf16(x2*sin)
              t3 = bf16_to_fp32(fp32_to_bf16(fp32_mul(xi, rope_c[dd])));    // bf16(x2*cos)
              t4 = bf16_to_fp32(fp32_to_bf16(fp32_mul(xr, rope_s[dd])));    // bf16(x1*sin)
              or_ = bf16_to_fp32(fp32_to_bf16(fp32_sub(t1, t2)));           // bf16(t1 - t2)
              oi  = bf16_to_fp32(fp32_to_bf16(fp32_add(t3, t4)));           // bf16(t3 + t4)
              out_vec[bb * 128 + dd]      = or_;
              out_vec[bb * 128 + 64 + dd] = oi;
            end
          end
        end

        OP_RMSNORM: begin
          // eps = cval (fp32), per_head = imm[31], gamma = b_vec[len].
          logic per_head;
          logic [31:0] eps;
          eps = cval;
          per_head = imm[31];
          if (per_head) begin
            integer nblk2;
            nblk2 = len / 128;
            for (blk = 0; blk < nblk2; blk++) begin
              logic [31:0] sumsq, mean, rinv;
              sumsq = 32'b0;
              for (j = 0; j < 128; j++) begin
                logic [31:0] av;
                av = a_vec[blk * 128 + j];
                sumsq = fp32_add(sumsq, fp32_mul(av, av));
              end
              mean = fp32_div(sumsq, i32_to_f32(32'd128));
              rinv = fp32_rsqrt(fp32_add(mean, eps));
              for (j = 0; j < 128; j++)
                out_vec[blk * 128 + j] =
                  fp32_mul(fp32_mul(a_vec[blk * 128 + j], rinv), b_vec[blk * 128 + j]);
            end
          end else begin
            logic [31:0] sumsq, mean, rinv;
            sumsq = 32'b0;
            for (j = 0; j < len; j++) begin
              sumsq = fp32_add(sumsq, fp32_mul(a_vec[j], a_vec[j]));
            end
            mean = fp32_div(sumsq, i32_to_f32(len));
            rinv = fp32_rsqrt(fp32_add(mean, eps));
            for (j = 0; j < len; j++)
              out_vec[j] = fp32_mul(fp32_mul(a_vec[j], rinv), b_vec[j]);
          end
        end
        OP_QUANT: begin
          // a fp32, scales in b_vec (broadcast per-128-group / per-tensor by cval[20]).
          for (i = 0; i < len; i++) begin
            logic [31:0] sc, qv;
            integer qmin, qmax;
            sc = cval[20] ? b_vec[i / 128] : b_vec[0];
            qv = f32_to_i32_rne(fp32_div(a_vec[i], sc));
            if (srcB == DT_INT4) begin qmin = -8;  qmax = 7;  end
            else                 begin qmin = -127; qmax = 127; end
            if ($signed(qv) < qmin) qv = 32'(qmin);
            if ($signed(qv) > qmax) qv = 32'(qmax);
            out_vec[i] = qv;
          end
        end

        OP_DEQUANT: begin
          for (i = 0; i < len; i++) begin
            logic [31:0] sc;
            sc = cval[20] ? b_vec[i / 128] : b_vec[0];
            out_vec[i] = fp32_mul(i32_to_f32(a_vec[i]), sc);
          end
        end
        default: begin
          for (i = 0; i < len; i++) out_vec[i] = a_vec[i];
        end
      endcase
    end
  end

endmodule
`endif // VECTOR_ENGINE_SV
