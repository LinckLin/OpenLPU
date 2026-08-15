// ============================================================================
// synth_datapath.sv — QCore FP datapath representative synthesis top (P10/M9/P10b).
// BF16 / FP32 / INT32 arithmetic primitives (the MAC array + vector ALU core).
//
// P10b change (plans/m8-wait-plan.md §3): the primitives are now *pipelined*
// into explicit clocked stages so the physical datapath meets the target clock
// instead of the direct-combination mapping (M9 §3).  Pipeline stage counts are
// bounded by the frozen cycle model (qcore_pkg::vector_latency, 04 §3.2):
//   fp32_add / fp32_sub             : 2 stages  (VADD/VSUB      = 2)
//   fp32_max                        : 1 stage   (VMAX           = 2)
//   fp32_mul                        : 3 stages  (VMUL/VSCALE    = 3)
//   i32_to_f32 (dequant)            : 2 stages  (DEQUANT        = 5)
//   f32_to_i32 (quant)              : 3 stages  (QUANT          = 5)
//   fp32_to_bf16 (writeback)        : 1 stage
//   bf16_to_fp32 (input)            : combinational (bit repack, ~0 delay)
// Every stage therefore stays within the already-charged latency; throughput is
// 1 op/cycle (fully pipelined).  Semantics are bit-identical to the frozen
// softfloat_pkg functions (truncating FP32 add/mul, denormal flush, Inf/NaN
// propagation) — verified by the netlist-level spot check (see asic/report).
//
// Div/recip/rsqrt (Newton-Raphson) and transcendentals (exp/log2/sin/cos) are
// excluded: physical impl is iterative and vector_latency() already charges
// their multi-cycle latency.  FP16 conversions are omitted (secondary dtype).
// ============================================================================
`ifndef SYNTH_DATAPATH_SV
`define SYNTH_DATAPATH_SV

// ----------------------------------------------------------------------------
// fp32_add2 — FP32 add/sub, 2-stage pipeline (align | add+normalize).
// ----------------------------------------------------------------------------
module fp32_add2 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  // ---- stage 0 combinational ---------------------------------------------
  logic        sa_c, sb_c;
  logic [7:0]  ea_c, eb_c, er_c;
  logic [22:0] ma_c, mb_c;
  logic        a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
  logic [23:0] bigm_c, sml_c;
  logic        sb_big_c;
  logic [7:0]  diff_c;
  logic        nan_c, inf_c, zero_c;
  logic        a_inf_c, b_inf_c, a_zero_c, b_zero_c;

  assign sa_c = a[31]; assign sb_c = b[31];
  assign ea_c = a[30:23]; assign eb_c = b[30:23];
  assign ma_c = a[22:0];  assign mb_c = b[22:0];
  assign a_nan  = (ea_c == 8'hFF) && (ma_c != 0);
  assign b_nan  = (eb_c == 8'hFF) && (mb_c != 0);
  assign a_inf  = (ea_c == 8'hFF) && (ma_c == 0);
  assign b_inf  = (eb_c == 8'hFF) && (mb_c == 0);
  assign a_zero = (ea_c == 0);
  assign b_zero = (eb_c == 0);

  assign nan_c  = a_nan || b_nan;
  assign inf_c  = a_inf || b_inf;
  assign zero_c = a_zero || b_zero;
  assign a_inf_c = a_inf; assign b_inf_c = b_inf;
  assign a_zero_c = a_zero; assign b_zero_c = b_zero;


  always_comb begin
    if (ea_c >= eb_c) begin
      bigm_c = {1'b1, ma_c}; sml_c = {1'b1, mb_c};
      er_c = ea_c; sb_big_c = sa_c; diff_c = ea_c - eb_c;
    end else begin
      bigm_c = {1'b1, mb_c}; sml_c = {1'b1, ma_c};
      er_c = eb_c; sb_big_c = sb_c; diff_c = eb_c - ea_c;
    end
  end

  // ---- stage 0 registers --------------------------------------------------
  logic        sa0, sb0, op_eq0, sb_big0, nan0, inf0, zero0;
  logic        a_inf0, b_inf0, a_zero0, b_zero0;
  logic [23:0] bigm0, sml0;
  logic [7:0]  er0;
  logic [31:0] a0, b0;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sa0 <= 0; sb0 <= 0; op_eq0 <= 0; sb_big0 <= 0;
      nan0 <= 0; inf0 <= 0; zero0 <= 0;
      a_inf0 <= 0; b_inf0 <= 0; a_zero0 <= 0; b_zero0 <= 0;
      bigm0 <= 0; sml0 <= 0; er0 <= 0; a0 <= 0; b0 <= 0;
    end else begin
      sa0 <= sa_c; sb0 <= sb_c;
      op_eq0 <= (sa_c == sb_c);
      sb_big0 <= sb_big_c;
      nan0 <= nan_c; inf0 <= inf_c; zero0 <= zero_c;
      a_inf0 <= a_inf_c; b_inf0 <= b_inf_c; a_zero0 <= a_zero_c; b_zero0 <= b_zero_c;
      bigm0 <= bigm_c;
      sml0 <= (diff_c >= 8'd24) ? 24'b0 : (sml_c >> diff_c[4:0]);
      er0 <= er_c;
      a0 <= a; b0 <= b;
    end
  end

  // ---- stage 1 combinational ----------------------------------------------
  logic        sr1;
  logic [24:0] msum1;
  logic        exact_zero1;
  logic [24:0] msum_n1;
  logic [7:0]  er_n1;
  logic [31:0] y1;
  logic [4:0]  lead1;

  always_comb begin
    if (op_eq0) begin
      sr1 = sa0;
      msum1 = {1'b0, bigm0} + {1'b0, sml0};
    end else begin
      if (bigm0 >= sml0) begin sr1 = sb_big0;       msum1 = {1'b0, bigm0} - {1'b0, sml0}; end
      else               begin sr1 = ~sb_big0;      msum1 = {1'b0, sml0} - {1'b0, bigm0}; end
    end
  end

  always_comb begin
    lead1 = 5'd0;
    for (int k = 23; k >= 0; k--) if (msum1[k]) begin lead1 = k[4:0]; k = -1; end
  end
  assign exact_zero1 = ~msum1[24] && ~(|msum1[23:0]);

  always_comb begin
    if (msum1[24]) begin
      msum_n1 = msum1 >> 1;
      er_n1 = er0 + 8'd1;
    end else if (lead1 < 5'd23) begin
      msum_n1 = msum1 << (23 - lead1);
      er_n1 = er0 - (23 - lead1);
    end else begin
      msum_n1 = msum1;
      er_n1 = er0;
    end
  end

  always_comb begin
    y1 = 32'b0;
    if (nan0)                y1 = 32'h7FC00000;
    else if (a_inf0 || b_inf0) begin
      if (a_inf0 && b_inf0 && (sa0 != sb0)) y1 = 32'h7FC00000;
      else                                   y1 = a_inf0 ? a0 : b0;
    end
    else if (a_zero0)        y1 = b_zero0 ? (sa0 & sb0 ? 32'h80000000 : 32'b0) : b0;
    else if (b_zero0)        y1 = a0;
    else if (exact_zero1)    y1 = 32'b0;
    else if (er_n1 >= 8'hFF) y1 = sr1 ? 32'hFF800000 : 32'h7F800000;
    else                     y1 = {sr1, er_n1, msum_n1[22:0]};
  end

  always_ff @(posedge clk) begin
    if (!rst_n) y <= 0;
    else        y <= y1;
  end
endmodule

// ----------------------------------------------------------------------------
// fp32_mul3 — FP32 multiply, 3-stage pipeline (decode | 24x24 | normalize).
// ----------------------------------------------------------------------------
module fp32_mul3 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  logic        sa_c, sb_c;
  logic [7:0]  ea_c, eb_c;
  logic [22:0] ma_c, mb_c;
  logic        a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
  logic        nan_c, inf_c, zero_c;
  logic signed [9:0] er_c;

  assign sa_c = a[31]; assign sb_c = b[31];
  assign ea_c = a[30:23]; assign eb_c = b[30:23];
  assign ma_c = a[22:0];  assign mb_c = b[22:0];
  assign a_nan  = (ea_c == 8'hFF) && (ma_c != 0);
  assign b_nan  = (eb_c == 8'hFF) && (mb_c != 0);
  assign a_inf  = (ea_c == 8'hFF) && (ma_c == 0);
  assign b_inf  = (eb_c == 8'hFF) && (mb_c == 0);
  assign a_zero = (ea_c == 0);
  assign b_zero = (eb_c == 0);
  assign nan_c  = a_nan || b_nan;
  assign inf_c  = a_inf || b_inf;
  assign zero_c = a_zero || b_zero;
  assign er_c = $signed({1'b0, ea_c}) + $signed({1'b0, eb_c}) - 10'sd127;
  // ---- stage 0 registers ---------------------------------------------------
  logic        sr0, nan0, inf0, zero0;
  logic [22:0] ma0, mb0;
  logic signed [9:0] er0;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sr0 <= 0; nan0 <= 0; inf0 <= 0; zero0 <= 0;
      ma0 <= 0; mb0 <= 0; er0 <= 0;
    end else begin
      sr0 <= sa_c ^ sb_c;
      nan0 <= nan_c; inf0 <= inf_c; zero0 <= zero_c;
      ma0 <= ma_c; mb0 <= mb_c; er0 <= er_c;
    end
  end

  // ---- stage 1: 24x24 multiply (register only the top 25 product bits) -----
  logic        sr1, nan1, inf1, zero1;
  logic signed [9:0] er1;
  logic [47:0] prod_c;
  logic [24:0] prod_hi;

  assign prod_c = {1'b1, ma0} * {1'b1, mb0};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sr1 <= 0; nan1 <= 0; inf1 <= 0; zero1 <= 0;
      er1 <= 0; prod_hi <= 0;
    end else begin
      sr1 <= sr0; nan1 <= nan0; inf1 <= inf0; zero1 <= zero0;
      er1 <= er0;
      prod_hi <= prod_c[47:23];
    end
  end

  // ---- stage 2: normalize + special-case mux --------------------------------
  // prod_hi[24] = product bit 47 (normalization shift); mantissa = top 23 of
  // the shifted product (product[46:24] on overflow, else product[45:23]).
  logic [22:0] mout;
  logic signed [9:0] er_adj;
  logic [31:0] y2;

  always_comb begin
    if (prod_hi[24]) begin mout = prod_hi[23:1]; er_adj = er1 + 10'sd1; end
    else             begin mout = prod_hi[22:0]; er_adj = er1;          end
  end

  always_comb begin
    y2 = 32'b0;
    if (nan1)                    y2 = 32'h7FC00000;
    else if (inf1)               y2 = zero1 ? 32'h7FC00000 : (sr1 ? 32'hFF800000 : 32'h7F800000);
    else if (zero1)              y2 = sr1 ? 32'h80000000 : 32'b0;
    else if (er_adj >= 10'sd255) y2 = sr1 ? 32'hFF800000 : 32'h7F800000;
    else if (er_adj <= 10'sd0)   y2 = sr1 ? 32'h80000000 : 32'b0;
    else                         y2 = {sr1, er_adj[7:0], mout};
  end

  always_ff @(posedge clk) begin
    if (!rst_n) y <= 0;
    else        y <= y2;
  end
endmodule

module fp32_max1 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  logic [31:0] yc;
  always_comb begin
    if ((a[30:23] == 8'hFF && a[22:0] != 0))      yc = b;   // a NaN
    else if ((b[30:23] == 8'hFF && b[22:0] != 0)) yc = a;   // b NaN
    else if (a[31] && ~b[31])                     yc = b;
    else if (~a[31] && b[31])                     yc = a;
    else if (a[31])                               yc = (a[30:0] <= b[30:0]) ? a : b;
    else                                          yc = (a[30:0] >= b[30:0]) ? a : b;
  end
  always_ff @(posedge clk) begin
    if (!rst_n) y <= 0;
    else        y <= yc;
  end
endmodule

// ----------------------------------------------------------------------------
// i32_to_f32_2 — INT32 -> FP32 (RNE above 2^24), 2-stage pipeline.
//   stage 0: 2's-complement magnitude + leading-one detect + exponent
//   stage 1: align shift + RNE round + exponent bump
// ----------------------------------------------------------------------------
module i32_to_f32_2 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] i,
  output logic [31:0] y
);
  logic        s_c;
  logic [31:0] mag_c;
  logic [4:0]  lead_c;
  logic        zero_c;
  assign s_c   = i[31];
  assign mag_c = s_c ? ((~i) + 32'd1) : i;
  assign zero_c = (i == 0);
  always_comb begin
    lead_c = 5'd31;
    for (int k = 31; k >= 0; k--) if (mag_c[k]) begin lead_c = k[4:0]; k = -1; end
  end

  logic        s0, zero0;
  logic [31:0] mag0;
  logic [4:0]  lead0;
  logic [7:0]  e0;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s0 <= 0; zero0 <= 1; mag0 <= 0; lead0 <= 0; e0 <= 0;
    end else begin
      s0 <= s_c; zero0 <= zero_c; mag0 <= mag_c;
      lead0 <= lead_c; e0 <= 8'd127 + lead_c;
    end
  end

  // ---- stage 1 (module-scope temporaries, Yosys-2005-clean) ----------------
  logic [31:0] mag_shl;
  logic [31:0] mant;
  logic        g, st, up;
  logic [22:0] m_c;
  logic [7:0]  e_out;

  assign mag_shl = mag0 << (23 - lead0);
  always_comb begin
    mant = 32'b0;
    g = 0; st = 0; up = 0;
    m_c = 23'b0;
    e_out = e0;
    if (!zero0) begin
      if (lead0 <= 5'd23) begin
        m_c = mag_shl[22:0];
      end else begin
        mant = mag0 >> (lead0 - 23);
        g = mag0[lead0 - 24];
        st = 0;
        for (int k = 0; k < 8; k++) st = st | (mag0[k] & (k < (lead0 - 5'd24)));
        m_c = mant[22:0];
        up = g & (st | m_c[0]);
        if (up) begin
          if (m_c == 23'h7FFFFF) begin m_c = 23'b0; e_out = e0 + 8'd1; end
          else                   m_c = m_c + 23'd1;
        end
      end
    end
  end

  logic [31:0] yc;
  assign yc = zero0 ? 32'b0 : {s0, e_out, m_c};
  always_ff @(posedge clk) begin
    if (!rst_n) y <= 0;
    else        y <= yc;
  end
endmodule

// ----------------------------------------------------------------------------
// f32_to_i32_3 — FP32 -> INT32 (RNE, saturating), 3-stage pipeline.
//   stage 0: decode + shift amount + 1.m mantissa
//   stage 1: align shift + RNE round + sticky -> magnitude
//   stage 2: 2's-complement + saturation
// ----------------------------------------------------------------------------
module f32_to_i32_3 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] f,
  output logic [31:0] y
);
  logic        s_c;
  logic [7:0]  e_c;
  logic [22:0] m_c;
  logic        infnan_c, below1_c, sat31_c;
  logic [7:0]  shf_c;
  logic [23:0] v_c;
  assign s_c = f[31]; assign e_c = f[30:23]; assign m_c = f[22:0];
  assign infnan_c = (e_c == 8'hFF);
  assign below1_c = (e_c < 8'd127);
  assign sat31_c  = (e_c >= 8'd158);
  assign shf_c    = e_c - 8'd127;
  assign v_c      = {1'b1, m_c};

  logic        s0, infnan0, below10, sat310;
  logic [7:0]  shf0;
  logic [23:0] v0;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s0 <= 0; infnan0 <= 0; below10 <= 1; sat310 <= 0; shf0 <= 0; v0 <= 0;
    end else begin
      s0 <= s_c; infnan0 <= infnan_c; below10 <= below1_c; sat310 <= sat31_c;
      shf0 <= shf_c; v0 <= v_c;
    end
  end

  // ---- stage 1: shift + round -> magnitude ---------------------------------
  logic [32:0] v33;
  logic [32:0] v_shl;
  logic [32:0] tmp;
  logic        rb, sty, up;
  logic [31:0] mag_c;

  assign v33   = {9'b0, v0};
  assign v_shl = v33 << (shf0 - 8'd23);
  always_comb begin
    tmp = 33'b0;
    rb = 0; sty = 0; up = 0;
    mag_c = 32'b0;
    if (infnan0 || sat310) begin
      mag_c = 32'h7FFFFFFF;
    end else if (below10) begin
      mag_c = 32'b0;
    end else if (shf0 >= 8'd23) begin
      mag_c = v_shl[31:0];
    end else begin
      tmp = v33 >> (23 - shf0);
      rb  = v33[23 - shf0 - 1];
      sty = 0;
      for (int k = 0; k < 23; k++) sty = sty | (v33[k] & (k < (23 - shf0 - 1)));
      mag_c = tmp[31:0];
      up = rb & (sty | mag_c[0]);
      if (up) mag_c = mag_c + 32'd1;
    end
  end

  logic        s1, infnan1, sat311;
  logic [31:0] mag0;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s1 <= 0; infnan1 <= 0; sat311 <= 0; mag0 <= 0;
    end else begin
      s1 <= s0; infnan1 <= infnan0; sat311 <= sat310; mag0 <= mag_c;
    end
  end

  // ---- stage 2: 2's-complement + saturation ---------------------------------
  logic [31:0] yc;
  always_comb begin
    if (infnan1 || sat311) yc = s1 ? 32'h80000000 : 32'h7FFFFFFF;
    else                   yc = s1 ? ((~mag0) + 32'd1) : mag0;
  end
  always_ff @(posedge clk) begin
    if (!rst_n) y <= 0;
    else        y <= yc;
  end
endmodule

// ----------------------------------------------------------------------------
// top
// ----------------------------------------------------------------------------
module synth_datapath (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] a,
  input  logic [31:0] b,
  input  logic [31:0] i_in,
  input  logic [15:0] bf_in,
  output logic [31:0] add_o,
  output logic [31:0] sub_o,
  output logic [31:0] mul_o,
  output logic [31:0] max_o,
  output logic [31:0] i2f_o,
  output logic [31:0] f2i_o,
  output logic [15:0] f2b_o,
  output logic [31:0] b2f_o
);
  fp32_add2 u_add (.clk(clk), .rst_n(rst_n), .a(a), .b(b), .y(add_o));
  fp32_add2 u_sub (.clk(clk), .rst_n(rst_n), .a(a), .b({~b[31], b[30:0]}), .y(sub_o));
  fp32_mul3 u_mul (.clk(clk), .rst_n(rst_n), .a(a), .b(b), .y(mul_o));
  fp32_max1 u_max (.clk(clk), .rst_n(rst_n), .a(a), .b(b), .y(max_o));
  i32_to_f32_2 u_i2f (.clk(clk), .rst_n(rst_n), .i(i_in), .y(i2f_o));
  f32_to_i32_3 u_f2i (.clk(clk), .rst_n(rst_n), .f(a), .y(f2i_o));

  // bf16 -> fp32 (bit repack).
  assign b2f_o = {bf_in[15], bf_in[14:7], bf_in[6:0], 16'b0};

  // fp32 -> bf16 (RNE, denormal flush), 1-stage writeback.
  logic        f2b_s;
  logic [7:0]  f2b_e;
  logic [22:0] f2b_m;
  logic [6:0]  f2b_hi;
  logic        f2b_up;
  logic [7:0]  f2b_sum8;
  logic [7:0]  f2b_eo;
  logic [6:0]  f2b_mo;
  logic [15:0] f2b_c;

  always_comb begin
    f2b_s = a[31]; f2b_e = a[30:23]; f2b_m = a[22:0];
    f2b_hi = f2b_m[22:16];
    f2b_up = f2b_m[15] & (f2b_m[16] | (|f2b_m[14:0]) | f2b_hi[0]);
    f2b_sum8 = {1'b0, f2b_hi} + {7'b0, f2b_up};
    if (f2b_sum8[7]) begin f2b_eo = f2b_e + 8'd1; f2b_mo = 7'b0; end
    else             begin f2b_eo = f2b_e;        f2b_mo = f2b_sum8[6:0]; end
    f2b_c = 16'b0;
    if (f2b_e == 8'hFF) begin
      f2b_c = (f2b_m == 0) ? {f2b_s, 8'hFF, 7'b0} : {f2b_s, 8'hFF, f2b_m[22:16] | 7'b1};
    end else if (f2b_e == 0) begin
      f2b_c = {f2b_s, 16'b0};
    end else begin
      f2b_c = (f2b_eo >= 8'hFF) ? {f2b_s, 8'hFF, 7'b0} : {f2b_s, f2b_eo, f2b_mo};
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) f2b_o <= 0;
    else        f2b_o <= f2b_c;
  end
endmodule

`endif // SYNTH_DATAPATH_SV
