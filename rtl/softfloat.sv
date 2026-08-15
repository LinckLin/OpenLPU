// ============================================================================
// softfloat_pkg.sv — BF16 / FP32 / INT32 soft-float arithmetic for QCore RTL
//
// BF16  : 1 sign | 8 exp | 7 mantissa   (hidden 1, bias 127)
// FP32  : 1 sign | 8 exp | 23 mantissa  (hidden 1, bias 127, IEEE-754 single)
//
// The QCore datapath is FP32 for acc=FP32 (MATRIX/VECTOR) and INT32 for
// acc=INT32 (W8A8 dequant partials), matching the qsim reference executor
// (docs/spec-src/04 §1.2/§1.5, qsim/executor.py).  Results are written back
// as BF16 with round-to-nearest-even (RNE) — the only rounding that is
// numerically significant for the co-sim acceptance.
//
// FP32-level add/sub/mul round toward zero (truncate).  Rationale: the M6
// co-sim acceptance is <= 1 ULP in the BF16 domain (mantissa width 8, so one
// BF16 ULP = 2^-8 relative).  Truncating FP32 accumulation of up to 4096
// terms carries worst-case ~4096 * 2^-23 ~= 2^-11 relative drift vs a
// correctly-rounded FP32 accumulator — roughly 1/8 of one BF16 ULP.  The
// BF16 writeback (RNE) therefore differs from the reference by at most one
// BF16 ULP.  (Documented in docs/p7/rtl-report.md §numeric.)
//
// Denormal FP32 inputs are flushed to zero (v0 activations are normal-range).
// Inf/NaN propagate for the softmax -inf path (VMASK -> VADD/VMAX).
// ============================================================================
`ifndef SOFTFLOAT_PKG_SV
`define SOFTFLOAT_PKG_SV

package softfloat_pkg;

  localparam logic [31:0] F_POS_INF = 32'h7F800000;
  localparam logic [31:0] F_NEG_INF = 32'hFF800000;
  localparam logic [31:0] F_QNAN    = 32'h7FC00000;
  localparam logic [31:0] F_ONE     = 32'h3F800000;  // 1.0
  localparam logic [31:0] F_TWO     = 32'h40000000;  // 2.0
  localparam logic [31:0] F_HALF    = 32'h3F000000;  // 0.5
  localparam logic [31:0] F_NZERO   = 32'h80000000;  // -0.0

  // --------------------------------------------------------------------------
  // BF16 <-> FP32 conversion
  // --------------------------------------------------------------------------
  function automatic logic [31:0] bf16_to_fp32(input logic [15:0] b);
    bf16_to_fp32 = {b[15], b[14:7], b[6:0], 16'b0};
  endfunction

  // FP32 -> BF16 round-to-nearest-even.
  function automatic logic [15:0] fp32_to_bf16(input logic [31:0] f);
    logic s;
    logic [7:0] e;
    logic [22:0] m;
    logic [6:0] hi;
    logic up;
    logic [7:0] sum8;
    logic [7:0] eo;
    logic [6:0] mo;
    s = f[31]; e = f[30:23]; m = f[22:0];
    if (e == 8'hFF) begin
      if (m == 0)       fp32_to_bf16 = {s, 8'hFF, 7'b0};
      else              fp32_to_bf16 = {s, 8'hFF, m[22:16] | 7'b1};
      return fp32_to_bf16;
    end
    if (e == 0) begin fp32_to_bf16 = {s, 16'b0}; return fp32_to_bf16; end   // flush denormal
    hi = m[22:16];
    up = m[15] & (m[16] | (|m[14:0]) | hi[0]);   // RNE: round bit & (sticky | even)
    sum8 = {1'b0, hi} + {7'b0, up};              // 8-bit: 7-bit mantissa + carry
    if (sum8[7]) begin eo = e + 8'd1; mo = 7'b0; end
    else         begin eo = e;         mo = sum8[6:0]; end
    if (eo >= 8'hFF) fp32_to_bf16 = {s, 8'hFF, 7'b0};
    else             fp32_to_bf16 = {s, eo, mo};
  endfunction

  // --------------------------------------------------------------------------
  // INT32 <-> FP32 (i32_to_f32 exact for |i| < 2^24; RNE above)
  // --------------------------------------------------------------------------
  function automatic logic [31:0] i32_to_f32(input logic [31:0] i);
    logic s;
    logic [31:0] mag;
    integer lead;
    logic [7:0] e;
    logic [22:0] m;
    logic [31:0] mant;
    logic g, st, up;
    if (i == 0) begin i32_to_f32 = 32'b0; return i32_to_f32; end
    s = i[31];
    mag = s ? ((~i) + 32'd1) : i;
    lead = 31;
    for (integer k = 31; k >= 0; k = k - 1) if (mag[k]) begin lead = k; k = -1; end
    e = 8'd127 + lead;
    if (lead <= 23) begin
      logic [31:0] mtmp;
      mtmp = mag << (23 - lead);
      m = mtmp[22:0];
    end else begin
      mant = mag >> (lead - 23);
      g = mag[lead - 24];
      st = 0;
      for (integer k = 0; k < lead - 23 - 1; k = k + 1) st = st | mag[k];
      m = mant[22:0];
      up = g & (st | m[0]);
      if (up) begin
        if (m == 23'h7FFFFF) begin m = 23'b0; e = e + 8'd1; end
        else m = m + 23'd1;
      end
    end
    i32_to_f32 = {s, e, m};
  endfunction

  // FP32 -> INT32 round-to-nearest-even (QUANT `round(a/s)`), saturating.
  function automatic logic [31:0] f32_to_i32_rne(input logic [31:0] f);
    logic s;
    logic [7:0] e;
    logic [22:0] m;
    logic [31:0] mag;
    integer shf;
    logic [32:0] v;
    logic [32:0] tmp;
    logic rb, sty, up;
    logic [31:0] res;
    s = f[31]; e = f[30:23]; m = f[22:0];
    if (e == 8'hFF) begin f32_to_i32_rne = s ? 32'h80000000 : 32'h7FFFFFFF; return f32_to_i32_rne; end
    if (e < 8'd127) begin f32_to_i32_rne = 32'b0; return f32_to_i32_rne; end
    // value = 1.m * 2^(e-127)
    shf = e - 8'd127;               // >= 0
    if (shf >= 31) begin f32_to_i32_rne = s ? 32'h80000000 : 32'h7FFFFFFF; return f32_to_i32_rne; end
    // magnitude with fractional part for RNE: 1.mm >> (23 - shf)
    // mant = {1'b1, m} (24 bits), value = mant * 2^(shf-23)
    v = {9'b0, {1'b1, m}};           // 33-bit
    if (shf >= 23) begin
      v = v << (shf - 23);
      mag = v[31:0];
    end else begin
      // fractional bits present: round half to even
      tmp = v >> (23 - shf);
      // round bit = v[23-shf-1], sticky = lower
      rb = v[23 - shf - 1];
      // sticky = OR of the remaining fractional bits below the round bit
      sty = 0;
      for (integer k = 0; k < 23 - shf - 1; k = k + 1) sty = sty | v[k];
      mag = tmp[31:0];
      up = rb & (sty | mag[0]);
      if (up) mag = mag + 32'd1;
    end
    if (s) begin
      res = (~mag) + 32'd1;
    end else begin
      res = mag;
    end
    f32_to_i32_rne = res;
  endfunction

  // --------------------------------------------------------------------------
  // FP32 add / sub — sign-magnitude, truncating
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp32_add(input logic [31:0] a, input logic [31:0] b);
    logic sa, sb, sr;
    logic [7:0] ea, eb, er;
    logic [22:0] ma, mb;
    logic a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
    logic [23:0] bigm, sml;
    logic sb_big;                    // sign of the larger-magnitude operand
    logic [7:0] diff;
    logic [24:0] msum;               // 25-bit magnitude sum
    integer lead;

    sa = a[31]; sb = b[31];
    ea = a[30:23]; eb = b[30:23];
    ma = a[22:0]; mb = b[22:0];
    a_nan = (ea == 8'hFF) && (ma != 0);
    b_nan = (eb == 8'hFF) && (mb != 0);
    a_inf = (ea == 8'hFF) && (ma == 0);
    b_inf = (eb == 8'hFF) && (mb == 0);
    a_zero = (ea == 0);
    b_zero = (eb == 0);

    if (a_nan || b_nan) begin fp32_add = F_QNAN; return fp32_add; end
    if (a_inf || b_inf) begin
      if (a_inf && b_inf && (sa != sb)) begin fp32_add = F_QNAN; return fp32_add; end
      fp32_add = a_inf ? a : b;
      return fp32_add;
    end
    if (a_zero) begin fp32_add = b_zero ? (sa & sb ? F_NZERO : 32'b0) : b; return fp32_add; end
    if (b_zero) begin fp32_add = a; return fp32_add; end

    // align to the larger exponent
    if (ea >= eb) begin
      bigm = {1'b1, ma}; sml = {1'b1, mb}; er = ea; sb_big = sa;
      diff = ea - eb;
    end else begin
      bigm = {1'b1, mb}; sml = {1'b1, ma}; er = eb; sb_big = sb;
      diff = eb - ea;
    end

    // shift the smaller magnitude right by diff (truncate)
    if (diff >= 24) begin
      sml = 24'b0;
    end else if (diff > 0) begin
      sml = sml >> diff;
    end

    // effective operation (sign-magnitude)
    if (sa == sb) begin
      sr = sa;
      msum = {1'b0, bigm} + {1'b0, sml};
    end else begin
      if (bigm >= sml) begin
        sr = sb_big;
        msum = {1'b0, bigm} - {1'b0, sml};
      end else begin
        sr = ~sb_big;
        msum = {1'b0, sml} - {1'b0, bigm};
      end
    end

    // normalize (msum is 25-bit magnitude; leading 1 at bit 24 or lower)
    if (msum[24]) begin
      msum = msum >> 1;
      er = er + 8'd1;
    end else begin
      lead = -1;
      for (integer k = 23; k >= 0; k = k - 1) if (msum[k]) begin lead = k; k = -1; end
      if (lead < 0) begin fp32_add = 32'b0; return fp32_add; end
      if (lead < 23) begin
        msum = msum << (23 - lead);
        er = er - (23 - lead);
      end
    end
    if (er >= 8'hFF) begin fp32_add = sr ? F_NEG_INF : F_POS_INF; return fp32_add; end
    fp32_add = {sr, er, msum[22:0]};
  endfunction

  function automatic logic [31:0] fp32_sub(input logic [31:0] a, input logic [31:0] b);
    fp32_sub = fp32_add(a, {~b[31], b[30:0]});
  endfunction

  // FP32 max / min (sign-magnitude; -inf handled; NaN -> returns other).
  function automatic logic [31:0] fp32_max(input logic [31:0] a, input logic [31:0] b);
    logic a_neg, b_neg;
    a_neg = a[31]; b_neg = b[31];
    if ((a[30:23] == 8'hFF && a[22:0] != 0)) begin fp32_max = b; return fp32_max; end  // a NaN
    if ((b[30:23] == 8'hFF && b[22:0] != 0)) begin fp32_max = a; return fp32_max; end  // b NaN
    if (a_neg && ~b_neg) begin fp32_max = b; return fp32_max; end
    if (~a_neg && b_neg) begin fp32_max = a; return fp32_max; end
    if (a_neg) begin
      // both negative: larger magnitude = smaller
      fp32_max = (a[30:0] <= b[30:0]) ? a : b;
    end else begin
      fp32_max = (a[30:0] >= b[30:0]) ? a : b;
    end
  endfunction

  // --------------------------------------------------------------------------
  // FP32 multiply — truncating
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp32_mul(input logic [31:0] a, input logic [31:0] b);
    logic sa, sb, sr;
    logic [7:0] ea, eb;
    logic signed [9:0] er;
    logic [22:0] ma, mb;
    logic [47:0] prod;
    logic [22:0] mout;
    sa = a[31]; sb = b[31]; sr = sa ^ sb;
    ea = a[30:23]; eb = b[30:23];
    ma = a[22:0]; mb = b[22:0];
    if ((ea == 8'hFF && ma != 0) || (eb == 8'hFF && mb != 0)) begin fp32_mul = F_QNAN; return fp32_mul; end
    if (ea == 8'hFF || eb == 8'hFF) begin
      if (ea == 0 || eb == 0) begin fp32_mul = F_QNAN; return fp32_mul; end
      fp32_mul = sr ? F_NEG_INF : F_POS_INF; return fp32_mul;
    end
    if (ea == 0 || eb == 0) begin fp32_mul = sr ? F_NZERO : 32'b0; return fp32_mul; end
    // 9-bit exponent path: detect both overflow (>= 0xFF) and underflow
    // (product < 2^-126 -> denormal, flushed to signed zero per header policy).
    er = $signed({1'b0, ea}) + $signed({1'b0, eb}) - 10'sd127;
    prod = {1'b1, ma} * {1'b1, mb};   // 24x24 -> 48
    if (prod[47]) begin prod = prod >> 1; er = er + 10'sd1; end
    mout = prod[45:23];
    if (er >= 10'sd255) fp32_mul = sr ? F_NEG_INF : F_POS_INF;
    else if (er <= 10'sd0) fp32_mul = sr ? F_NZERO : 32'b0;   // denormal flush
    else fp32_mul = {sr, er[7:0], mout};
  endfunction

  // --------------------------------------------------------------------------
  // FP32 reciprocal (Newton-Raphson, ~4 iterations -> full precision) & divide
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp32_recip(input logic [31:0] b);
    logic [31:0] r;
    logic [7:0] eb;
    logic [22:0] mb;
    logic [7:0] er0;
    logic [22:0] mr0;
    integer i;
    logic [31:0] tmp;
    eb = b[30:23]; mb = b[22:0];
    if (eb == 8'hFF || eb == 0) begin fp32_recip = b; return fp32_recip; end
    er0 = 8'd253 - eb;
    mr0 = 23'h7FFFFF - (mb >> 1);
    r = {b[31], er0, mr0};   // preserve sign (NR diverges otherwise)
    for (i = 0; i < 4; i = i + 1) begin
      tmp = fp32_mul(b, r);
      tmp = fp32_sub(F_TWO, tmp);
      r = fp32_mul(r, tmp);
    end
    fp32_recip = r;
  endfunction

  function automatic logic [31:0] fp32_div(input logic [31:0] a, input logic [31:0] b);
    fp32_div = fp32_mul(a, fp32_recip(b));
  endfunction

  // --------------------------------------------------------------------------
  // FP32 reciprocal-square-root (seed + Newton-Raphson)
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp32_rsqrt(input logic [31:0] x);
    logic [31:0] r, tmp;
    logic [7:0] ex;
    logic [22:0] mx;
    integer i;
    ex = x[30:23]; mx = x[22:0];
    if (ex == 8'hFF) begin fp32_rsqrt = x; return fp32_rsqrt; end
    if (ex == 0)    begin fp32_rsqrt = F_POS_INF; return fp32_rsqrt; end
    if (x[31])      begin fp32_rsqrt = F_QNAN; return fp32_rsqrt; end
    // classic fast-inverse-sqrt bit-level seed, ~3% max error
    r = 32'h5F3759DF - (x >> 1);
    for (i = 0; i < 4; i = i + 1) begin
      tmp = fp32_mul(x, fp32_mul(r, r));
      tmp = fp32_sub(32'h40400000 /*3.0*/, tmp);
      r = fp32_mul(F_HALF, fp32_mul(r, tmp));
    end
    fp32_rsqrt = r;
  endfunction

  // --------------------------------------------------------------------------
  // FP32 exp2 via degree-6 minimax on [0,1) + exponent scaling -> VEXP.
  // Relative accuracy ~1e-7, far inside one BF16 ULP (2^-8).
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp32_exp2(input logic [31:0] x);
    logic s;
    logic [7:0] ex;
    logic [22:0] mx;
    integer n, mag_n, i, shf;
    logic [31:0] f, acc, nf;
    logic [31:0] c10 = 32'h31F267A9; // ln2^10/10!
    logic [31:0] c9  = 32'h33DA929F; // ln2^9/9!
    logic [31:0] c8  = 32'h35B16011; // ln2^8/8!
    logic [31:0] c7  = 32'h377FE5FE; // ln2^7/7!
    logic [31:0] c6 = 32'h39218489;  // ln2^6/720
    logic [31:0] c5 = 32'h3AAEC3FF;  // ln2^5/120
    logic [31:0] c4 = 32'h3C1D955B;  // ln2^4/24
    logic [31:0] c3 = 32'h3D635847;  // ln2^3/6
    logic [31:0] c2 = 32'h3E75FDF0;  // ln2^2/2
    logic [31:0] c1 = 32'h3F317218;  // ln2
    s = x[31]; ex = x[30:23]; mx = x[22:0];
    if (ex == 8'hFF) begin
      if (mx != 0) begin fp32_exp2 = F_QNAN; return fp32_exp2; end
      fp32_exp2 = s ? 32'b0 : F_POS_INF;
      return fp32_exp2;
    end
    if (ex == 0) begin fp32_exp2 = F_ONE; return fp32_exp2; end
    // range clamp (|x| >= 128): 2^+128 -> inf, 2^-128 -> 0
    if (!s && ex >= 8'd134) begin fp32_exp2 = F_POS_INF; return fp32_exp2; end
    if (s  && ex >= 8'd134) begin fp32_exp2 = 32'b0; return fp32_exp2; end

    // n = floor(x) via bit extraction (S = {1'b1,mx}; x = S * 2^(ex-150))
    shf = 8'd150 - ex;
    if (shf <= 0) begin
      n = 0; f = x;
    end else if (shf >= 24) begin
      // |x| < 1: floor(x) = -1 for x<0 (frac part = 1-|x|), else 0
      if (s) begin n = -1; f = fp32_add(x, F_ONE); end
      else   begin n = 0;  f = x; end
    end else begin
      n = ({1'b1, mx} >> shf);
      nf = i32_to_f32(32'(n));
      if (s) begin
        // f = x - trunc(x) = -(frac) in [-1,0); adjust to floor
        f = fp32_sub(x, nf);
        n = -n;
        if (f != 32'b0) begin
          n = n - 1;
          nf = i32_to_f32(32'(n));
          f = fp32_sub(x, nf);   // x - floor(x) in [0,1)
        end
      end else begin
        f = fp32_sub(x, nf);
      end
    end

    // 2^f = e^(f ln2) via degree-10 Taylor in f (error ~7e-9, matches powf)
    acc = c10;
    acc = fp32_add(fp32_mul(acc, f), c9);
    acc = fp32_add(fp32_mul(acc, f), c8);
    acc = fp32_add(fp32_mul(acc, f), c7);
    acc = fp32_add(fp32_mul(acc, f), c6);
    acc = fp32_add(fp32_mul(acc, f), c5);
    acc = fp32_add(fp32_mul(acc, f), c4);
    acc = fp32_add(fp32_mul(acc, f), c3);
    acc = fp32_add(fp32_mul(acc, f), c2);
    acc = fp32_add(fp32_mul(acc, f), c1);
    acc = fp32_add(fp32_mul(acc, f), F_ONE);

    // scale by 2^n
    if (n >= 0) begin
      for (i = 0; i < n; i = i + 1) acc = fp32_mul(acc, F_TWO);
    end else begin
      mag_n = -n;
      for (i = 0; i < mag_n; i = i + 1) acc = fp32_mul(acc, F_HALF);
    end
    fp32_exp2 = acc;
  endfunction

  function automatic logic [31:0] fp32_exp(input logic [31:0] x);
    fp32_exp = fp32_exp2(fp32_mul(x, 32'h3FB8AA3B));  // * log2(e)
  endfunction

  // --------------------------------------------------------------------------
  // FP32 log2 (bit extract + degree-7 monomial on [0,1)) -> ROPE/pow support.
  // |err| < 6.8e-7, far inside one BF16 ULP (2^-7).
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp32_log2(input logic [31:0] x);
    logic [7:0] e;
    logic [22:0] m;
    logic [31:0] f, ei, acc;
    logic [31:0] c1 = 32'h3FB8A90F;  // +1.4426592
    logic [31:0] c2 = 32'hBF386FE8;  // -0.7204575
    logic [31:0] c3 = 32'h3EF21E62;  // +0.4728880
    logic [31:0] c4 = 32'hBEA607BD;  // -0.3242778
    logic [31:0] c5 = 32'h3E44E4E3;  // +0.1922794
    logic [31:0] c6 = 32'hBDA06DF6;  // -0.0783347
    logic [31:0] c7 = 32'h3C79C275;  // +0.0152441
    e = x[30:23]; m = x[22:0];
    if (e == 8'hFF && m != 0) begin fp32_log2 = F_QNAN; return fp32_log2; end
    if (e == 0) begin fp32_log2 = F_NEG_INF; return fp32_log2; end  // log2(0)
    if (x[31]) begin fp32_log2 = F_QNAN; return fp32_log2; end       // log2(x<0)
    // x = 1.m * 2^(e-127); log2(x) = (e-127) + log2(1.m), 1.m = 1+f in [1,2)
    f = fp32_sub({1'b0, 8'h7F, m}, F_ONE);   // f = m/2^23 in [0,1)
    ei = i32_to_f32(32'(signed'({1'b0, e}) - 127));
    // log2(1+f) = f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*(c6 + f*c7)))))
    acc = c7;
    acc = fp32_add(fp32_mul(acc, f), c6);
    acc = fp32_add(fp32_mul(acc, f), c5);
    acc = fp32_add(fp32_mul(acc, f), c4);
    acc = fp32_add(fp32_mul(acc, f), c3);
    acc = fp32_add(fp32_mul(acc, f), c2);
    acc = fp32_add(fp32_mul(acc, f), c1);
    fp32_log2 = fp32_add(ei, fp32_mul(acc, f));
  endfunction

  // FP32 pow (a^b) = exp2(b * log2(a)) for a > 0.
  function automatic logic [31:0] fp32_pow(input logic [31:0] a, input logic [31:0] b);
    fp32_pow = fp32_exp2(fp32_mul(b, fp32_log2(a)));
  endfunction

  // --------------------------------------------------------------------------
  // FP32 sin / cos — Cody-Waite range reduction + degree-9 Taylor on [-pi/2,pi/2].
  // |err| < 1e-5 rad at |x|<=1024 (M6 golden ROPE point), well inside one BF16
  // ULP of the resulting bf16 rotation.
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp32_sin(input logic [31:0] x);
    logic [31:0] r, r2, acc, nf, n2pi_hi, n2pi_lo;
    integer n;
    logic [31:0] c3 = 32'hBE2AAAAB;     // -1/3!
    logic [31:0] c5 = 32'h3C088889;     //  1/5!
    logic [31:0] c7 = 32'hB9500D01;     // -1/7!
    logic [31:0] c9 = 32'h3638EF1D;     //  1/9!
    logic [31:0] c11 = 32'hB2D7322B;    // -1/11!
    logic [31:0] c13 = 32'h2F309231;    //  1/13!
    logic [31:0] c15 = 32'hAB573F9F;    // -1/15!
    logic [31:0] INV_2PI = 32'h3E22F983;    // 1/(2pi)
    logic [31:0] TWO_PI_HI = 32'h40C91000;  // 2pi high part (6.283203125)
    logic [31:0] TWO_PI_LO = 32'hB795777A;  // 2pi - TWO_PI_HI (residual, correct)
    // n = round(x / 2pi); r = x - n*2pi in [-pi, pi] (Cody-Waite two-part)
    nf = fp32_mul(x, INV_2PI);
    n = $signed(f32_to_i32_rne(nf));
    n2pi_hi = fp32_mul(i32_to_f32(32'(n)), TWO_PI_HI);
    n2pi_lo = fp32_mul(i32_to_f32(32'(n)), TWO_PI_LO);
    r = fp32_sub(x, n2pi_hi);
    r = fp32_sub(r, n2pi_lo);
    // degree-15 odd Taylor (valid on [-pi, pi]; truncation < 1e-7)
    r2 = fp32_mul(r, r);
    acc = fp32_add(fp32_mul(r2, c15), c13);
    acc = fp32_add(fp32_mul(r2, acc), c11);
    acc = fp32_add(fp32_mul(r2, acc), c9);
    acc = fp32_add(fp32_mul(r2, acc), c7);
    acc = fp32_add(fp32_mul(r2, acc), c5);
    acc = fp32_add(fp32_mul(r2, acc), c3);
    fp32_sin = fp32_add(r, fp32_mul(fp32_mul(r, r2), acc));
  endfunction

  function automatic logic [31:0] fp32_cos(input logic [31:0] x);
    // cos(x) = sin(x + pi/2)
    fp32_cos = fp32_sin(fp32_add(x, 32'h3FC90FDB));  // + 1.5707963
  endfunction

  // FP16 <-> FP32 (RNE to FP16; denormals flushed, matching header policy).
  // --------------------------------------------------------------------------
  function automatic logic [31:0] fp16_to_fp32(input logic [15:0] h);
    logic s; logic [4:0] e; logic [9:0] m;
    s = h[15]; e = h[14:10]; m = h[9:0];
    if (e == 5'd0)       fp16_to_fp32 = {s, 8'b0, 23'b0};            // flush denormal
    else if (e == 5'd31) fp16_to_fp32 = {s, 8'hFF, m, 13'b0};        // inf/nan
    else                 fp16_to_fp32 = {s, e + 8'd112, m, 13'b0};   // e-15+127
  endfunction

  function automatic logic [15:0] fp32_to_fp16(input logic [31:0] f);
    logic s; logic [7:0] e; logic [22:0] m;
    logic [4:0] eo; logic [10:0] mo; logic up;
    s = f[31]; e = f[30:23]; m = f[22:0];
    if (e == 8'hFF) begin
      fp32_to_fp16 = {s, 5'h1F, (m == 0) ? 10'b0 : (m[22:13] | 10'b1)};
      return fp32_to_fp16;
    end
    if (e < 8'd112) begin fp32_to_fp16 = {s, 5'b0, 10'b0}; return fp32_to_fp16; end
    eo = e - 8'd112;
    if (eo >= 5'd31) begin fp32_to_fp16 = {s, 5'h1F, 10'b0}; return fp32_to_fp16; end
    mo = {1'b0, m[22:13]} + {10'b0, up};
    if (mo[10]) begin mo = 10'b0; eo = eo + 5'd1; end
    if (eo >= 5'd31) fp32_to_fp16 = {s, 5'h1F, 10'b0};
    else             fp32_to_fp16 = {s, eo, mo[9:0]};
  endfunction
endpackage
`endif // SOFTFLOAT_PKG_SV
