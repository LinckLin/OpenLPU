// ============================================================================
// kv_bfeed.sv — B' B-operand feed: quantized-KV dequant (+ on-the-fly RoPE).
//
// The streaming fusion (rotator-impl, plan v3): a BMM whose CD descriptor sets
// CD[31] (KV_QUANT) reads its B operand directly from the quantized KV slabs in
// HBM, dequantizes per element, and — for K (CD[30] ROTATE_K) — applies
// absolute-position RoPE, feeding the rotated value straight into the MAC
// array.  No BF16 staging write to SRAM (the retired staged dequant path).
//
// Numeric contract (bit-exact vs qsim/executor.py `_matrix` B-feed reference):
//   K dequant:  scale_c = bf16(s_q * k_norm[c]);  k_hat = bf16(q[c] * scale_c)
//   V dequant:  v_hat = bf16(q4[c] * s_v)
//   K rotate:   HF rotate_half in bf16 arithmetic (per-op rounding), absolute
//               position pos_base + n, theta = 1e6 (frozen ROPE_INVF LUT).
// The same softfloat core + rope_sincos used by the vector ROPE op and
// kv_quantdequant — no new numeric path.
//
// K path (rotate_k=1): precompute k_hat[128][N] into the kh4096x64 SRAM macro
//   (4096 x 64; word {ch[6:0], n[6:2]} packs 4 tokens x 16b bf16), then produce
//   each rotated channel row by one 64-bit read per word group (DC N=4 -> 2
//   reads/row; PF N=128 -> 64 reads/row, ch and ch^64 each).  V path
//   (rotate_k=0): stream each row (reduction dim = seq position) directly,
//   dequant per element.
//
// The flat khat[0:16383] / knorm[0:127] arrays are replaced by compiled SMIC28
// SRAM macros (kh4096x64 / kn128x16, 1-cycle read) so DC no longer infers 256
// parallel read ports from the 16K x 32 flat array.  Row production is
// serialized to one 64-bit read per cycle; row_valid pulse handshake is kept
// (default low + completion pulse, CP copies then deasserts row_req).
// ============================================================================
`ifndef KV_BFEED_SV
`define KV_BFEED_SV

`include "qcore_pkg.sv"
`include "softfloat.sv"
`include "rope_lut.sv"
`include "sram_macros.sv"

module kv_bfeed #(
  parameter int MAX_N = 128          // max tile N (positions / head_dim)
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,          // pulse: begin (K: precompute k_hat)
  input  logic        rotate_k,       // 1 = K fold dequant + rotate, 0 = V INT4
  input  logic [15:0] pos_base,       // absolute pos of row index 0
  input  logic [7:0]  N,              // tile N (K: positions; V: channels=128)
  input  logic [39:0] data_base,      // K/V data slab byte base (ARb)
  input  logic [39:0] scale_base,     // per-token scale record byte base
  input  logic [39:0] k_norm_base,    // static k_norm table byte base (K only)
  // shared memory read port (muxed by the CP)
  output logic        rd_sel,
  output logic [39:0] rd_addr,
  input  logic [7:0]  rd_data,
  // row production: CP asserts row_req with the reduction index, copies row[]
  input  logic [15:0] row_ch,         // reduction index kk (K: channel; V: pos)
  input  logic        row_req,        // pulse: produce row[row_ch]
  output logic [31:0] row [MAX_N],    // dequant(+rot) fp32 (bf16-exact)
  output logic        row_valid,      // high while row[] is stable/valid
  output logic        ready           // precompute done (K) / always (V)
);
  import qcore_pkg::*;
  import softfloat_pkg::*;
  import rope_lut_pkg::*;

  // ---- SRAM macros (SMIC28 SM18CA001 / SM18CD001, 1-cycle read) ------------
  logic [11:0] kh_addr;
  logic [63:0] kh_wdata, kh_rdata;
  logic        kh_en, kh_we;          // active-high enable / write
  kh4096x64 u_kh (
    .Q(kh_rdata), .CLK(clk),
    .CEN(~kh_en), .WEN(~kh_we),
    .A(kh_addr), .D(kh_wdata),
    .EMA(3'b011), .EMAW(2'b01), .EMAS(1'b0), .RET1N(1'b1)
  );

  logic [6:0]  kn_addr;
  logic [15:0] kn_wdata, kn_rdata;
  logic        kn_en, kn_we;
  kn128x16 u_kn (
    .q(kn_rdata), .clk(clk),
    .cen(~kn_en), .wen(~kn_we),
    .a(kn_addr), .d(kn_wdata),
    .ema(3'b011), .emaw(2'b01), .emas(1'b0), .ret1n(1'b1)
  );

  // ---- state / counters -----------------------------------------------------
  logic [3:0]  state;
  logic [7:0]  ch;            // channel counter (K)
  logic [7:0]  n;             // token/position counter
  logic [2:0]  sub;           // byte sub-step
  logic [7:0]  accb;          // low byte of a 2-byte read
  logic [31:0] s_f;           // s_v (fp32, bf16-exact) — V path
  logic [15:0] row_ch_r;      // registered row request index
  logic [7:0]  vbyte;         // V: current packed q4 byte
  logic [3:0]  vsub;          // V: 0=read byte, 1=low nibble, 2=high nibble
  logic [7:0]  vn;            // V: byte index (0..63)
  // K precompute storage
  logic [15:0] sq_reg [0:127]; // s_q per token (128 x 16b bf16, full register)
  logic [15:0] wbuf  [0:3];    // kh word-assembly buffer (4 lanes x 16b)
  logic        kn_wait;        // 1 = waiting 1 cyc for kn128x16 read to land
  // row production (serialized)
  logic [1:0]  prod_phase;     // 0=ch-word read in flight, 1=xi-word read, 2=rot
  logic [4:0]  prod_pn;        // word-group index 0..ngrp-1
  logic [63:0] word_ch;        // registered row_ch 64-bit word

  localparam logic [3:0]
    S_IDLE = 4'd0,
    S_KNORM = 4'd1,   // read static k_norm[c] (2B SRAM) -> kn128x16
    S_SQ   = 4'd2,    // read per-token s_q[n] (2B HBM) -> sq_reg[]
    S_Q    = 4'd3,    // read q[c][n] (1B HBM) -> k_hat -> kh4096x64 word
    S_VSCL = 4'd4,    // V: read s_v (2B HBM)
    S_VDAT = 4'd5,    // V: read packed q4 byte -> dequant -> row[]
    S_DONE = 4'd6,
    S_PROD = 4'd7;    // K: serialized rotated-row production

  // ceil(N/4) word groups per channel row
  wire [5:0] ngrp = (N + 8'd3) >> 2;

  assign ready = (rotate_k) ? (state == S_DONE) : 1'b1;

  // ---- combinational HBM read address --------------------------------------
  always_comb begin
    rd_sel = 1'b0; rd_addr = 40'b0;
    case (state)
      S_KNORM: begin rd_sel = 1'b0; rd_addr = k_norm_base + ch*2 + sub[0]; end
      S_SQ:    begin rd_sel = 1'b1;
                     rd_addr = scale_base + (({24'b0, pos_base} + {16'b0, n}) << 2) + sub[0]; end
      S_Q:     begin rd_sel = 1'b1; rd_addr = data_base + ({24'b0, n} << 7) + ch; end
      S_VSCL:  begin rd_sel = 1'b1;
                     rd_addr = scale_base + (({24'b0, pos_base} + row_ch_r) << 2) + 2 + sub[0]; end
      S_VDAT:  begin rd_sel = 1'b1;
                     rd_addr = data_base + (({24'b0, pos_base} + row_ch_r) << 6) + vn; end
      default: begin end
    endcase
  end

  // ---- combinational K dequant (kh16 = bf16 k_hat for current (ch, n)) -----
  logic [15:0] kh16;
  always_comb begin
    logic [31:0] q_f, scale_c;
    q_f     = i32_to_f32({{24{rd_data[7]}}, rd_data});
    scale_c = bf16_to_fp32(fp32_to_bf16(fp32_mul(bf16_to_fp32(sq_reg[n]), bf16_to_fp32(kn_rdata))));
    kh16    = fp32_to_bf16(fp32_mul(q_f, scale_c));
  end

  // ---- combinational kh word write data (wbuf + current lane + tail zero) --
  logic [63:0] kh_word;
  always_comb begin
    kh_word = {wbuf[3], wbuf[2], wbuf[1], wbuf[0]};
    case (n[1:0])
      2'd0: kh_word[15:0]  = kh16;
      2'd1: kh_word[31:16] = kh16;
      2'd2: kh_word[47:32] = kh16;
      2'd3: kh_word[63:48] = kh16;
    endcase
    if (n == N - 8'd1) begin
      case (n[1:0])
        2'd0: kh_word[63:16] = 48'b0;
        2'd1: kh_word[63:32] = 32'b0;
        2'd2: kh_word[63:48] = 16'b0;
        default: ;
      endcase
    end
  end

  // word write strobe: last lane of a group, or the tail token (post kn-wait)
  wire kh_word_we = (state == S_Q) && !kn_wait &&
                    ((n[1:0] == 2'd3) || (n == N - 8'd1));

  // ---- macro control: kh4096x64 --------------------------------------------
  always_comb begin
    kh_en = 1'b0; kh_we = 1'b0; kh_addr = 12'b0; kh_wdata = 64'b0;
    case (state)
      S_Q: if (kh_word_we) begin
        kh_en = 1'b1; kh_we = 1'b1;
        kh_addr  = {ch[6:0], n[6:2]};
        kh_wdata = kh_word;
      end
      S_PROD: begin
        kh_we = 1'b0;
        case (prod_phase)
          2'd0: begin kh_en = 1'b1; kh_addr = {row_ch_r[6:0], prod_pn[4:0]}; end
          2'd1: begin kh_en = 1'b1; kh_addr = {(row_ch_r[6:0] ^ 7'b1000000), prod_pn[4:0]}; end
          default: kh_en = 1'b0;
        endcase
      end
      default: ;
    endcase
  end

  // ---- macro control: kn128x16 ---------------------------------------------
  always_comb begin
    kn_en = 1'b0; kn_we = 1'b0; kn_addr = 7'b0; kn_wdata = 16'b0;
    case (state)
      S_KNORM: if (sub == 3'd1) begin
        kn_en = 1'b1; kn_we = 1'b1; kn_addr = ch[6:0]; kn_wdata = {rd_data, accb};
      end
      S_Q: begin
        kn_en = 1'b1; kn_we = 1'b0; kn_addr = ch[6:0];
      end
      default: ;
    endcase
  end

  // ---- FSM -----------------------------------------------------------------
  always_ff @(posedge clk) begin
    integer i;
    if (!rst_n) begin
      state <= S_IDLE; ch <= 8'b0; n <= 8'b0; sub <= 3'b0; accb <= 8'b0;
      s_f <= 32'b0; row_ch_r <= 16'b0; vbyte <= 8'b0;
      vsub <= 4'b0; vn <= 8'b0; row_valid <= 1'b0;
      kn_wait <= 1'b0; prod_phase <= 2'b0; prod_pn <= 5'b0;
    end else begin
      if (start) begin
        // (re-)precompute on a fresh tile/batch (works from any state)
        ch <= 8'b0; n <= 8'b0; sub <= 3'b0; accb <= 8'b0;
        kn_wait <= 1'b0; prod_phase <= 2'b0; prod_pn <= 5'b0;
        if (rotate_k) state <= S_KNORM;
        else          state <= S_DONE;   // V: no precompute
      end else begin
        row_valid <= 1'b0;   // pulse handshake: asserted only at row completion
        case (state)
          S_IDLE: begin end   // wait for start (override handles the entry)

          // ---------- K precompute: k_norm[c] (2B SRAM) -> kn128x16 ---------
          S_KNORM: begin
            if (sub == 3'd0) begin
              accb <= rd_data; sub <= 3'd1;
            end else begin
              // kn128x16 write is combinational (kn control above)
              sub <= 3'd0;
              if (ch == 8'd127) begin ch <= 8'b0; n <= 8'b0; state <= S_SQ; end
              else ch <= ch + 8'd1;
            end
          end

          // ---------- K precompute: s_q[n] (2B HBM) -> sq_reg[] -------------
          S_SQ: begin
            if (sub == 3'd0) begin
              accb <= rd_data; sub <= 3'd1;
            end else begin
              sq_reg[n] <= {rd_data, accb};
              sub <= 3'd0;
              if (n == N - 8'd1) begin
                n <= 8'b0; ch <= 8'b0; kn_wait <= 1'b1; state <= S_Q;
              end else n <= n + 8'd1;
            end
          end

          // ---------- K precompute: q[c][n] (1B HBM) -> k_hat -> kh4096x64 --
          S_Q: begin
            if (kn_wait) begin
              // kn128x16 read (knorm[ch]) landed this cycle
              kn_wait <= 1'b0;
            end else begin
              wbuf[n[1:0]] <= kh16;
              if (n == N - 8'd1) begin
                n <= 8'b0;
                if (ch == 8'd127) state <= S_DONE;
                else begin ch <= ch + 8'd1; kn_wait <= 1'b1; end
              end else n <= n + 8'd1;
            end
          end

          // ---------- row production ----------------------------------------
          S_DONE: begin
            if (rotate_k) begin
              // K: serialized rotated row (pair = ch ^ 64)
              if (row_req) begin
                row_ch_r <= row_ch;
                prod_pn <= 5'b0;
                prod_phase <= 2'd0;
                state <= S_PROD;
              end
            end else begin
              // V: on row_req, read s_v then packed q4 data for the row
              if (row_req) begin
                row_ch_r <= row_ch; sub <= 3'b0; accb <= 8'b0; state <= S_VSCL;
              end
            end
          end

          S_PROD: begin
            case (prod_phase)
              2'd0: begin
                // ch-word read issued last cycle is in flight
                prod_phase <= 2'd1;
              end
              2'd1: begin
                word_ch <= kh_rdata;      // ch-word landed
                prod_phase <= 2'd2;
              end
              default: begin
                // xi-word landed (kh_rdata); rotate the 4-token group
                for (i = 0; i < 4; i++) begin
                  logic [31:0] xr, xi, ang, cos_d, sin_d, t1, t2, r;
                  logic [63:0] sc;
                  logic [15:0] posn;
                  integer token;
                  token = {27'b0, prod_pn, 2'b0} + i;   // prod_pn*4 + i
                  if (token < N) begin
                    xr  = bf16_to_fp32(word_ch[i*16 +: 16]);
                    xi  = bf16_to_fp32(kh_rdata[i*16 +: 16]);
                    posn = pos_base + token[15:0];
                    ang = fp32_mul(i32_to_f32({16'b0, posn}), ROPE_INVF[row_ch_r[5:0]]);
                    sc  = rope_sincos(ang);
                    cos_d = bf16_to_fp32(fp32_to_bf16(sc[63:32]));
                    sin_d = bf16_to_fp32(fp32_to_bf16(sc[31:0]));
                    t1  = bf16_to_fp32(fp32_to_bf16(fp32_mul(xr, cos_d)));
                    t2  = bf16_to_fp32(fp32_to_bf16(fp32_mul(xi, sin_d)));
                    r   = row_ch_r[6] ? fp32_add(t1, t2) : fp32_sub(t1, t2);
                    row[token] = bf16_to_fp32(fp32_to_bf16(r));
                  end
                end
                if (prod_pn == ngrp - 1) begin
                  row_valid <= 1'b1;
                  prod_pn <= 5'b0; prod_phase <= 2'd0;
                  state <= S_DONE;
                end else begin
                  prod_pn <= prod_pn + 5'd1;
                  prod_phase <= 2'd0;
                end
              end
            endcase
          end

          S_VSCL: begin
            if (sub == 3'd0) begin
              accb <= rd_data; sub <= 3'd1;
            end else begin
              s_f <= bf16_to_fp32({rd_data, accb});
              sub <= 3'b0; vn <= 8'b0; vsub <= 4'b0; state <= S_VDAT;
            end
          end

          S_VDAT: begin
            logic [3:0] nibble;
            logic [31:0] nib, q4f;
            // vn = byte index 0..63; byte -> channels 2*vn (low), 2*vn+1 (high)
            if (vsub == 4'd0) begin
              vbyte <= rd_data; vsub <= 4'd1;
            end else if (vsub == 4'd1) begin
              nibble = vbyte[3:0];
              nib    = {28'b0, nibble};
              if (nibble[3]) nib = nib | 32'hFFFFFFF0;
              q4f    = i32_to_f32(nib);
              row[({7'b0, vn} << 1)] = bf16_to_fp32(fp32_to_bf16(fp32_mul(q4f, s_f)));
              vsub <= 4'd2;
            end else begin
              nibble = vbyte[7:4];
              nib    = {28'b0, nibble};
              if (nibble[3]) nib = nib | 32'hFFFFFFF0;
              q4f    = i32_to_f32(nib);
              row[({7'b0, vn} << 1) + 1] = bf16_to_fp32(fp32_to_bf16(fp32_mul(q4f, s_f)));
              if (vn == 8'd63) begin
                row_valid <= 1'b1; vn <= 8'b0; vsub <= 4'b0; state <= S_DONE;
              end else begin
                vn <= vn + 8'd1; vsub <= 4'b0;
              end
            end
          end

          default: state <= S_IDLE;
        endcase
      end
    end
  end

endmodule
`endif // KV_BFEED_SV
