// ============================================================================
// matrix_engine.sv — Matrix Engine numeric core (co-sim model, clocked K-stream).
//
// A single GEMM/GEMV/BMM tile: C[M x N] = A[M x K] x B[K x N] (+ post-process).
// The 128x128 dual-MAC systolic array (04 §1) is the synthesis target (P8/P9);
// this co-sim model reproduces the *exact* numeric result of the qsim reference
// executor, and the Command Processor charges the frozen per-instruction cycle
// count (qcore_pkg matrix_pf_cycles / matrix_dc_batch_cycles).
//
// Clocked K-stream (P7 perf fix): one MAC per clock cycle, driven by `step`
// pulses from the CP.  The M x N accumulator, the per-group INT32 partials, the
// seed C and the dequant scales are all Verilator-inferred RAMs (O(1) random
// access), and the result is read back through a combinational read port
// (c_raddr -> c_rdata) — so there are no large array ports (which Verilator
// 4.038 would copy element-by-element every cycle).  The 16.7M MACs of a full
// 128x1024x128 tile execute as a compact runtime loop with no huge
// dynamic-index register mux and no unrolled-softfloat code explosion.
//
// The numeric result is bit-identical to the previous combinational engine:
//   - fp path:   acc(m,n) = fp32_add(acc, fp32_mul(a,b)) in K order (k outer),
//                seeded from cin_elems at k==0 unless acc_init.
//   - int8 path: exact INT32 accumulate in K order.
//   - dequant:   per-128-K-group exact INT32 partial, folded (i32_to_f32 *
//                scale) into the fp32 accumulator at each group boundary.
//
// The batch dimension is handled by the CP (one tile per batch element).
// Operands arrive dtype-decoded to 32-bit working values by the CP:
//   fp  paths: bf16/fp16 elements -> fp32 bits;  int paths: int8 -> sign-ext.
//   scale[]  : bf16 scales -> fp32 bits, layout [n*G + g] (written via ports).
// Result c_rdata: fp32 bits (fp / dequant) or int32 bits (int8 no-dequant).
//
// Note: the systolic tile is fixed 128x128, so N == MAX_N == 128.
// ============================================================================
`ifndef MATRIX_ENGINE_SV
`define MATRIX_ENGINE_SV

`include "qcore_pkg.sv"
`include "softfloat.sv"

module matrix_engine #(
  parameter int MAX_M = 128,   // array rows / output rows
  parameter int MAX_N = 128,   // array columns / output columns
  parameter int MAX_K = 4096   // reduction dim (covers hidden 1024, ctx <= 4096)
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [7:0]  M,
  input  logic [7:0]  N,
  input  logic [15:0] K,
  input  logic [2:0]  srcA,
  input  logic [2:0]  srcB,
  input  logic        acc_init,
  input  logic        dequant,
  input  logic        start,          // pulse: reset counters, enter MAC phase
  input  logic        step,           // pulse: one MAC (consume a_slice/b_slice)
  input  logic [31:0] a_slice      [MAX_M],           // A[:, k] (decoded)
  input  logic [31:0] b_slice      [MAX_N],           // B[k, :] (decoded)
  // C seed / scale RAM write ports (avoid huge array-port copies per cycle)
  input  logic [13:0] cin_waddr,
  input  logic [31:0] cin_wdata,
  input  logic        cin_we,
  input  logic [11:0] scale_waddr,
  input  logic [31:0] scale_wdata,
  input  logic        scale_we,
  // registered result read port
  input  logic [13:0] c_raddr,
  output logic [31:0] c_rdata,
  output logic        done               // high once the full result is in acc
);
  import qcore_pkg::*;
  import softfloat_pkg::*;

  // RAMs (Verilator-inferred: only accessed in always_ff)
  logic [31:0] acc     [MAX_M * MAX_N];              // fp32 / int32 accumulator
  logic [31:0] partial [MAX_M * MAX_N];              // int32 per-group partial (dequant)
  logic [31:0] cin_elems   [MAX_M * MAX_N];          // seed C (written via ports)
  logic [31:0] scale_elems [MAX_N * (MAX_K / 128)];  // dequant scales (written via ports)

  logic [15:0] kk;                                   // current K slice
  logic [7:0]  mm, nn;                               // current (row, col)
  logic        mac_active;
  logic [8:0]  G;
  logic [15:0] acc_idx;                              // mm*N + nn (wide)
  logic [15:0] scale_idx;                            // nn*G + (kk>>7) (wide)

  assign G         = K[15:7];                        // groups of 128 along K
  assign acc_idx   = {8'b0, mm} * {8'b0, N} + {8'b0, nn};
  assign scale_idx = {8'b0, nn} * {8'b0, G} + {8'b0, kk[15:7]};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      kk <= 16'b0; mm <= 8'b0; nn <= 8'b0;
      mac_active <= 1'b0; done <= 1'b0;
    end
    else begin
      if (start) begin
        kk <= 16'b0; mm <= 8'b0; nn <= 8'b0;
        mac_active <= 1'b1; done <= 1'b0;
      end
      else if (mac_active) begin
        if (step) begin
          // ---- one MAC at (mm, nn, kk) --------------------------------------
          if (dequant) begin
            if (srcB == DT_INT4) begin
              // W4A16 (srcA=BF16/FP16, srcB=INT4, acc=FP32): fp32 in-group
              // accumulation (02 §6 / 04 §1.2).  The 4-bit weight arrives
              // sign-extended in b_slice (CP decode_elem); unpack the low
              // nibble and convert to fp32 (exact for |w| <= 8), multiply by
              // the bf16 activation mantissa value, and accumulate the
              // per-128-K-group partial in fp32.  Group boundary folds the
              // partial through the per-group scale (reused post-process).
              // W4A8 (srcA=INT8) is backlog and never encoded.
              logic [31:0] p_old;
              logic [31:0] w_f;
              logic [31:0] np;
              w_f   = i32_to_f32({{28{b_slice[nn][3]}}, b_slice[nn][3:0]});
              p_old = (kk[6:0] == 7'b0) ? 32'b0 : partial[acc_idx];
              np = fp32_add(p_old, fp32_mul(a_slice[mm], w_f));
              partial[acc_idx] <= np;
              if ((kk & 16'h7F) == 16'h7F) begin
                logic [31:0] base;
                // first group seeds from 0 (acc_init) or the C-seed; later
                // groups accumulate on acc.  (acc_init=1 must NOT reuse a
                // stale acc[] from the previous tile.)
                base = (kk == 16'd127) ? (acc_init ? 32'b0 : cin_elems[acc_idx])
                                       : acc[acc_idx];
                acc[acc_idx] <= fp32_add(base,
                  fp32_mul(np, scale_elems[scale_idx]));
                partial[acc_idx] <= 32'b0;
              end
            end else begin
              // W8A8 (srcA=srcB=INT8): exact INT32 partial per 128-group ->
              // fp32 scale -> fp32 accumulate.
              logic [31:0] p_old;
              logic [31:0] np;
              logic signed [31:0] prod8;  // int8 x int8 -> 32-bit signed
              prod8 = $signed(a_slice[mm][7:0]) * $signed(b_slice[nn][7:0]);
              p_old = (kk[6:0] == 7'b0) ? 32'b0 : partial[acc_idx];
              np = p_old + prod8;
              partial[acc_idx] <= np;
              if ((kk & 16'h7F) == 16'h7F) begin
                logic [31:0] base;
                base = (kk == 16'd127) ? (acc_init ? 32'b0 : cin_elems[acc_idx])
                                       : acc[acc_idx];
                acc[acc_idx] <= fp32_add(base,
                  fp32_mul(i32_to_f32(np), scale_elems[scale_idx]));
                partial[acc_idx] <= 32'b0;
              end
            end
          end
          else if (srcA == DT_INT8 && srcB == DT_INT8) begin
            // exact INT32 accumulate (bit-exact regardless of order)
            logic [31:0] prod;
            prod = $signed(a_slice[mm][7:0]) * $signed(b_slice[nn][7:0]);
            if (kk == 16'b0)
              acc[acc_idx] <=
                (acc_init ? 32'sd0 : cin_elems[acc_idx]) + prod;
            else
              acc[acc_idx] <= acc[acc_idx] + prod;
          end
          else begin
            // BF16/FP16 fp32 accumulate in K order (k outer => same per-(m,n) order)
            if (kk == 16'b0)
              acc[acc_idx] <=
                fp32_add(acc_init ? 32'b0 : cin_elems[acc_idx],
                         fp32_mul(a_slice[mm], b_slice[nn]));
            else
              acc[acc_idx] <=
                fp32_add(acc[acc_idx], fp32_mul(a_slice[mm], b_slice[nn]));
          end
          // ---- advance (nn -> mm -> kk) ------------------------------------
          if (nn == N - 8'd1) begin
            nn <= 8'b0;
            if (mm == M - 8'd1) begin
              mm <= 8'b0;
              if (kk == K - 16'd1) begin
                mac_active <= 1'b0;     // result ready in acc
                done <= 1'b1;
              end else kk <= kk + 16'd1;
            end else mm <= mm + 8'd1;
          end else nn <= nn + 8'd1;
        end
      end
      // RAM write ports (independent of the MAC/readout state machine)
      if (cin_we)   cin_elems[cin_waddr]     <= cin_wdata;
      if (scale_we) scale_elems[scale_waddr] <= scale_wdata;
    end
  end

  // Result read port: combinational read of the accumulator RAM.  The CP
  // latches `c_rdata` in S_MX_RDOUT with `wr_elem <= c_rdata` while advancing
  // `rd_ptr`; a *registered* port adds one cycle of latency that made the first
  // element of every batch after the first read a stale address (acc[128] for
  // DC M=1, which is never written -> col0 == 0 for tiles 1..15).
  assign c_rdata = acc[c_raddr];

endmodule
`endif // MATRIX_ENGINE_SV
