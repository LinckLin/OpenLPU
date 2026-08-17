// ============================================================================
// qcore_pkg.sv — QCore RTL package: parameters, types, frozen constants
//
// QCore = Command Processor + Matrix/Vector/DMA engines + KV datapath
//        + 16-bank scratchpad SRAM + HBM (co-sim model).
//
// Project convention (P7 M6, plans/p6-p7-plan.md §4.1):
//   * single clock domain, 1 GHz (1 cyc = 1 ns)
//   * async reset, active-low, synchronous release
//   * SRAM 16 bank x 512 KiB, 2R1W + fixed-priority arbitration
//   * HBM 64 B burst, parametrized latency
//   * Verilator version locked (see docs/p7/rtl-report.md)
//
// Authority: docs/spec.md §3/§4; docs/spec-src/{02,03,04,05} (frozen).
// ============================================================================
`ifndef QCORE_PKG_SV
`define QCORE_PKG_SV

package qcore_pkg;

  // --------------------------------------------------------------------------
  // Matrix array geometry (D4 / 04 §0): 128x128 dual-MAC.  Full-size 128x128 is
  // the acceptance instance; the width is parameterised so a 32x32 dev instance
  // keeps the exact same interface (plans/p6-p7-plan.md §4 — dev path only).
  // --------------------------------------------------------------------------
  localparam int MATRIX_N = 128;      // array columns (N); 32 for dev
  localparam int MATRIX_ROWS = 128;   // array rows (K-station); 32 for dev
  localparam int DC_LANES = 16;       // DC mode: 16 lane x 8 row GEMV
  localparam int DC_LANE_ROWS = 8;    // rows per lane in DC
  localparam int LANES = 128;         // vector engine lane count

  // Memory (03 §2 / §3): 16 bank x 512 KiB scratchpad + HBM.
  // BANK_BYTES is the single architectural knob: SRAM_BYTES / SRAM_WORDS /
  // BANK_WORDS are derived from it.  The synthesizable crossbar (rtl/sram.sv
  // `sram_top`) takes BANK_BYTES as an *instance parameter* so a shrunk
  // scratchpad (e.g. 256 KiB/bank -> 4 MiB, ~84% of the 4.75 MiB ZCU104
  // budget) keeps the 16-bank interleave and 6-port structure; the co-sim
  // (qmem) stays at the frozen 8 MiB default.  See plans/m8-wait-plan.md §3.
  localparam int N_BANK = 16;
  localparam int BANK_BYTES = 512 * 1024;         // 512 KiB / bank (frozen)
  localparam int SRAM_BYTES = N_BANK * BANK_BYTES;// 8 MiB
  localparam int WORD_BYTES = 16;                 // SRAM access granularity
  localparam int SRAM_WORDS = SRAM_BYTES / WORD_BYTES;  // 2^19
  localparam int BANK_WORDS = BANK_BYTES / WORD_BYTES;  // 2^15

  // HBM co-sim model: 1 GiB (2^30 B) dense window.  The architectural address
  // space stays 40-bit (16 GiB); the co-sim materialises the window that
  // covers 0.6B weights (<= 32 MiB) and the full 0.6B KV region (28 layers x
  // 8 heads x 2 x 2 MiB = 896 MiB, highest touched byte ~= 0.9 GiB).  Addresses
  // above the window are an out-of-window fault in the co-sim (03 §4.4 fault).
  localparam int HBM_BYTES = 1024 * 1024 * 1024;  // 2^30, 1 GiB
  localparam int HBM_BURST = 64;                  // 64 B burst (03 §3.4)
  localparam int HBM_ADDR_BITS = 40;              // architectural width

  // Co-sim tile bounds (v0, 0.6B): single instruction tile dims.
  localparam int MAX_M = 128;
  localparam int MAX_N = 128;
  localparam int MAX_K = 4096;      // covers hidden 1024, intermediate 3072,
                                    // attention seq <= 4096 (cache4096 decode)
  localparam int MAX_BATCH = 16;
  localparam int MAX_A_ELEMS = MAX_M * MAX_K * MAX_BATCH;
  localparam int MAX_B_ELEMS = MAX_K * MAX_N;
  localparam int MAX_C_ELEMS = MAX_M * MAX_N * MAX_BATCH;
  localparam int MAX_SCALE_ELEMS = MAX_N * (MAX_K / 128);  // N x G (G = K/128)

  // Frozen timing constants (timing.py §1; 03/04).
  localparam int T_FIRST = 100;        // HBM fixed latency (03 §3.3)
  localparam int HBM_READ_BPC = 720;   // sustained read  B/cyc (900*0.8)
  localparam int HBM_WRITE_BPC = 240;  // sustained write B/cyc (300*0.8)
  localparam int SRAM_READ_BPC = 512;  // 16 bank x 2R x 16 B
  localparam int SRAM_WRITE_BPC = 256; // 16 bank x 1W x 16 B
  localparam int MODE_SWITCH = 300;    // PF<->DC switch (04 §2.3)
  localparam int ARRAY_MAC = MATRIX_N * MATRIX_ROWS * 2; // 32768 INT8 MAC/cyc

  // Engine tags (02 §2.2) and opcodes (02 §3) — frozen.
  localparam logic [7:0] ENG_SYS    = 8'h00;
  localparam logic [7:0] ENG_DMA    = 8'h01;
  localparam logic [7:0] ENG_MATRIX = 8'h02;
  localparam logic [7:0] ENG_VECTOR = 8'h03;
  localparam logic [7:0] ENG_KV     = 8'h04;

  localparam logic [7:0] OP_MODE   = 8'h00;
  localparam logic [7:0] OP_CONFIG = 8'h01;
  localparam logic [7:0] OP_BARRIER= 8'h02;
  localparam logic [7:0] OP_WAIT   = 8'h03;
  localparam logic [7:0] OP_NOP    = 8'h04;
  localparam logic [7:0] OP_DMA_LOAD   = 8'h20;
  localparam logic [7:0] OP_DMA_STORE  = 8'h21;
  localparam logic [7:0] OP_DMA_PREFETCH=8'h22;
  localparam logic [7:0] OP_GEMM   = 8'h40;
  localparam logic [7:0] OP_GEMV   = 8'h41;
  localparam logic [7:0] OP_BMM    = 8'h42;
  localparam logic [7:0] OP_VADD   = 8'h80;
  localparam logic [7:0] OP_VSUB   = 8'h81;
  localparam logic [7:0] OP_VMUL   = 8'h82;
  localparam logic [7:0] OP_VDIV   = 8'h83;
  localparam logic [7:0] OP_VRECIP = 8'h84;
  localparam logic [7:0] OP_VEXP   = 8'h85;
  localparam logic [7:0] OP_VRSQRT = 8'h86;
  localparam logic [7:0] OP_VSILU  = 8'h87;
  localparam logic [7:0] OP_VMAX   = 8'h88;
  localparam logic [7:0] OP_VMOV   = 8'h89;
  localparam logic [7:0] OP_VSCALE = 8'h8A;
  localparam logic [7:0] OP_VMASK  = 8'h8B;
  localparam logic [7:0] OP_VREDUCE_SUM = 8'h8C;
  localparam logic [7:0] OP_VREDUCE_MAX = 8'h8D;
  localparam logic [7:0] OP_ROPE   = 8'h8E;
  localparam logic [7:0] OP_RMSNORM= 8'h8F;
  localparam logic [7:0] OP_QUANT  = 8'h90;
  localparam logic [7:0] OP_DEQUANT= 8'h91;
  localparam logic [7:0] OP_KV_APPEND      = 8'hC0;
  localparam logic [7:0] OP_KV_STORE_BLOCK = 8'hC1;
  localparam logic [7:0] OP_KV_LOAD        = 8'hC2;
  localparam logic [7:0] OP_KV_GATHER      = 8'hC3;

  // dtype codes (02 §2.2) and acc codes.
  localparam logic [2:0] DT_BF16 = 3'd0;
  localparam logic [2:0] DT_FP16 = 3'd1;
  localparam logic [2:0] DT_INT8 = 3'd2;
  localparam logic [2:0] DT_INT4 = 3'd3;
  localparam logic [2:0] DT_INT32= 3'd4;
  localparam logic [2:0] DT_INT16= 3'd5;
  localparam logic [2:0] DT_FP8  = 3'd6;

  localparam logic [1:0] ACC_INT32 = 2'd0;
  localparam logic [1:0] ACC_FP32  = 2'd1;
  localparam logic [1:0] ACC_FP16  = 2'd2;

  // Register file indices (02 §2.3 / 05 §2).
  localparam int AR_KV_BASE = 63;
  localparam int C_KV_POS   = 30;
  localparam int C_SLAB_SHIFT = 31;
  // B' (INT8-K fold + INT4-V) KV scale metadata ABI (bprime-impl report):
  //   AR_KV_SCALE_BASE = HBM byte base of per-token scale slabs
  //     (per (layer,head): 8192 x 4 B = 32 KB; record = [s_q 2B][s_v 2B]).
  //   C_KVNORM_BASE     = SRAM word addr of the static per-channel folded
  //     scale (k_norm) table: per (layer,head) 128 signed BF16 = 256 B.
  localparam int AR_KV_SCALE_BASE = 62;
  localparam int C_KVNORM_BASE    = 29;
  localparam int KV_SCALE_SLAB_STRIDE = 8192 * 4;
  localparam int K_NORM_HEAD_BYTES    = 128 * 2;
  // B' B-feed fusion (rotator-impl, plan v3): CD register [31:21] extended
  // semantics + BMM instruction reserved [20:5] = pos_base.  The 33-instruction
  // set is unchanged; these bits only fire under the new descriptor (CD[31]=1).
  //   CD[31]    KV_QUANT : 1 = B operand is quantized KV (B-feed dequant)
  //   CD[30]    ROTATE_K : 1 = apply absolute-position RoPE to K in B-feed
  //   CD[29:21] KV_IDX   : (layer*8 + head), selects k_norm/scale slab
  localparam int CD_KV_QUANT  = 31;
  localparam int CD_ROTATE_K  = 30;
  localparam int CD_KV_IDX_HI = 29;
  localparam int CD_KV_IDX_LO = 21;
  localparam int POS_BASE_HI  = 20;   // BMM reserved [20:5] = pos_base (16b)
  localparam int POS_BASE_LO  = 5;
  // Instruction fields: generic header offsets (02 §2.1/§2.2).
  localparam int ENG_HI = 127;   // engine tag  [127:120]
  localparam int ENG_LO = 120;
  localparam int OP_HI  = 119;   // opcode      [119:112]
  localparam int OP_LO  = 112;
  localparam int DT_HI  = 111;   // dtype flags [111:104]
  localparam int DT_LO  = 104;

  // 32-bit scratchpad word (16 B) as two 64-bit halves for Verilator-friendliness.
  typedef logic [127:0] sram_word_t;

  // Fixed-priority SRAM arbitration (03 §2.3), highest first:
  //   MATRIX.A > MATRIX.B > MATRIX.C > VECTOR > DMA > KV
  localparam int PRIO_MATRIX_A = 0;
  localparam int PRIO_MATRIX_B = 1;
  localparam int PRIO_MATRIX_C = 2;
  localparam int PRIO_VECTOR   = 3;
  localparam int PRIO_DMA      = 4;
  localparam int PRIO_KV       = 5;

  // --------------------------------------------------------------------------
  // Frozen per-instruction cycle model (04 §1.4/§2.2/§3.2, 03 §3.3, timing.py).
  // The co-sim reports these per-instruction latencies; qsim baseline timing.py
  // computes the identical numbers.  Integer arithmetic, no rounding.
  // --------------------------------------------------------------------------
  function automatic int ceil_div(int a, int b);
    return (a + b - 1) / b;
  endfunction

  // PF GEMM/GEMV/BMM tile: ceil(K/256)*M + 256 (04 §1.4).
  function automatic int matrix_pf_cycles(int M, int K);
    return ceil_div(K, 256) * M + 256;
  endfunction

  // DC 16-lane GEMV batch: ceil(K/16) (04 §2.2).
  function automatic int matrix_dc_batch_cycles(int K);
    return ceil_div(K, 16);
  endfunction

  // VECTOR per-128-element-block latency table (04 §3.2); returned for one
  // 128-element block, the co-sim scales by ceil(len/128) for elementwise ops.
  function automatic int vector_latency(input logic [7:0] op);
    case (op)
      OP_VADD, OP_VSUB, OP_VMAX:   vector_latency = 2;
      OP_VMUL, OP_VSCALE:          vector_latency = 3;
      OP_VDIV:                     vector_latency = 10;
      OP_VRECIP, OP_VRSQRT:        vector_latency = 7;
      OP_VEXP:                     vector_latency = 8;
      OP_VSILU:                    vector_latency = 9;
      OP_VMOV, OP_VMASK:           vector_latency = 1;
      OP_VREDUCE_SUM, OP_VREDUCE_MAX, OP_ROPE: vector_latency = 8;
      OP_QUANT, OP_DEQUANT:        vector_latency = 5;
      OP_RMSNORM:                  vector_latency = 16;  // per-head microcode
      default:                     vector_latency = 1;
    endcase
  endfunction

  // HBM / SRAM transfer cycles (03 §3.3 sustained B/cyc).  LOAD is SRAM-write
  // limited, STORE is HBM-write limited; both add T_FIRST fixed latency.
  function automatic int hbm_read_cycles(int nbytes);
    return ceil_div(nbytes, HBM_READ_BPC);
  endfunction
  function automatic int hbm_write_cycles(int nbytes);
    return ceil_div(nbytes, HBM_WRITE_BPC);
  endfunction
  function automatic int sram_write_cycles(int nbytes);
    return ceil_div(nbytes, SRAM_WRITE_BPC);
  endfunction
  function automatic int sram_read_cycles(int nbytes);
    return ceil_div(nbytes, SRAM_READ_BPC);
  endfunction

  function automatic int dtype_size(input logic [2:0] dt);
    case (dt)
      DT_BF16, DT_FP16, DT_INT16: dtype_size = 2;
      DT_INT8:  dtype_size = 1;
      DT_INT4:  dtype_size = 1;  // packed 2/byte; handled by engine
      DT_INT32, DT_FP8: dtype_size = 4;
      default:  dtype_size = 2;
    endcase
  endfunction

endpackage

`endif // QCORE_PKG_SV
