// ============================================================================
// command_processor.sv — QCore Command Processor (co-sim model).
//
// Fetches 128-bit Q-ISA instructions (02 §2), decodes them, resolves AR/C
// register operands (bit63 = HBM flag, SRAM word addr x16), marshals operands
// byte-by-byte from qmem into 32-bit working buffers, runs the combinational
// matrix/vector engines, encodes results back to qmem, and charges the frozen
// per-instruction cycle latency (qcore_pkg / timing.py).
//
// Functional execution is decoupled from cycle accounting: the CP charges the
// frozen model latency to total_cycles and reports it per instruction via
// trace_valid/trace_cycles, while byte-level marshalling proceeds on the
// actual clock.  This reproduces the qsim baseline timing exactly.
//
// Cycle model (frozen; mirrored 1:1 by the co-sim Python harness):
//   CONFIG / NOP / BARRIER / WAIT : 1
//   MODE (mode changes)           : MODE_SWITCH (300) else 1
//   DMA.LOAD / PREFETCH           : T_FIRST + sram_write_cycles(n)
//   DMA.STORE                     : T_FIRST + hbm_write_cycles(n)
//   GEMM/GEMV/BMM PF              : matrix_pf_cycles(M,K)
//   GEMM/GEMV/BMM DC              : matrix_dc_batch_cycles(K)
//   VECTOR                        : vector_latency(op) * max(1, ceil(len/128))
//   KV.APPEND                     : T_FIRST + hbm_write_cycles(512)
//   KV.STORE_BLOCK                : T_FIRST + hbm_write_cycles(2*count*256)
//   KV.LOAD                       : T_FIRST + sram_write_cycles(w)
//   KV.GATHER                     : T_FIRST + sram_write_cycles(copies*count*256)
// ============================================================================
`ifndef COMMAND_PROCESSOR_SV
`define COMMAND_PROCESSOR_SV

`include "qcore_pkg.sv"
`include "softfloat.sv"
`include "matrix_engine.sv"
`include "vector_engine.sv"
`include "dma_engine.sv"
`include "kv_addrgen.sv"

module command_processor #(
  parameter int NINST   = 4096,
  parameter int MAX_VEC = 4096
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic [11:0]  imem_waddr,
  input  logic         imem_we,
  input  logic [127:0] imem_wdata,
  input  logic [15:0]  prog_len,
  output logic         mem_rd_sel,
  output logic [39:0]  mem_rd_addr,
  input  logic [7:0]   mem_rd_data,
  output logic         mem_wr_en,
  output logic         mem_wr_sel,
  output logic [39:0]  mem_wr_addr,
  output logic [7:0]   mem_wr_data,
  output logic         done,
  output logic [63:0]  total_cycles,
  output logic         trace_valid,
  output logic [15:0]  trace_index,
  output logic [31:0]  trace_cycles
);
  import qcore_pkg::*;
  import softfloat_pkg::*;

  logic [63:0] AR [0:63];
  logic [31:0] C  [0:31];
  logic        mode;                  // 0 = PF, 1 = DC
  logic [127:0] imem [0:NINST-1];


  // instruction memory load (testbench backdoor)
  always_ff @(posedge clk) begin
    if (imem_we) imem[imem_waddr] <= imem_wdata;
  end

  // -- working buffers -------------------------------------------------------
  logic [31:0] va [MAX_VEC];
  logic [31:0] vb [MAX_VEC];
  logic [31:0] a_slice [128];
  logic [31:0] b_slice [128];
  // matrix-engine C-seed / scale RAM write-port signals (combinational)
  logic [13:0] cin_waddr;
  logic [31:0] cin_wdata;
  logic        cin_we;
  logic [11:0] scale_waddr;
  logic [31:0] scale_wdata;
  logic        scale_we;
  localparam logic [4:0]
    S_IDLE=0, S_FETCH=1, S_SINGLE=2,
    S_VEC_RDA=3, S_VEC_RDB=4, S_VEC_BROAD=5, S_VEC_GO=6, S_VEC_WR=7,
    S_MX_START=8, S_MX_STRM_A=9, S_MX_STRM_B=10, S_MX_RUN=11, S_MX_WAIT=12,
    S_MX_RDC=13, S_MX_RDS=14, S_MX_RDOUT=15, S_MX_WR=16,
    S_DMA=17, S_KV=18, S_DONE=19;

  logic [4:0]  state;
  logic [15:0] pc;
  integer      idx, bc, bidx, gidx;
  logic [31:0] accb;
  logic [31:0] accb_next;
  assign accb_next = (bc == 0) ? {24'b0, mem_rd_data}
                               : (accb | ({24'b0, mem_rd_data} << (8 * bc)));

  // latched operands
  logic [7:0]  op;
  logic [2:0]  sa, sb;
  logic [1:0]  ac;
  logic [39:0] a_base, b_base, c_base, scale_base;
  logic        a_sel, b_sel, c_sel, scale_sel;
  logic [7:0]  M, N;
  logic [15:0] K;
  logic [5:0]  batch;
  logic [15:0] len;
  logic [31:0] cval;
  logic        bcast;
  logic [31:0] imm;
  logic [31:0] row_stride_a, row_stride_b, row_stride_c;
  logic [31:0] batch_stride_a, batch_stride_b, batch_stride_c;
  logic        acc_init, dequant, ta, tb;
  logic [2:0]  out_dt, scale_dt;
  integer      in_esz, out_esz;
  integer      ngroups;
  logic [15:0] mx_k;      // K-stream counter (mirrors engine kk)
  logic [7:0]  mx_j;      // slice element index (0..M-1 / 0..N-1)
  logic [14:0] mx_mac;    // MAC index within current slice (0..M*N-1)
  integer      mx_ntot;   // M*N (width-safe product)
  integer      mx_nscales;// N*(K/128) (width-safe product)
  logic [13:0] rd_ptr;    // matrix result read address
  logic [31:0] wr_elem;   // matrix result element being written out
  logic [31:0] latency;

  // KV latched
  logic [5:0]  kv_layer;
  logic [2:0]  kv_head;
  logic [12:0] kv_pos;
  logic [13:0] kv_count;
  logic [1:0]  kv_sel2;
  logic        kv_bcast;
  logic [4:0]  kv_cstride;
  integer      kv_ntransfer;
  logic        kv_running;

  // decode current instruction
  wire [7:0] opcode_d = imem[pc][119:112];
  wire [7:0] eng_d    = imem[pc][127:120];

  // -- address helpers ---------------------------------------------------------
  function automatic logic [39:0] ar_addr(input logic [63:0] ar);
    ar_addr = ar[63] ? ar[39:0] : ({17'b0, ar[18:0]} << 4);
  endfunction
  function automatic logic ar_sel(input logic [63:0] ar);
    ar_sel = ar[63];
  endfunction
  function automatic integer elem_esz(input logic [2:0] dt);
    case (dt)
      DT_INT8, DT_INT4:  elem_esz = 1;
      DT_BF16, DT_FP16, DT_INT16: elem_esz = 2;
      DT_INT32:          elem_esz = 4;
      default:           elem_esz = 2;
    endcase
  endfunction
  function automatic integer elem_byte_off(input integer i, input logic [2:0] dt);
    case (dt)
      DT_INT4:  elem_byte_off = i / 2;
      DT_INT8:  elem_byte_off = i;
      DT_INT32: elem_byte_off = i * 4;
      default:  elem_byte_off = i * 2;
    endcase
  endfunction
  function automatic logic [31:0] decode_elem(input logic [31:0] raw,
                                              input logic [2:0] dt, input logic hi);
    case (dt)
      DT_BF16:  decode_elem = bf16_to_fp32(raw[15:0]);
      DT_FP16:  decode_elem = fp16_to_fp32(raw[15:0]);
      DT_INT8:  decode_elem = {{24{raw[7]}}, raw[7:0]};
      DT_INT4:  decode_elem = hi ? {{28{raw[7]}}, raw[7:4]} : {{28{raw[3]}}, raw[3:0]};
      DT_INT16: decode_elem = {{16{raw[15]}}, raw[15:0]};
      DT_INT32: decode_elem = raw;
      default:  decode_elem = raw;
    endcase
  endfunction
  function automatic logic [31:0] encode_elem(input logic [31:0] v, input logic [2:0] dt);
    case (dt)
      DT_BF16:  encode_elem = {16'b0, fp32_to_bf16(v)};
      DT_FP16:  encode_elem = {16'b0, fp32_to_fp16(v)};
      DT_INT8:  encode_elem = {24'b0, v[7:0]};
      DT_INT4:  encode_elem = {24'b0, v[7:0]};
      DT_INT16: encode_elem = {16'b0, v[15:0]};
      DT_INT32: encode_elem = v;
      default:  encode_elem = v;
    endcase
  endfunction

  // -- engines ------------------------------------------------------------------
  logic [31:0] vo [MAX_VEC];
  logic [31:0] vo_len;
  logic mx_start, mx_step, mx_done, vec_go;
  logic [13:0] c_raddr;
  logic [31:0] c_rdata;

  matrix_engine #(.MAX_M(128), .MAX_N(128), .MAX_K(4096)) u_matrix (
    .clk(clk), .rst_n(rst_n),
    .M(M), .N(N), .K(K), .srcA(sa), .srcB(sb),
    .acc_init(acc_init), .dequant(dequant),
    .start(mx_start), .step(mx_step), .done(mx_done),
    .a_slice(a_slice), .b_slice(b_slice),
    .cin_waddr(cin_waddr), .cin_wdata(cin_wdata), .cin_we(cin_we),
    .scale_waddr(scale_waddr), .scale_wdata(scale_wdata), .scale_we(scale_we),
    .c_raddr(c_raddr), .c_rdata(c_rdata)
  );
  vector_engine #(.MAX_VEC(MAX_VEC)) u_vector (
    .op(op), .srcA(sa), .srcB(sb), .acc(ac),
    .len(len), .cval(cval), .bcast(bcast), .imm(imm), .go(vec_go),
    .a_vec(va), .b_vec(vb), .out_vec(vo), .out_len(vo_len)
  );

  // -- DMA / KV transfer wiring -------------------------------------------------
  logic dma_start, dma_done, dma_active;
  // registered descriptor (DMA ops)
  logic dma_src_sel_r, dma_dst_sel_r, dma_mode_r;
  logic [39:0] dma_src_base_r, dma_dst_base_r;
  logic [15:0] dma_row_bytes_r, dma_num_rows_r;
  logic [31:0] dma_stride_r;
  // combinational descriptor (KV ops)
  logic kv_sel_kv;
  logic [39:0] kvg_base;
  logic [17:0] kvg_len;
  logic kv_src_sel_c, kv_dst_sel_c;
  logic [39:0] kv_src_base_c, kv_dst_base_c;
  logic [15:0] kv_row_bytes_c, kv_num_rows_c;

  kv_addrgen u_kvg (
    .kv_base(kv_base), .slab_shift(C[31]),
    .layer(kv_layer), .head(kv_head), .kv(kv_sel_kv),
    .pos_start(kv_pos), .count(kv_count),
    .out_base(kvg_base), .out_len(kvg_len)
  );
  logic [39:0] kv_base;

  dma_engine u_dma (
    .clk(clk), .rst_n(rst_n), .start(dma_start),
    .src_sel(dma_active && state == S_KV ? kv_src_sel_c : dma_src_sel_r),
    .dst_sel(dma_active && state == S_KV ? kv_dst_sel_c : dma_dst_sel_r),
    .src_base(dma_active && state == S_KV ? kv_src_base_c : dma_src_base_r),
    .dst_base(dma_active && state == S_KV ? kv_dst_base_c : dma_dst_base_r),
    .row_bytes(dma_active && state == S_KV ? kv_row_bytes_c : dma_row_bytes_r),
    .num_rows(dma_active && state == S_KV ? kv_num_rows_c : dma_num_rows_r),
    .src_stride(dma_active && state == S_KV ? 32'd256 : dma_stride_r),
    .mode(dma_active && state == S_KV ? 1'b1 : dma_mode_r),
    .rd_addr(dma_rd_addr), .rd_sel(dma_rd_sel), .rd_data(mem_rd_data),
    .wr_en(dma_wr_en), .wr_addr(dma_wr_addr), .wr_sel(dma_wr_sel),
    .wr_data(dma_wr_data), .done(dma_done)
  );
  logic [39:0] dma_rd_addr, dma_wr_addr;
  logic dma_rd_sel, dma_wr_sel, dma_wr_en;
  logic [7:0] dma_wr_data;

  assign dma_active = (state == S_DMA) || (state == S_KV);
  assign mem_rd_sel  = dma_active ? dma_rd_sel  : op_rd_sel;
  assign mem_rd_addr = dma_active ? dma_rd_addr : op_rd_addr;
  assign mem_wr_en   = dma_active ? dma_wr_en   : op_wr_en;
  assign mem_wr_sel  = dma_active ? dma_wr_sel  : op_wr_sel;
  assign mem_wr_addr = dma_active ? dma_wr_addr : op_wr_addr;
  assign mem_wr_data = dma_active ? dma_wr_data : op_wr_data;

  // combinational KV transfer descriptor (per gidx)
  always_comb begin
    kv_sel_kv = 1'b0;
    kv_src_sel_c = 1'b0; kv_src_base_c = 40'b0;
    kv_dst_sel_c = 1'b0; kv_dst_base_c = 40'b0;
    kv_row_bytes_c = 16'd256; kv_num_rows_c = 16'd1;
    case (op)
      OP_KV_APPEND: begin
        kv_sel_kv = (gidx == 0) ? 1'b0 : 1'b1;
        kv_src_sel_c = (gidx == 0) ? a_sel : b_sel;
        kv_src_base_c = (gidx == 0) ? a_base : b_base;
        kv_dst_sel_c = 1'b1; kv_dst_base_c = kvg_base;
        kv_num_rows_c = 16'd1;
        kv_ntransfer = 2;
      end
      OP_KV_STORE_BLOCK: begin
        kv_sel_kv = (gidx == 0) ? 1'b0 : 1'b1;
        kv_src_sel_c = (gidx == 0) ? a_sel : b_sel;
        kv_src_base_c = (gidx == 0) ? a_base : b_base;
        kv_dst_sel_c = 1'b1; kv_dst_base_c = kvg_base;
        kv_num_rows_c = {2'b0, kv_count};
        kv_ntransfer = 2;
      end
      OP_KV_LOAD: begin
        if (kv_sel2 == 2'd2) begin
          kv_sel_kv = (gidx == 0) ? 1'b0 : 1'b1;
          kv_dst_sel_c = (gidx == 0) ? a_sel : b_sel;
          kv_dst_base_c = (gidx == 0) ? a_base : b_base;
          kv_ntransfer = 2;
        end else begin
          kv_sel_kv = (kv_sel2 == 2'd0) ? 1'b0 : 1'b1;
          kv_dst_sel_c = (kv_sel2 == 2'd0) ? a_sel : b_sel;
          kv_dst_base_c = (kv_sel2 == 2'd0) ? a_base : b_base;
          kv_ntransfer = 1;
        end
        kv_src_sel_c = 1'b1; kv_src_base_c = kvg_base;
        kv_num_rows_c = {2'b0, kv_count};
      end
      OP_KV_GATHER: begin
        kv_sel_kv = kv_sel2[0];
        kv_src_sel_c = 1'b1; kv_src_base_c = kvg_base;
        kv_dst_sel_c = a_sel;
        kv_dst_base_c = a_base + gidx * (C[kv_cstride] * 32'd16);
        kv_num_rows_c = {2'b0, kv_count};
        kv_ntransfer = kv_bcast ? 4 : 1;
      end
      default: kv_ntransfer = 1;
    endcase
  end

  // -- operand read/write addressing -------------------------------------------
  logic [39:0] op_rd_addr, op_wr_addr;
  logic op_rd_sel, op_wr_en, op_wr_sel;
  logic [7:0] op_wr_data;

  always_comb begin
    op_rd_sel = 1'b0; op_rd_addr = 40'b0;
    case (state)
      S_VEC_RDA: begin
        op_rd_sel = a_sel;
        op_rd_addr = a_base + elem_byte_off(idx, sa) + bc;
      end
      S_VEC_RDB: begin
        op_rd_sel = (op == OP_QUANT || op == OP_DEQUANT) ? 1'b0 : b_sel;
        op_rd_addr = (op == OP_QUANT || op == OP_DEQUANT)
                   ? (scale_base + elem_byte_off(idx, scale_dt) + bc)
                   : (b_base + elem_byte_off(idx, sb) + bc);
      end
      S_MX_STRM_A: begin
        integer off, kk_i, j_i;
        kk_i = mx_k; j_i = mx_j;
        off = ta ? (kk_i * row_stride_a + j_i * in_esz)
                 : (j_i * row_stride_a + kk_i * in_esz);
        op_rd_sel = a_sel;
        op_rd_addr = a_base + off + bc;
      end
      S_MX_STRM_B: begin
        integer off, kk_i, j_i;
        kk_i = mx_k; j_i = mx_j;
        // B element byte offset uses B's own dtype (INT4 = 2/byte); the fast
        // index is N (tb=0, storage [K,N]) or K (tb=1, storage [N,K]).
        off = tb ? (j_i * row_stride_b + elem_byte_off(kk_i, sb))
                 : (kk_i * row_stride_b + elem_byte_off(j_i, sb));
        op_rd_sel = b_sel;
        op_rd_addr = b_base + off + bc;
      end
      S_MX_RDC: begin
        op_rd_sel = c_sel;
        op_rd_addr = c_base + elem_byte_off(idx, out_dt) + bc;
      end
      S_MX_RDS: begin
        op_rd_sel = scale_sel;
        op_rd_addr = scale_base + idx * 2 + bc;
      end
      default: begin end
    endcase
  end

  always_comb begin
    op_wr_en = 1'b0; op_wr_sel = 1'b0; op_wr_addr = 40'b0; op_wr_data = 8'b0;
    case (state)
      S_VEC_WR: begin
        logic [31:0] e;
        e = encode_elem(vo[idx], out_dt);
        op_wr_en = 1'b1;
        op_wr_sel = c_sel;
        op_wr_addr = c_base + elem_byte_off(idx, out_dt) + bc;
        case (bc)
          0: op_wr_data = e[7:0];
          1: op_wr_data = e[15:8];
          2: op_wr_data = e[23:16];
          default: op_wr_data = e[31:24];
        endcase
      end
      S_MX_WR: begin
        integer mm, nn, off;
        logic [31:0] e;
        mm = idx / N; nn = idx % N;
        off = mm * row_stride_c + nn * out_esz;
        e = encode_elem(wr_elem, out_dt);
        op_wr_en = 1'b1;
        op_wr_sel = c_sel;
        op_wr_addr = c_base + off + bc;
        case (bc)
          0: op_wr_data = e[7:0];
          1: op_wr_data = e[15:8];
          2: op_wr_data = e[23:16];
          default: op_wr_data = e[31:24];
        endcase
      end
      default: begin end
    endcase
  end

  // -- latency model --------------------------------------------------------------
  always_comb begin
    latency = 32'd1;
    case (op)
      OP_MODE: latency = (mode == imem[pc][103:102]) ? 32'd1 : MODE_SWITCH;
      OP_DMA_LOAD, OP_DMA_PREFETCH: latency = T_FIRST + sram_write_cycles(dma_row_bytes_r * dma_num_rows_r);
      OP_DMA_STORE: latency = T_FIRST + hbm_write_cycles(dma_row_bytes_r * dma_num_rows_r);
      OP_GEMM, OP_GEMV, OP_BMM:
        latency = (mode == 1'b1) ? matrix_dc_batch_cycles(K) : matrix_pf_cycles(M, K);
      OP_KV_APPEND: latency = T_FIRST + hbm_write_cycles(512);
      OP_KV_STORE_BLOCK: latency = T_FIRST + hbm_write_cycles(kv_count * 256 * 2);
      OP_KV_LOAD: begin
        integer nb;
        nb = (kv_sel2 == 2'd2) ? kv_count * 256 * 2 : kv_count * 256;
        latency = T_FIRST + sram_write_cycles(nb);
      end
      OP_KV_GATHER: latency = T_FIRST + sram_write_cycles(kv_count * 256 * (kv_bcast ? 4 : 1));
      default:
        if (op[7] == 1'b1)
          latency = vector_latency(op) * ((len == 0) ? 1 : ceil_div(len, 128));
        else
          latency = 32'd1;
    endcase
  end

  // width-safe element counts (M*N / N*(K/128) would truncate at 8 bits)
  always_comb begin
    mx_ntot    = 32'(M) * 32'(N);
    mx_nscales = 32'(N) * 32'(K >> 7);
  end

  // matrix-engine C-seed / scale RAM write ports (combinational; engine
  // samples them at the posedge, so accb_next carries the just-assembled word)
  always_comb begin
    cin_we      = (state == S_MX_RDC) && (bc == elem_esz(out_dt) - 1);
    cin_waddr   = idx[13:0];
    cin_wdata   = decode_elem(accb_next, out_dt, 1'b0);
    scale_we    = (state == S_MX_RDS) && (bc == 1);
    scale_waddr = idx[11:0];
    scale_wdata = decode_elem(accb_next, scale_dt, 1'b0);
  end

  // -- main FSM --------------------------------------------------------------------
  assign done = (state == S_DONE);
  assign mx_start = (state == S_MX_START);
  assign mx_step  = (state == S_MX_RUN);
  assign vec_go = (state == S_VEC_GO) || (state == S_VEC_WR);
  assign c_raddr = rd_ptr;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE; pc <= 16'b0;
      total_cycles <= 64'b0; trace_valid <= 1'b0;
      trace_index <= 16'b0; trace_cycles <= 32'b0;
      dma_start <= 1'b0; kv_running <= 1'b0;
    end else begin
      trace_valid <= 1'b0;
      dma_start <= 1'b0;

      case (state)
        S_IDLE: if (start) state <= S_FETCH;

        S_FETCH: begin
          op <= opcode_d;
          sa <= imem[pc][111:109];
          sb <= imem[pc][108:106];
          ac <= imem[pc][105:104];
          M <= imem[pc][85:78];
          N <= imem[pc][77:70];
          K <= imem[pc][69:54];
          batch <= imem[pc][53:48];
          len <= imem[pc][85:70];
          imm <= imem[pc][64:33];

          if (opcode_d inside {OP_CONFIG, OP_BARRIER, OP_WAIT, OP_NOP, OP_MODE})
            state <= S_SINGLE;
          else if (opcode_d inside {OP_DMA_LOAD, OP_DMA_STORE, OP_DMA_PREFETCH}) begin
            dma_src_sel_r <= ar_sel(AR[imem[pc][103:98]]);
            dma_src_base_r <= ar_addr(AR[imem[pc][103:98]]);
            dma_dst_sel_r <= ar_sel(AR[imem[pc][97:92]]);
            dma_dst_base_r <= ar_addr(AR[imem[pc][97:92]]);
            dma_row_bytes_r <= imem[pc][91:76];
            dma_num_rows_r <= imem[pc][54] ? imem[pc][75:60] : 16'd1;
            dma_stride_r <= C[imem[pc][59:55]];
            dma_mode_r <= imem[pc][54];
            state <= S_DMA; dma_start <= 1'b1;
          end
          else if (opcode_d inside {OP_KV_APPEND, OP_KV_STORE_BLOCK, OP_KV_LOAD, OP_KV_GATHER}) begin
            kv_layer <= imem[pc][91:86];
            kv_head <= imem[pc][85:83];
            kv_base <= ar_addr(AR[63]);
            kv_sel2 <= imem[pc][82:81];
            kv_bcast <= imem[pc][81];
            kv_cstride <= imem[pc][53:49];
            if (opcode_d == OP_KV_APPEND) begin
              kv_pos <= C[30][12:0]; kv_count <= 14'd1;
            end else if (opcode_d == OP_KV_STORE_BLOCK) begin
              kv_pos <= imem[pc][82:70]; kv_count <= imem[pc][69:56];
            end else begin
              kv_pos <= imem[pc][80:68]; kv_count <= imem[pc][67:54];
            end
            a_sel <= ar_sel(AR[imem[pc][103:98]]);
            a_base <= ar_addr(AR[imem[pc][103:98]]);
            b_sel <= ar_sel(AR[imem[pc][97:92]]);
            b_base <= ar_addr(AR[imem[pc][97:92]]);
            gidx <= 0; kv_running <= 1'b1;
            state <= S_KV;
          end
          else if (opcode_d inside {OP_GEMM, OP_GEMV, OP_BMM}) begin
            a_sel <= ar_sel(AR[imem[pc][103:98]]);
            a_base <= ar_addr(AR[imem[pc][103:98]]);
            b_sel <= ar_sel(AR[imem[pc][97:92]]);
            b_base <= ar_addr(AR[imem[pc][97:92]]);
            c_sel <= ar_sel(AR[imem[pc][91:86]]);
            c_base <= ar_addr(AR[imem[pc][91:86]]);
            row_stride_a <= C[imem[pc][47:43]][31:16];
            batch_stride_a <= C[imem[pc][47:43]][15:0];
            row_stride_b <= C[imem[pc][42:38]][31:16];
            batch_stride_b <= C[imem[pc][42:38]][15:0];
            row_stride_c <= C[imem[pc][37:33]][31:16];
            batch_stride_c <= C[imem[pc][37:33]][15:0];
            acc_init <= imem[pc][27];
            dequant <= imem[pc][25];
            ta <= imem[pc][24];
            tb <= imem[pc][23];
            // dequant scale descriptor CD = C[imem[pc][32:28]]: [20]=mode,
            // [19]=scale dtype, [18:0]=SRAM word addr (02 §6 / 04 §1.5).
            scale_sel  <= 1'b0;
            scale_base <= {21'b0, C[imem[pc][32:28]][18:0]} << 4;
            scale_dt   <= C[imem[pc][32:28]][19] ? DT_FP16 : DT_BF16;
            if (imem[pc][25]) out_dt <= DT_BF16;
            else if (imem[pc][111:109] == DT_BF16 || imem[pc][111:109] == DT_FP16)
              out_dt <= imem[pc][111:109];
            else out_dt <= DT_INT32;
            in_esz <= elem_esz(imem[pc][111:109]);
            bidx <= 0; idx <= 0; bc <= 0; mx_k <= 0; mx_j <= 0; mx_mac <= 0;
            if (!imem[pc][27])       state <= S_MX_RDC;   // !acc_init -> seed C
            else if (imem[pc][25])   state <= S_MX_RDS;   // dequant -> scales
            else                     state <= S_MX_START;
          end
          else begin
            // VECTOR
            a_sel <= ar_sel(AR[imem[pc][103:98]]);
            a_base <= ar_addr(AR[imem[pc][103:98]]);
            b_sel <= ar_sel(AR[imem[pc][97:92]]);
            b_base <= ar_addr(AR[imem[pc][97:92]]);
            c_sel <= ar_sel(AR[imem[pc][91:86]]);
            c_base <= ar_addr(AR[imem[pc][91:86]]);
            cval <= C[imem[pc][69:65]];
            bcast <= (imem[pc][69:65] != 5'b0);
            if (opcode_d == OP_QUANT) out_dt <= (imem[pc][108:106] == DT_INT4) ? DT_INT4 : DT_INT8;
            else if (opcode_d == OP_DEQUANT) out_dt <= (imem[pc][108:106] == DT_FP16) ? DT_FP16 : DT_BF16;
            else out_dt <= imem[pc][111:109];
            if (opcode_d == OP_QUANT || opcode_d == OP_DEQUANT) begin
              scale_base <= {21'b0, C[imem[pc][69:65]][18:0]} << 4;
              scale_dt <= C[imem[pc][69:65]][19] ? DT_FP16 : DT_BF16;
            end
            idx <= 0; bc <= 0;
            if (opcode_d == OP_VMASK) state <= S_VEC_GO;
            else                      state <= S_VEC_RDA;
          end
        end

        S_SINGLE: begin
          if (op == OP_CONFIG) begin
            if (imem[pc][97]) AR[imem[pc][103:98]] <= imem[pc][96:33];
            else              C[imem[pc][103:98]] <= imem[pc][96:33];
          end else if (op == OP_MODE) begin
            mode <= imem[pc][103:102];
          end
          total_cycles <= total_cycles + latency;
          trace_valid <= 1'b1; trace_index <= pc; trace_cycles <= latency;
          pc <= pc + 1;
          state <= (pc + 1 >= prog_len) ? S_DONE : S_FETCH;
        end

        S_VEC_RDA: begin
          integer esz, ntot;
          esz = elem_esz(sa);
          accb <= (bc == 0) ? {24'b0, mem_rd_data} : (accb | ({24'b0, mem_rd_data} << (8 * bc)));
          ntot = len;
          if (op == OP_VREDUCE_SUM || op == OP_VREDUCE_MAX)
            ntot = (cval == 0 ? 1 : cval) * len;
          ngroups <= (op == OP_VREDUCE_SUM || op == OP_VREDUCE_MAX) ? (cval == 0 ? 1 : cval) : 1;
          if (bc == esz - 1) begin
            va[idx] <= decode_elem(accb_next, sa, (sa == DT_INT4 && idx[0]));
            bc <= 0;
            if (idx == ntot - 1) begin
              idx <= 0; bc <= 0;
              if ((op == OP_VADD || op == OP_VSUB || op == OP_VMUL || op == OP_VDIV || op == OP_VMAX)
                  || op == OP_RMSNORM || op == OP_QUANT || op == OP_DEQUANT)
                state <= S_VEC_RDB;
              else
                state <= S_VEC_GO;
            end else idx <= idx + 1;
          end else bc <= bc + 1;
        end

        S_VEC_RDB: begin
          integer esz, ntot;
          logic [2:0] dt;
          dt = (op == OP_QUANT || op == OP_DEQUANT) ? scale_dt : sb;
          esz = elem_esz(dt);
          accb <= (bc == 0) ? {24'b0, mem_rd_data} : (accb | ({24'b0, mem_rd_data} << (8 * bc)));
          ntot = len;
          if ((op == OP_VADD || op == OP_VSUB || op == OP_VMUL || op == OP_VDIV || op == OP_VMAX) && bcast)
            ntot = 1;
          if (op == OP_QUANT || op == OP_DEQUANT)
            ntot = (cval[20] == 1'b0) ? 1 : len / 128;
          if (bc == esz - 1) begin
            vb[idx] <= decode_elem(accb_next, dt, (dt == DT_INT4 && idx[0]));
            bc <= 0;
            if (idx == ntot - 1) begin
              idx <= 0; bc <= 0;
              state <= S_VEC_GO;
            end else idx <= idx + 1;
          end else bc <= bc + 1;
        end


        S_VEC_GO: begin
          idx <= 0; bc <= 0;
          out_esz <= elem_esz(out_dt);
          state <= S_VEC_WR;
        end

        S_VEC_WR: begin
          if (bc == out_esz - 1) begin
            bc <= 0;
            if (idx == vo_len - 1) begin
              total_cycles <= total_cycles + latency;
              trace_valid <= 1'b1; trace_index <= pc; trace_cycles <= latency;
              pc <= pc + 1;
              state <= (pc + 1 >= prog_len) ? S_DONE : S_FETCH;
            end else idx <= idx + 1;
          end else bc <= bc + 1;
        end

        S_MX_START: begin
          mx_j <= 0; bc <= 0; mx_mac <= 0;
          state <= S_MX_STRM_A;
        end

        S_MX_STRM_A: begin
          accb <= (bc == 0) ? {24'b0, mem_rd_data} : (accb | ({24'b0, mem_rd_data} << (8 * bc)));
          if (bc == in_esz - 1) begin
            a_slice[mx_j] <= decode_elem(accb_next, sa, 1'b0);
            bc <= 0;
            if (mx_j == M - 1) begin mx_j <= 0; state <= S_MX_STRM_B; end
            else mx_j <= mx_j + 1;
          end else bc <= bc + 1;
        end

        S_MX_STRM_B: begin
          integer esz_b;
          logic   nib_hi;
          accb <= (bc == 0) ? {24'b0, mem_rd_data} : (accb | ({24'b0, mem_rd_data} << (8 * bc)));
          esz_b  = elem_esz(sb);
          nib_hi = (sb == DT_INT4) && (tb ? mx_k[0] : mx_j[0]);
          if (bc == esz_b - 1) begin
            b_slice[mx_j] <= decode_elem(accb_next, sb, nib_hi);
            bc <= 0;
            if (mx_j == N - 1) begin mx_j <= 0; mx_mac <= 0; state <= S_MX_RUN; end
            else mx_j <= mx_j + 1;
          end else bc <= bc + 1;
        end

        S_MX_RUN: begin
          // one MAC per cycle (engine advances its internal kk/mm/nn)
          if (mx_mac == mx_ntot - 1) begin
            mx_mac <= 0;
            if (mx_k == K - 1) begin
              state <= S_MX_WAIT;
            end else begin
              mx_k <= mx_k + 1;
              mx_j <= 0;
              state <= S_MX_STRM_A;
            end
          end else mx_mac <= mx_mac + 1;
        end

        S_MX_WAIT: begin
          if (mx_done) begin
            idx <= 0; bc <= 0;
            out_esz <= elem_esz(out_dt);
            rd_ptr <= 0;
            state <= S_MX_RDOUT;
          end
        end

        S_MX_RDOUT: begin
          // c_rdata = acc[rd_ptr] (registered); latch it and prefetch next
          wr_elem <= c_rdata;
          rd_ptr <= rd_ptr + 14'd1;
          state <= S_MX_WR;
        end

        S_MX_RDC: begin
          accb <= (bc == 0) ? {24'b0, mem_rd_data} : (accb | ({24'b0, mem_rd_data} << (8 * bc)));
          if (bc == elem_esz(out_dt) - 1) begin
            bc <= 0;
            if (idx == mx_ntot - 1) begin
              idx <= 0;
              if (dequant) state <= S_MX_RDS; else state <= S_MX_START;
            end else idx <= idx + 1;
          end else bc <= bc + 1;
        end

        S_MX_RDS: begin
          accb <= (bc == 0) ? {24'b0, mem_rd_data} : (accb | ({24'b0, mem_rd_data} << (8 * bc)));
          if (bc == 1) begin
            bc <= 0;
            if (idx == mx_nscales - 1) begin idx <= 0; state <= S_MX_START; end
            else idx <= idx + 1;
          end else bc <= bc + 1;
        end

        S_MX_WR: begin
          if (bc == out_esz - 1) begin
            bc <= 0;
            if (idx == mx_ntot - 1) begin
              if (bidx == (batch == 0 ? 0 : batch - 1)) begin
                total_cycles <= total_cycles + latency;
                trace_valid <= 1'b1; trace_index <= pc; trace_cycles <= latency;
                pc <= pc + 1;
                state <= (pc + 1 >= prog_len) ? S_DONE : S_FETCH;
              end else begin
                bidx <= bidx + 1; idx <= 0; bc <= 0; mx_k <= 0; mx_j <= 0; mx_mac <= 0;
                a_base <= a_base + batch_stride_a;
                b_base <= b_base + batch_stride_b;
                c_base <= c_base + batch_stride_c;
                state <= S_MX_START;
              end
            end else begin idx <= idx + 1; state <= S_MX_RDOUT; end
          end else bc <= bc + 1;
        end

        S_DMA: begin
          if (dma_done) begin
            total_cycles <= total_cycles + latency;
            trace_valid <= 1'b1; trace_index <= pc; trace_cycles <= latency;
            pc <= pc + 1;
            state <= (pc + 1 >= prog_len) ? S_DONE : S_FETCH;
          end
        end

        S_KV: begin
          if (kv_running) begin
            // issue the gidx-th transfer (descriptor is combinational from gidx)
            kv_running <= 1'b0;
            dma_start <= 1'b1;
          end else if (dma_done) begin
            if (gidx == kv_ntransfer - 1) begin
              total_cycles <= total_cycles + latency;
              trace_valid <= 1'b1; trace_index <= pc; trace_cycles <= latency;
              pc <= pc + 1;
              state <= (pc + 1 >= prog_len) ? S_DONE : S_FETCH;
            end else begin
              gidx <= gidx + 1;
              kv_running <= 1'b1;
            end
          end
        end

        S_DONE: begin end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
`endif // COMMAND_PROCESSOR_SV
