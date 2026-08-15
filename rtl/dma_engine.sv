// ============================================================================
// dma_engine.sv — DMA transfer engine (co-sim model).
//
// Performs a byte-level bulk copy between SRAM and HBM through qmem's byte
// ports.  Handles the ISA 2D descriptor (03 §4.1 / 02 §5.1):
//   src_addr(r) = src_base + (mode==2D ? r * src_stride : 0)
//   dst_addr(r) = dst_base + r * row_bytes            (SRAM side dense)
// direction is implicit in src_sel/dst_sel (0=SRAM, 1=HBM):
//   DMA.LOAD  = HBM -> SRAM,  DMA.STORE = SRAM -> HBM.
//
// The Command Processor charges the frozen transfer latency (T_FIRST +
// sram_write_cycles/hbm_write_cycles, qcore_pkg) independently of the actual
// byte-copy clock count; this engine only establishes the data.  64 B burst
// head/body/tail decomposition is a P9 synthesis concern — qmem is a flat
// byte-addressable model, so the functional copy is byte-exact.
//
// 1 byte is copied per cycle (combinational read + registered write).  KV
// transfers are driven through this same engine with HBM addresses produced
// by kv_addrgen (05 §1.3); KV.GATHER additionally issues 4 copies.
// ============================================================================
`ifndef DMA_ENGINE_SV
`define DMA_ENGINE_SV

`include "qcore_pkg.sv"

module dma_engine (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  // transfer descriptor (byte addresses; sel: 0=SRAM, 1=HBM)
  input  logic        src_sel,
  input  logic        dst_sel,
  input  logic [39:0] src_base,
  input  logic [39:0] dst_base,
  input  logic [15:0] row_bytes,
  input  logic [15:0] num_rows,
  input  logic [31:0] src_stride,     // 2D source row stride (bytes)
  input  logic        mode,           // 0 = 1D, 1 = 2D
  // qmem byte ports (combinational read, posedge write)
  output logic [39:0] rd_addr,
  output logic        rd_sel,
  input  logic [7:0]  rd_data,
  output logic        wr_en,
  output logic [39:0] wr_addr,
  output logic        wr_sel,
  output logic [7:0]  wr_data,
  output logic        done
);
  import qcore_pkg::*;

  logic [1:0] state;
  logic [15:0] row, bpos;

  localparam logic [1:0] S_IDLE = 2'd0, S_COPY = 2'd1;
  assign wr_en = (state == S_COPY);

  // Combinational read/write addressing for the current (row, byte).
  always_comb begin
    rd_addr = src_base + ((mode) ? src_stride * row : 32'b0) + bpos;
    rd_sel  = src_sel;
    wr_addr = dst_base + (row_bytes * row) + bpos;
    wr_sel  = dst_sel;
    wr_data = rd_data;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE;
      row <= 16'b0;
      bpos <= 16'b0;
      done <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            row <= 16'b0;
            bpos <= 16'b0;
            state <= S_COPY;
          end
        end
        S_COPY: begin
          if (bpos == row_bytes - 1) begin
            bpos <= 16'b0;
            if (row == num_rows - 1) begin
              state <= S_IDLE;
              done  <= 1'b1;
            end else begin
              row <= row + 1;
            end
          end else begin
            bpos <= bpos + 1;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
`endif // DMA_ENGINE_SV
