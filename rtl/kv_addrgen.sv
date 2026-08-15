// ============================================================================
// kv_addrgen.sv — KV cache address generator (05 §1.3 bit-exact formula).
//
// The KV datapath is DMA + this address generator (plans/p6-p7-plan.md §4):
//   KV.LOAD / KV.STORE_BLOCK / KV.APPEND are DMA transfers whose HBM side
//   address is derived here; KV.GATHER additionally replicates the payload
//   x4 into the SRAM broadcast destination (05 §4.4).
//
// Address formula (05 §1.3), SLAB_SHIFT-parameterised (C31 in {20,21,22}):
//   slab_index(layer, head, kv) = (layer<<4) | (head<<1) | kv
//   A(layer,head,kv,pos,d) = AR_KV_BASE | (slab_index << SLAB_SHIFT)
//                           | (pos << 8) | (d << 1)
// Per token row: 256 B (= 4 x 64 B bursts); element d: byte offset d<<1.
// ============================================================================
`ifndef KV_ADDRGEN_SV
`define KV_ADDRGEN_SV

`include "qcore_pkg.sv"

module kv_addrgen (
  input  logic [39:0] kv_base,      // AR_KV_BASE (bit63 already stripped)
  input  logic [31:0] slab_shift,   // C31: 20 | 21 | 22
  input  logic [5:0]  layer,        // 0..35
  input  logic [2:0]  head,         // 0..7
  input  logic        kv,           // 0=K 1=V
  input  logic [12:0] pos_start,
  input  logic [13:0] count,        // tokens
  output logic [39:0] out_base,     // HBM byte address of pos_start row
  output logic [17:0] out_len       // count*256 bytes
);
  logic [9:0] slab_index;

  // slab_index = (layer<<4) | (head<<1) | kv
  assign slab_index = {layer, head, kv};

  assign out_base = kv_base + (slab_index << slab_shift[4:0]) + ({27'b0, pos_start} << 8);
  assign out_len  = {4'b0, count} * 18'd256;

endmodule
`endif // KV_ADDRGEN_SV
