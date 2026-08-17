// ============================================================================
// bb_sram.sv — generic single-port SRAM black box for full-design DC synthesis.
//
// The co-sim functional model stores the instruction stream in the flat array
// `logic [127:0] imem [0:NINST-1]` (rtl/ref/asicsnap/command_processor.sv),
// which Verilator infers as O(1) random-access storage.  For DC that array
// would infer as a 4096x128 flip-flop bank (~512 K flops) — part of the P10
// §10.5 full-design resource wall.  A real ASIC keeps the instruction stream in
// an SRAM, so hoist_dc.py re-maps it to this black-box macro: 1 synchronous
// write port + 1 combinational read port (co-sim read semantics).
//
// Like sram_macro.sv (8 MiB scratchpad), the macro is NOT mapped to gates and
// its area/power stay the separate estimate in docs/p10/asic-report.md.  The
// co-sim copy in rtl/ is untouched; this black box exists only in asic/dc/gen_full/.
// ============================================================================
`ifndef BB_SRAM_SV
`define BB_SRAM_SV

(* blackbox *)
module bb_sram #(
  parameter int AW = 12,   // address width (default: 4096-entry instruction SRAM)
  parameter int DW = 128   // data width    (default: 128-bit instruction word)
) (
  input  logic          clk,
  input  logic          we,
  input  logic [AW-1:0] waddr,
  input  logic [DW-1:0] wdata,
  input  logic [AW-1:0] raddr,
  output logic [DW-1:0] rdata     // combinational read (co-sim semantic)
);

endmodule

`endif // BB_SRAM_SV
