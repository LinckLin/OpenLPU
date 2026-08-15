// ============================================================================
// sram_macro.sv — 8 MiB scratchpad SRAM macro (black box for P10 synthesis).
//
// The 16-bank x 512 KiB, 2R1W SRAM storage (rtl/sram.sv `sram_bank`) is the
// physical macro boundary.  The co-sim (`qmem`) flattens it to a byte-addressable
// array with combinational read; a real compiled macro has 1-cycle read and a
// multi-port bank structure (sram_top crossbar).  For synthesis we black-box the
// storage: it is NOT mapped to gates.  Area/power use the OpenRAM / published
// density estimate in docs/p10/asic-report.md.
//
// Interface kept at co-sim semantics (combinational read + posedge byte write)
// so the CP byte-marshalling logic synthesizes unchanged; the timing model used
// by STA (ram_stub.lib) attaches a realistic 130 nm access time.
// ============================================================================
`ifndef SRAM_MACRO_SV
`define SRAM_MACRO_SV

(* blackbox *)
module sram_macro (
  input  logic        clk,
  input  logic [22:0] addr,       // byte address within 8 MiB
  output logic [7:0]  rd_data,    // combinational read (co-sim semantic)
  input  logic [7:0]  wr_data,
  input  logic        wr_en       // posedge byte write
);
endmodule

`endif // SRAM_MACRO_SV
