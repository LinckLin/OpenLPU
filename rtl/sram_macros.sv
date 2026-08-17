// ============================================================================
// sram_macros.sv — SMIC28 SRAM macro behavioral models (co-sim only).
//
// Functional (Verilator) stand-ins for the compiled SMIC28 macros in
// asic/sram_macros/ (SM18CA001 single-port SRAM / SM18CD001 single-port RF).
// Port names, widths and directions match the generated Verilog models and
// PORTS.md exactly (uppercase for SM18CA001, lowercase for SM18CD001):
//
//   kh4096x64  : 4096 x 64 single-port SRAM, 1-cycle read, synchronous write
//   ang4096x64 : 4096 x 64 single-port SRAM (stage-2 angle cache; unused yet)
//   kn128x16   : 128 x 16 single-port register file, 1-cycle read
//
// The compiled ARM models are gate-level timing models (specify blocks +
// timing-check feedback loops) that Verilator 4.038 cannot elaborate
// (UNOPTFLAT circular logic).  Co-sim only needs the functional contract —
// 1-cycle registered read, synchronous write — which these models reproduce
// bit-exactly.  DC synthesis uses the real Liberty (.lib -> .db) instead:
// asic/dc/hoist_dc.py strips these bodies to `(* blackbox *)` stubs and
// dc_top.tcl links them against asic/dc/db/<macro>_<corner>.db.
`ifndef SRAM_MACROS_SV
`define SRAM_MACROS_SV

module kh4096x64 (
  output logic [63:0] Q,
  input  logic        CLK,
  input  logic        CEN,      // chip enable, active low
  input  logic        WEN,      // write enable, active low (1 = read)
  input  logic [11:0] A,
  input  logic [63:0] D,
  input  logic [2:0]  EMA,      // extra margin adjust (tied 011)
  input  logic [1:0]  EMAW,     // EMA width (tied 01)
  input  logic        EMAS,     // EMA select (tied 0)
  input  logic        RET1N     // retention, active low (tied 1)
);
  logic [63:0] mem [0:4095];
  always_ff @(posedge CLK) begin
    if (!CEN) begin
      if (!WEN) mem[A] <= D;
      else      Q   <= mem[A];
    end
  end
endmodule

module ang4096x64 (
  output logic [63:0] Q,
  input  logic        CLK,
  input  logic        CEN,
  input  logic        WEN,
  input  logic [11:0] A,
  input  logic [63:0] D,
  input  logic [2:0]  EMA,
  input  logic [1:0]  EMAW,
  input  logic        EMAS,
  input  logic        RET1N
);
  logic [63:0] mem [0:4095];
  always_ff @(posedge CLK) begin
    if (!CEN) begin
      if (!WEN) mem[A] <= D;
      else      Q   <= mem[A];
    end
  end
endmodule

module kn128x16 (
  output logic [15:0] q,
  input  logic        clk,
  input  logic        cen,
  input  logic        wen,
  input  logic [6:0]  a,
  input  logic [15:0] d,
  input  logic [2:0]  ema,
  input  logic [1:0]  emaw,
  input  logic        emas,
  input  logic        ret1n
);
  logic [15:0] mem [0:127];
  always_ff @(posedge clk) begin
    if (!cen) begin
      if (!wen) mem[a] <= d;
      else      q   <= mem[a];
    end
  end
endmodule

`endif // SRAM_MACROS_SV
