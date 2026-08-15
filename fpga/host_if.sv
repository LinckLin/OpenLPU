// ============================================================================
// host_if.sv — QCore host<->device control-plane abstraction (P9,
// board-independent).
//
// Exposes the four control-plane functions the host needs as a flat 32-bit
// register file, so the physical carrier (UART / Ethernet / PCIe) is mapped
// to this register interface at P9 freeze — the device logic is carrier-
// agnostic:
//
//   1. config           — CTRL (start / soft reset)
//   2. command queue    — PROG_LEN + CMDQ_W0..W3 + CMDQ_GO  (128-bit Q-ISA
//                         instructions loaded into the Command Processor)
//   3. qbin load        — MEM_SEL / MEM_ADDR_* / MEM_WDATA  (byte stream into
//                         device memory via ddr_if's host port)
//   4. logits readback  — MEM_RDATA / MEM_ADV                (byte stream out
//                         of device memory) + STATUS / TOTAL / TRACE
//
// Register map (host_addr is a 32-bit word index; writes are posedge, reads
// are combinational; host_ready is always asserted — a no-wait CSR bus):
//
//   +------+-------------+-----+--------------------------------------------+
//   | 0x00 | CTRL        |  W  | [0]=start pulse, [1]=soft reset            |
//   | 0x01 | STATUS      |  R  | [0]=done, [1]=running                      |
//   | 0x02 | TOTAL_LO    |  R  | total_cycles[31:0]                         |
//   | 0x03 | TOTAL_HI    |  R  | total_cycles[63:32]                        |
//   | 0x04 | TRACE       |  R  | [0]=trace_valid, [31:16]=trace_index       |
//   | 0x05 | TRACE_CYC   |  R  | trace_cycles[31:0]                         |
//   | 0x06 | PROG_LEN    |  W  | [15:0]=instruction count                   |
//   | 0x07 | CMDQ_ADDR   |  W  | [11:0]=instruction index (auto-inc)        |
//   | 0x08 | CMDQ_W0     |  W  | inst[31:0]                                 |
//   | 0x09 | CMDQ_W1     |  W  | inst[63:32]                                |
//   | 0x0A | CMDQ_W2     |  W  | inst[95:64]                                |
//   | 0x0B | CMDQ_W3     |  W  | inst[127:96]                               |
//   | 0x0C | CMDQ_GO     |  W  | commit buffered inst to imem[addr], addr++ |
//   | 0x10 | MEM_ADDR_LO |  W  | mem addr[31:0]                             |
//   | 0x11 | MEM_ADDR_HI |  W  | mem addr[ADDR_BITS-1:32]                   |
//   | 0x12 | MEM_SEL     |  W  | [0]=sel (0=SRAM, 1=DDR)                    |
//   | 0x13 | MEM_WDATA   |  W  | [7:0]=byte, [8]=commit(write + addr++)     |
//   | 0x14 | MEM_RDATA   |  R  | [7:0]=byte at {sel, addr}                  |
//   | 0x15 | MEM_ADV     |  W  | write -> addr++                            |
//   +------+-------------+-----+--------------------------------------------+
// ============================================================================
`ifndef HOST_IF_SV
`define HOST_IF_SV

module host_if #(
  parameter int NINST     = 4096,
  parameter int ADDR_BITS = 40
) (
  input  logic clk,
  input  logic rst_n,
  // -- host register bus ---------------------------------------------------
  input  logic [11:0] host_addr,
  input  logic [31:0] host_wdata,
  input  logic        host_wen,
  input  logic        host_ren,
  output logic [31:0] host_rdata,
  output logic        host_ready,
  // -- command processor: control + imem backdoor --------------------------
  output logic         start,
  output logic [11:0]  imem_waddr,
  output logic         imem_we,
  output logic [127:0] imem_wdata,
  output logic [15:0]  prog_len,
  input  logic         done,
  input  logic [63:0]  total_cycles,
  input  logic         trace_valid,
  input  logic [15:0]  trace_index,
  input  logic [31:0]  trace_cycles,
  // -- memory host port (qbin load / logits readback) -----------------------
  output logic                hd_en,
  output logic                hd_sel,
  output logic [ADDR_BITS-1:0] hd_addr,
  output logic [7:0]          hd_wdata,
  input  logic [7:0]          hd_rdata
);

  localparam logic [11:0]
    R_CTRL       = 12'h000,
    R_STATUS     = 12'h001,
    R_TOTAL_LO   = 12'h002,
    R_TOTAL_HI   = 12'h003,
    R_TRACE      = 12'h004,
    R_TRACE_CYC  = 12'h005,
    R_PROG_LEN   = 12'h006,
    R_CMDQ_ADDR  = 12'h007,
    R_CMDQ_W0    = 12'h008,
    R_CMDQ_W1    = 12'h009,
    R_CMDQ_W2    = 12'h00A,
    R_CMDQ_W3    = 12'h00B,
    R_CMDQ_GO    = 12'h00C,
    R_MEM_ADDR_LO= 12'h010,
    R_MEM_ADDR_HI= 12'h011,
    R_MEM_SEL    = 12'h012,
    R_MEM_WDATA  = 12'h013,
    R_MEM_RDATA  = 12'h014,
    R_MEM_ADV    = 12'h015;

  logic [11:0]          cmdq_addr;
  logic [127:0]         cmdq_buf;
  logic [15:0]          prog_len_r;
  logic [ADDR_BITS-1:0] mem_addr;
  logic                 mem_sel;
  logic                 running;

  // one-cycle pulses (combinational, asserted while host_wen is high)
  wire start_p     = host_wen && (host_addr == R_CTRL)      && host_wdata[0];
  wire mem_commit  = host_wen && (host_addr == R_MEM_WDATA) && host_wdata[8];
  wire mem_adv     = host_wen && (host_addr == R_MEM_ADV);
  wire cmdq_go     = host_wen && (host_addr == R_CMDQ_GO);

  assign start      = start_p;
  assign hd_en      = mem_commit;
  assign hd_sel     = mem_sel;
  assign hd_addr    = mem_addr;
  assign hd_wdata   = host_wdata[7:0];
  assign imem_we    = cmdq_go;
  assign imem_waddr = cmdq_addr;
  assign imem_wdata = cmdq_buf;
  assign prog_len   = prog_len_r;
  assign host_ready = 1'b1;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cmdq_addr  <= 12'b0;
      cmdq_buf   <= 128'b0;
      prog_len_r <= 16'b0;
      mem_addr   <= {ADDR_BITS{1'b0}};
      mem_sel    <= 1'b0;
      running    <= 1'b0;
    end else begin
      // running flag (start -> done)
      if (start_p) running <= 1'b1;
      if (done)    running <= 1'b0;

      if (host_wen) begin
        case (host_addr)
          R_PROG_LEN:    prog_len_r <= host_wdata[15:0];
          R_CMDQ_ADDR:   cmdq_addr  <= host_wdata[11:0];
          R_CMDQ_W0:     cmdq_buf[31:0]   <= host_wdata;
          R_CMDQ_W1:     cmdq_buf[63:32]  <= host_wdata;
          R_CMDQ_W2:     cmdq_buf[95:64]  <= host_wdata;
          R_CMDQ_W3:     cmdq_buf[127:96] <= host_wdata;
          R_MEM_ADDR_LO: mem_addr[31:0] <= host_wdata;
          R_MEM_ADDR_HI: mem_addr[ADDR_BITS-1:32] <= host_wdata[ADDR_BITS-33:0];
          R_MEM_SEL:     mem_sel <= host_wdata[0];
          default: ;
        endcase
      end
      if (cmdq_go)              cmdq_addr <= cmdq_addr + 12'd1;
      if (mem_commit || mem_adv) mem_addr  <= mem_addr + 1'b1;
    end
  end

  always_comb begin
    host_rdata = 32'b0;
    if (host_ren) begin
      case (host_addr)
        R_STATUS:    host_rdata = {30'b0, running, done};
        R_TOTAL_LO:  host_rdata = total_cycles[31:0];
        R_TOTAL_HI:  host_rdata = total_cycles[63:32];
        R_TRACE:     host_rdata = {trace_index, 15'b0, trace_valid};
        R_TRACE_CYC: host_rdata = trace_cycles;
        R_MEM_RDATA: host_rdata = {24'b0, hd_rdata};
        default:     host_rdata = 32'b0;
      endcase
    end
  end

endmodule

`endif // HOST_IF_SV
