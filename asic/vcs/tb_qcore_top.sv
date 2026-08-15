// ============================================================================
// tb_qcore_top.sv — VCS functional testbench for qcore_top (co-sim).
//
// Replicates rtl/tb/sim_main.cpp (the Verilator harness) byte-for-byte at the
// binary file-format level, so VCS and Verilator results can be diffed
// bit-exactly (trace records + total cycles + final memory dump).
//
// The RTL has no DPI: all stimulus/observation goes through qcore_top's plain
// Verilog ports (imem backdoor + qmem backdoor + trace/done).  File formats
// (see docs/p7/rtl-report.md, rtl/tb/sim_main.cpp):
//   prog.bin     : 128-bit LE instructions (ninst = size/16)
//   preload.bin  : records sel[1] addr[8] nbytes[4] data[n]
//   dump_req.bin : records sel[1] addr[8] nbytes[4]
//   trace.bin    : 6-byte records (idx[2] cyc[4], LE)
//   total.bin    : 8-byte LE total_cycles
//   dump.bin     : concatenated dumped bytes
//
// VCS 4-state power-on vs Verilator 2-state power-on: the co-sim preload only
// emits *non-zero* SRAM runs, so the harness must start with an all-zero
// memory/register state to reproduce Verilator's 2-state semantics.  Run the
// compiled simv with:
//   +vcs+initreg+0 +vcs+initmem+0
// which initializes every 4-state variable and every unpacked-array memory to
// 0 (the reference simulator's power-on value).  Any remaining X that reaches
// a trace/dump byte is a real divergence and is reported, not suppressed.
// ============================================================================
`timescale 1ns/1ps

module tb_qcore_top;

  // -- DUT signals (all driven to a known value before use) -------------------
  logic        clk        = 0;
  logic        rst_n      = 0;
  logic        start      = 0;
  logic [11:0] imem_waddr = 0;
  logic        imem_we    = 0;
  logic [127:0] imem_wdata = 0;
  logic [15:0] prog_len   = 0;
  logic        bd_en      = 0;
  logic        bd_sel     = 0;
  logic [39:0] bd_addr    = 0;
  logic [7:0]  bd_wdata   = 0;
  logic [7:0]  bd_rdata;
  logic        done;
  logic [63:0] total_cycles;
  logic        trace_valid;
  logic [15:0] trace_index;
  logic [31:0] trace_cycles;

  // -- plusargs ---------------------------------------------------------------
  string prog_path, preload_path, dump_req_path, trace_path, total_path, dump_path;
  longint unsigned max_cycles;

  qcore_top #(.NINST(4096)) dut (
    .clk(clk), .rst_n(rst_n), .start(start),
    .imem_waddr(imem_waddr), .imem_we(imem_we), .imem_wdata(imem_wdata),
    .prog_len(prog_len),
    .bd_en(bd_en), .bd_sel(bd_sel), .bd_addr(bd_addr),
    .bd_wdata(bd_wdata), .bd_rdata(bd_rdata),
    .done(done), .total_cycles(total_cycles),
    .trace_valid(trace_valid), .trace_index(trace_index), .trace_cycles(trace_cycles)
  );

  // ---------------------------------------------------------------------------
  // clock tick — mirrors sim_main.cpp tick(): clk=0 eval, clk=1 eval.
  // ---------------------------------------------------------------------------
  task automatic tick();
    clk = 1'b0;
    #1;
    clk = 1'b1;
    #1;
  endtask

  // ---------------------------------------------------------------------------
  // binary file helpers
  // ---------------------------------------------------------------------------
  function automatic int fopen_rb(input string p);
    int f;
    f = $fopen(p, "rb");
    if (f == 0) begin
      $display("FATAL: cannot open %s", p);
      $finish(2);
    end
    return f;
  endfunction

  function automatic int fopen_wb(input string p);
    int f;
    f = $fopen(p, "wb");
    if (f == 0) begin
      $display("FATAL: cannot open %s", p);
      $finish(2);
    end
    return f;
  endfunction

  // $fgetc returns 0..255 or -1 (EOF); read one byte, fatal on unexpected EOF.
  function automatic int fgetc_or_die(input int f, input string what);
    int c;
    c = $fgetc(f);
    if (c < 0) begin
      $display("FATAL: truncated %s", what);
      $finish(2);
    end
    return c[7:0];
  endfunction

  // ---------------------------------------------------------------------------
  // read entire (small) file into a dynamic byte array
  // ---------------------------------------------------------------------------
  task automatic slurp(input string path, ref logic [7:0] data[]);
    int f, c, n;
    f = fopen_rb(path);
    data = new[0];
    forever begin
      c = $fgetc(f);
      if (c < 0) break;
      n = data.size();
      data = new[n+1](data);
      data[n] = c[7:0];
    end
    $fclose(f);
  endtask

  // ---------------------------------------------------------------------------
  // load program (128-bit LE instructions)
  // ---------------------------------------------------------------------------
  task automatic load_program(input string path);
    logic [7:0] prog[];
    int n, ninst, i, k;
    slurp(path, prog);
    n = prog.size();
    ninst = n / 16;
    prog_len = ninst[15:0];
    for (i = 0; i < ninst; i++) begin
      imem_we = 1'b1;
      imem_waddr = i[11:0];
      imem_wdata = 128'b0;
      for (k = 0; k < 4; k++) begin
        // word k = prog[i*16 + k*4 + 0..3], little-endian
        imem_wdata[k*32 +: 32] = { prog[i*16 + k*4 + 3],
                                   prog[i*16 + k*4 + 2],
                                   prog[i*16 + k*4 + 1],
                                   prog[i*16 + k*4 + 0] };
      end
      tick();
    end
    imem_we = 1'b0;
  endtask

  // ---------------------------------------------------------------------------
  // preload memory (records: sel[1] addr[8] nbytes[4] data[n])
  // ---------------------------------------------------------------------------
  task automatic preload(input string path);
    int f, c, b, i;
    logic [7:0]  sel;
    logic [63:0] addr;
    logic [31:0] n;
    f = fopen_rb(path);
    forever begin
      c = $fgetc(f);
      if (c < 0) break;              // clean EOF at a record boundary
      sel = c[7:0];
      addr = 64'b0;
      for (b = 0; b < 8; b++) addr |= 64'(fgetc_or_die(f, "preload")) << (8*b);
      n = 32'b0;
      for (b = 0; b < 4; b++) n |= 32'(fgetc_or_die(f, "preload")) << (8*b);
      for (i = 0; i < n; i++) begin
        c = fgetc_or_die(f, "preload");
        bd_en    = 1'b1;
        bd_sel   = sel[0];
        bd_addr  = addr[39:0] + i;
        bd_wdata = c[7:0];
        tick();
      end
    end
    $fclose(f);
    // Deassert the backdoor write port (sim_main.cpp: leaving bd_en=1 would
    // re-write the last preload byte every cycle and corrupt engine writes).
    bd_en = 1'b0;
    #1;
  endtask

  // ---------------------------------------------------------------------------
  // run + record trace + total cycles
  // ---------------------------------------------------------------------------
  task automatic run_body(input string trace_path, input string total_path);
    int tr_fd, tot_fd;
    longint unsigned c;
    logic finished;
    tr_fd = fopen_wb(trace_path);
    start = 1'b1;
    clk = 1'b0; #1;
    clk = 1'b1; #1;
    start = 1'b0;
    finished = 1'b0;
    c = 0;
    while (c < max_cycles && !finished) begin
      clk = 1'b0; #1;
      clk = 1'b1; #1;
      if (trace_valid) begin
        $fwrite(tr_fd, "%c%c%c%c%c%c",
                trace_index[7:0], trace_index[15:8],
                trace_cycles[7:0], trace_cycles[15:8],
                trace_cycles[23:16], trace_cycles[31:24]);
      end
      if (done) finished = 1'b1;
      c = c + 1;
    end
    $fclose(tr_fd);
    if (!finished) begin
      $display("TIMEOUT after %0d cycles", max_cycles);
      $finish(3);
    end
    tot_fd = fopen_wb(total_path);
    $fwrite(tot_fd, "%c%c%c%c%c%c%c%c",
            total_cycles[7:0],  total_cycles[15:8],  total_cycles[23:16], total_cycles[31:24],
            total_cycles[39:32], total_cycles[47:40], total_cycles[55:48], total_cycles[63:56]);
    $fclose(tot_fd);
  endtask

  // ---------------------------------------------------------------------------
  // dump requested regions (records: sel[1] addr[8] nbytes[4])
  // ---------------------------------------------------------------------------
  task automatic dump_regions(input string path, input string dump_path);
    int f, out_fd, c, b, i;
    logic [7:0]  sel;
    logic [63:0] addr;
    logic [31:0] n;
    f = fopen_rb(path);
    out_fd = fopen_wb(dump_path);
    forever begin
      c = $fgetc(f);
      if (c < 0) break;
      sel = c[7:0];
      addr = 64'b0;
      for (b = 0; b < 8; b++) addr |= 64'(fgetc_or_die(f, "dump_req")) << (8*b);
      n = 32'b0;
      for (b = 0; b < 4; b++) n |= 32'(fgetc_or_die(f, "dump_req")) << (8*b);
      for (i = 0; i < n; i++) begin
        bd_en   = 1'b0;
        bd_sel  = sel[0];
        bd_addr = addr[39:0] + i;
        #1;
        $fwrite(out_fd, "%c", bd_rdata[7:0]);
      end
    end
    $fclose(f);
    $fclose(out_fd);
  endtask

  // ---------------------------------------------------------------------------
  // main
  // ---------------------------------------------------------------------------
  initial begin
    if (!$value$plusargs("prog=%s", prog_path))       begin $display("FATAL: missing +prog");       $finish(2); end
    if (!$value$plusargs("preload=%s", preload_path)) begin $display("FATAL: missing +preload");    $finish(2); end
    if (!$value$plusargs("dump_req=%s", dump_req_path)) begin $display("FATAL: missing +dump_req"); $finish(2); end
    if (!$value$plusargs("trace=%s", trace_path))     begin $display("FATAL: missing +trace");      $finish(2); end
    if (!$value$plusargs("total=%s", total_path))     begin $display("FATAL: missing +total");      $finish(2); end
    if (!$value$plusargs("dump=%s", dump_path))       begin $display("FATAL: missing +dump");       $finish(2); end
    if (!$value$plusargs("max_cycles=%0d", max_cycles)) max_cycles = 100000000;

    // reset: 4 clocks with rst_n=0, then synchronous release (sim_main.cpp)
    clk = 1'b0; rst_n = 1'b0; start = 1'b0;
    imem_we = 1'b0; imem_waddr = 12'b0; imem_wdata = 128'b0; prog_len = 16'b0;
    bd_en = 1'b0; bd_sel = 1'b0; bd_addr = 40'b0; bd_wdata = 8'b0;
    #1;
    repeat (4) tick();
    rst_n = 1'b1;
    #1;

    load_program(prog_path);
    preload(preload_path);
    run_body(trace_path, total_path);
    dump_regions(dump_req_path, dump_path);

    $finish(0);
  end

endmodule
