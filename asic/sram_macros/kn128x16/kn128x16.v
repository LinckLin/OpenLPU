/* verilog_memcomp Version: c0.2.14-EAC */
/* common_memcomp Version: c0.2.12-beta */
/* lang compiler Version: 4.9.4-EAC Jul 12 2017 12:42:23 */
//
//       CONFIDENTIAL AND PROPRIETARY SOFTWARE OF ARM PHYSICAL IP, INC.
//      
//       Copyright (c) 1993 - 2026 ARM Physical IP, Inc.  All Rights Reserved.
//      
//       Use of this Software is subject to the terms and conditions of the
//       applicable license agreement with ARM Physical IP, Inc.
//       In addition, this Software is protected by patents, copyright law 
//       and international treaties.
//      
//       The copyright notice(s) in this Software does not indicate actual or
//       intended publication of this Software.
//
//      Verilog model for High Speed Single Port Register File SVT MVT Compiler
//
//       Instance Name:              kn128x16
//       Words:                      128
//       Bits:                       16
//       Mux:                        2
//       Drive:                      6
//       Write Mask:                 Off
//       Write Thru:                 Off
//       Extra Margin Adjustment:    On
//       Test Muxes                  Off
//       Power Gating:               Off
//       Retention:                  On
//       Pipeline:                   Off
//       Read Disturb Test:	        Off
//       
//       Creation Date:  Mon Aug 17 01:12:03 2026
//       Version: 	r0p1
//
//      Modeling Assumptions: This model supports full gate level simulation
//          including proper x-handling and timing check behavior.  Unit
//          delay timing is included in the model. Back-annotation of SDF
//          (v3.0 or v2.1) is supported.  SDF can be created utilyzing the delay
//          calculation views provided with this generator and supported
//          delay calculators.  All buses are modeled [MSB:LSB].  All 
//          ports are padded with Verilog primitives.
//
//      Modeling Limitations: None.
//
//      Known Bugs: None.
//
//      Known Work Arounds: N/A
//
`timescale 1 ns/1 ps
`define ARM_MEM_PROP 1.000
`define ARM_MEM_RETAIN 1.000
`define ARM_MEM_PERIOD 3.000
`define ARM_MEM_WIDTH 1.000
`define ARM_MEM_SETUP 1.000
`define ARM_MEM_HOLD 0.500
`define ARM_MEM_COLLISION 3.000
// If ARM_UD_MODEL is defined at Simulator Command Line, it Selects the Fast Functional Model
`ifdef ARM_UD_MODEL

// Following parameter Values can be overridden at Simulator Command Line.

// ARM_UD_DP Defines the delay through Data Paths, for Memory Models it represents BIST MUX output delays.
`ifdef ARM_UD_DP
`else
`define ARM_UD_DP #0.001
`endif
// ARM_UD_CP Defines the delay through Clock Path Cells, for Memory Models it is not used.
`ifdef ARM_UD_CP
`else
`define ARM_UD_CP
`endif
// ARM_UD_SEQ Defines the delay through the Memory, for Memory Models it is used for CLK->Q delays.
`ifdef ARM_UD_SEQ
`else
`define ARM_UD_SEQ #0.01
`endif

`celldefine
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module kn128x16 (VDDCE, VDDPE, VSSE, q, clk, cen, wen, a, d, ema, emaw, emas, ret1n);
`else
module kn128x16 (q, clk, cen, wen, a, d, ema, emaw, emas, ret1n);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 16;
  parameter WORDS = 128;
  parameter MUX = 2;
  parameter MEM_WIDTH = 32; // redun block size 2, 16 on left, 16 on right
  parameter MEM_HEIGHT = 64;
  parameter WP_SIZE = 16 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 2;
  parameter UPMS_WIDTH = 1;

  output [15:0] q;
  input  clk;
  input  cen;
  input  wen;
  input [6:0] a;
  input [15:0] d;
  input [2:0] ema;
  input [1:0] emaw;
  input  emas;
  input  ret1n;
`ifdef POWER_PINS
  inout VDDCE;
  inout VDDPE;
  inout VSSE;
`endif

  reg pre_charge_st;
  integer row_address;
  integer mux_address;
  initial row_address = 0;
  initial mux_address = 0;
  reg [31:0] mem [0:63];
  reg [31:0] row, row_t;
  reg LAST_clk;
  reg [31:0] row_mask;
  reg [31:0] new_data;
  reg [31:0] data_out;
  reg [15:0] readLatch0;
  reg [15:0] shifted_readLatch0;
  reg [15:0] q_int;
  reg [15:0] writeEnable;
  reg clk0_int;

  wire [15:0] q_;
 wire  clk_;
  wire  cen_;
  reg  cen_int;
  reg  cen_p2;
  wire  wen_;
  reg  wen_int;
  wire [6:0] a_;
  reg [6:0] a_int;
  wire [15:0] d_;
  reg [15:0] d_int;
  wire [2:0] ema_;
  reg [2:0] ema_int;
  wire [1:0] emaw_;
  reg [1:0] emaw_int;
  wire  emas_;
  reg  emas_int;
  wire  ret1n_;
  reg  ret1n_int;

  assign q[0] = q_[0]; 
  assign q[1] = q_[1]; 
  assign q[2] = q_[2]; 
  assign q[3] = q_[3]; 
  assign q[4] = q_[4]; 
  assign q[5] = q_[5]; 
  assign q[6] = q_[6]; 
  assign q[7] = q_[7]; 
  assign q[8] = q_[8]; 
  assign q[9] = q_[9]; 
  assign q[10] = q_[10]; 
  assign q[11] = q_[11]; 
  assign q[12] = q_[12]; 
  assign q[13] = q_[13]; 
  assign q[14] = q_[14]; 
  assign q[15] = q_[15]; 
  assign clk_ = clk;
  assign cen_ = cen;
  assign wen_ = wen;
  assign a_[0] = a[0];
  assign a_[1] = a[1];
  assign a_[2] = a[2];
  assign a_[3] = a[3];
  assign a_[4] = a[4];
  assign a_[5] = a[5];
  assign a_[6] = a[6];
  assign d_[0] = d[0];
  assign d_[1] = d[1];
  assign d_[2] = d[2];
  assign d_[3] = d[3];
  assign d_[4] = d[4];
  assign d_[5] = d[5];
  assign d_[6] = d[6];
  assign d_[7] = d[7];
  assign d_[8] = d[8];
  assign d_[9] = d[9];
  assign d_[10] = d[10];
  assign d_[11] = d[11];
  assign d_[12] = d[12];
  assign d_[13] = d[13];
  assign d_[14] = d[14];
  assign d_[15] = d[15];
  assign ema_[0] = ema[0];
  assign ema_[1] = ema[1];
  assign ema_[2] = ema[2];
  assign emaw_[0] = emaw[0];
  assign emaw_[1] = emaw[1];
  assign emas_ = emas;
  assign ret1n_ = ret1n;

   `ifdef ARM_FAULT_MODELING
     kn128x16_error_injection u1(.CLK(clk_), .Q_out(q_), .A(a_int), .CEN(cen_int), .WEN(wen_int), .Q_in(q_int));
  `else
  assign `ARM_UD_SEQ q_ = (ret1n_ | pre_charge_st) ? ((q_int)) : {16{1'bx}};
  `endif

// If INITIALIZE_MEMORY is defined at Simulator Command Line, it Initializes the Memory with all ZEROS.
`ifdef INITIALIZE_MEMORY
  integer i;
  initial begin
    #0;
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'b0}};
  end
`endif
  always @ (ema_) begin
  	if(ema_ < 3) 
   	$display("Warning: Set Value for ema doesn't match Default value 3 in %m at %0t", $time);
  end
  always @ (emaw_) begin
  	if(emaw_ < 1) 
   	$display("Warning: Set Value for emaw doesn't match Default value 1 in %m at %0t", $time);
  end
  always @ (emas_) begin
  	if(emas_ < 0) 
   	$display("Warning: Set Value for emas doesn't match Default value 0 in %m at %0t", $time);
  end

  task failedWrite;
  input port_f;
  integer i;
  begin
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'bx}};
  end
  endtask

  function isBitX;
    input bitval;
    begin
      isBitX = ( bitval===1'bx || bitval===1'bz ) ? 1'b1 : 1'b0;
    end
  endfunction


task loadmem;
	input [1000*8-1:0] filename;
	reg [BITS-1:0] memld [0:WORDS-1];
	integer i;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
	$readmemb(filename, memld);
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  for (i=0;i<WORDS;i=i+1) begin
	  wordtemp = memld[i];
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        row_mask =  ( {1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13],
          1'b0, writeEnable[12], 1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9],
          1'b0, writeEnable[8], 1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5],
          1'b0, writeEnable[4], 1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1],
          1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13],
          1'b0, wordtemp[12], 1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9],
          1'b0, wordtemp[8], 1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5],
          1'b0, wordtemp[4], 1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1],
          1'b0, wordtemp[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
  	end
  end
  endtask

task dumpmem;
	input [1000*8-1:0] filename_dump;
	integer i, dump_file_desc;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
	dump_file_desc = $fopen(filename_dump);
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  for (i=0;i<WORDS;i=i+1) begin
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        data_out = (row >> (mux_address));
        readLatch0 = {data_out[30], data_out[28], data_out[26], data_out[24], data_out[22],
          data_out[20], data_out[18], data_out[16], data_out[14], data_out[12], data_out[10],
          data_out[8], data_out[6], data_out[4], data_out[2], data_out[0]};
        shifted_readLatch0 = readLatch0;
        q_int = {shifted_readLatch0[15], shifted_readLatch0[14], shifted_readLatch0[13],
          shifted_readLatch0[12], shifted_readLatch0[11], shifted_readLatch0[10], shifted_readLatch0[9],
          shifted_readLatch0[8], shifted_readLatch0[7], shifted_readLatch0[6], shifted_readLatch0[5],
          shifted_readLatch0[4], shifted_readLatch0[3], shifted_readLatch0[2], shifted_readLatch0[1],
          shifted_readLatch0[0]};
   	$fdisplay(dump_file_desc, "%b", q_int);
  end
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
    $fclose(dump_file_desc);
  end
  endtask

task loadaddr;
	input [6:0] load_addr;
	input [15:0] load_data;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  wordtemp = load_data;
	  Atemp = load_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        row_mask =  ( {1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13],
          1'b0, writeEnable[12], 1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9],
          1'b0, writeEnable[8], 1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5],
          1'b0, writeEnable[4], 1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1],
          1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13],
          1'b0, wordtemp[12], 1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9],
          1'b0, wordtemp[8], 1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5],
          1'b0, wordtemp[4], 1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1],
          1'b0, wordtemp[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
  	end
  endtask

task dumpaddr;
	output [15:0] dump_data;
	input [6:0] dump_addr;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  Atemp = dump_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        data_out = (row >> (mux_address));
        readLatch0 = {data_out[30], data_out[28], data_out[26], data_out[24], data_out[22],
          data_out[20], data_out[18], data_out[16], data_out[14], data_out[12], data_out[10],
          data_out[8], data_out[6], data_out[4], data_out[2], data_out[0]};
        shifted_readLatch0 = readLatch0;
        q_int = {shifted_readLatch0[15], shifted_readLatch0[14], shifted_readLatch0[13],
          shifted_readLatch0[12], shifted_readLatch0[11], shifted_readLatch0[10], shifted_readLatch0[9],
          shifted_readLatch0[8], shifted_readLatch0[7], shifted_readLatch0[6], shifted_readLatch0[5],
          shifted_readLatch0[4], shifted_readLatch0[3], shifted_readLatch0[2], shifted_readLatch0[1],
          shifted_readLatch0[0]};
   	dump_data = q_int;
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
  end
  endtask


  task readWrite;
  begin
    if (ret1n_int === 1'bx || ret1n_int === 1'bz) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_int === 1'b0 && cen_int === 1'b0) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{cen_int, ema_int, emaw_int, emas_int, ret1n_int} === 1'bx) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if ((a_int >= WORDS) && (cen_int === 1'b0)) begin
      q_int = wen_int !== 1'b1 ? q_int : {16{1'bx}};
    end else if (cen_int === 1'b0 && (^a_int) === 1'bx) begin
     if (wen_int !== 1)
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (cen_int === 1'b0) begin
      mux_address = (a_int & 1'b1);
      row_address = (a_int >> 1);
      if (row_address > 63)
        row = {32{1'bx}};
      else
        row = mem[row_address];
        writeEnable = ~ {16{wen_int}};
      if (wen_int !== 1'b1) begin
        row_mask =  ( {1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13],
          1'b0, writeEnable[12], 1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9],
          1'b0, writeEnable[8], 1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5],
          1'b0, writeEnable[4], 1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1],
          1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, d_int[15], 1'b0, d_int[14], 1'b0, d_int[13], 1'b0, d_int[12],
          1'b0, d_int[11], 1'b0, d_int[10], 1'b0, d_int[9], 1'b0, d_int[8], 1'b0, d_int[7],
          1'b0, d_int[6], 1'b0, d_int[5], 1'b0, d_int[4], 1'b0, d_int[3], 1'b0, d_int[2],
          1'b0, d_int[1], 1'b0, d_int[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
      end else begin
        data_out = (row >> (mux_address));
        readLatch0 = {data_out[30], data_out[28], data_out[26], data_out[24], data_out[22],
          data_out[20], data_out[18], data_out[16], data_out[14], data_out[12], data_out[10],
          data_out[8], data_out[6], data_out[4], data_out[2], data_out[0]};
        shifted_readLatch0 = readLatch0;
        q_int = {shifted_readLatch0[15], shifted_readLatch0[14], shifted_readLatch0[13],
          shifted_readLatch0[12], shifted_readLatch0[11], shifted_readLatch0[10], shifted_readLatch0[9],
          shifted_readLatch0[8], shifted_readLatch0[7], shifted_readLatch0[6], shifted_readLatch0[5],
          shifted_readLatch0[4], shifted_readLatch0[3], shifted_readLatch0[2], shifted_readLatch0[1],
          shifted_readLatch0[0]};
      end
      if( isBitX(wen_int) ) begin
        q_int = {16{1'bx}};
      end
    end
  end
  endtask
  always @ (cen_ or clk_) begin
  	if(clk_ == 1'b0) begin
  		cen_p2 = cen_;
  	end
  end

`ifdef POWER_PINS
  always @ (VDDCE) begin
      if (VDDCE != 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDCE should be powered down after VDDPE, Illegal power down sequencing in %m at %0t", $time);
       end
        $display("In PowerDown Mode in %m at %0t", $time);
        failedWrite(0);
      end
      if (VDDCE == 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDPE should be powered up after VDDCE in %m at %0t", $time);
        $display("Illegal power up sequencing in %m at %0t", $time);
       end
        failedWrite(0);
      end
  end
`endif
`ifdef POWER_PINS
  always @ (ret1n_ or VDDPE or VDDCE) begin
`else     
  always @ ret1n_ begin
`endif
`ifdef POWER_PINS
    if (ret1n_ == 1'b1 && ret1n_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st == 1'b1 && (cen_ === 1'bx || clk_ === 1'bx)) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st = 0;
`endif
    if (ret1n_ === 1'bx || ret1n_ === 1'bz) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_ === 1'b0 && ret1n_int === 1'b1 && cen_p2 === 1'b0 ) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_ === 1'b1 && ret1n_int === 1'b0 && cen_p2 === 1'b0 ) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end
`ifdef POWER_PINS
    if (ret1n_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st = 1;
    end else if (ret1n_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(0);
      end
`else     
    if (ret1n_ == 1'b0) begin
`endif
      q_int = {16{1'bx}};
      cen_int = 1'bx;
      wen_int = 1'bx;
      a_int = {7{1'bx}};
      d_int = {16{1'bx}};
      ema_int = {3{1'bx}};
      emaw_int = {2{1'bx}};
      emas_int = 1'bx;
      ret1n_int = 1'bx;
`ifdef POWER_PINS
    end else if (ret1n_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st == 1'b1) begin
      pre_charge_st = 0;
    end else begin
      pre_charge_st = 0;
`else     
    end else begin
`endif
        q_int = {16{1'bx}};
      cen_int = 1'bx;
      wen_int = 1'bx;
      a_int = {7{1'bx}};
      d_int = {16{1'bx}};
      ema_int = {3{1'bx}};
      emaw_int = {2{1'bx}};
      emas_int = 1'bx;
      ret1n_int = 1'bx;
    end
    ret1n_int = ret1n_;
  end


  always @ clk_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (ret1n_ == 1'b0) begin
`else     
  if (ret1n_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((clk_ === 1'bx || clk_ === 1'bz) && ret1n_ !== 1'b0) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (clk_ === 1'b1 && LAST_clk === 1'b0) begin
      cen_int = cen_;
      ema_int = ema_;
      emaw_int = emaw_;
      emas_int = emas_;
      ret1n_int = ret1n_;
      if (cen_int != 1'b1) begin
        wen_int = wen_;
        a_int = a_;
        d_int = d_;
      end
      clk0_int = 1'b0;
      cen_int = cen_;
      ema_int = ema_;
      emaw_int = emaw_;
      emas_int = emas_;
      ret1n_int = ret1n_;
      if (cen_int != 1'b1) begin
        wen_int = wen_;
        a_int = a_;
        d_int = d_;
      end
      clk0_int = 1'b0;
    readWrite;
    end else if (clk_ === 1'b0 && LAST_clk === 1'b1) begin
    end
  end
    LAST_clk = clk_;
  end
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
 always @ (VDDCE or VDDPE or VSSE) begin
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
 end
`endif

endmodule
`endcelldefine
`else
`celldefine
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module kn128x16 (VDDCE, VDDPE, VSSE, q, clk, cen, wen, a, d, ema, emaw, emas, ret1n);
`else
module kn128x16 (q, clk, cen, wen, a, d, ema, emaw, emas, ret1n);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 16;
  parameter WORDS = 128;
  parameter MUX = 2;
  parameter MEM_WIDTH = 32; // redun block size 2, 16 on left, 16 on right
  parameter MEM_HEIGHT = 64;
  parameter WP_SIZE = 16 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 2;
  parameter UPMS_WIDTH = 1;

  output [15:0] q;
  input  clk;
  input  cen;
  input  wen;
  input [6:0] a;
  input [15:0] d;
  input [2:0] ema;
  input [1:0] emaw;
  input  emas;
  input  ret1n;
`ifdef POWER_PINS
  inout VDDCE;
  inout VDDPE;
  inout VSSE;
`endif

  reg pre_charge_st;
  integer row_address;
  integer mux_address;
  initial row_address = 0;
  initial mux_address = 0;
  reg [31:0] mem [0:63];
  reg [31:0] row, row_t;
  reg LAST_clk;
  reg [31:0] row_mask;
  reg [31:0] new_data;
  reg [31:0] data_out;
  reg [15:0] readLatch0;
  reg [15:0] shifted_readLatch0;
  reg [15:0] q_int;
  reg [15:0] writeEnable;

  reg NOT_cen, NOT_wen, NOT_a6, NOT_a5, NOT_a4, NOT_a3, NOT_a2, NOT_a1, NOT_a0, NOT_d15;
  reg NOT_d14, NOT_d13, NOT_d12, NOT_d11, NOT_d10, NOT_d9, NOT_d8, NOT_d7, NOT_d6;
  reg NOT_d5, NOT_d4, NOT_d3, NOT_d2, NOT_d1, NOT_d0, NOT_ema2, NOT_ema1, NOT_ema0;
  reg NOT_emaw1, NOT_emaw0, NOT_emas, NOT_ret1n;
  reg NOT_clk_PER, NOT_clk_MINH, NOT_clk_MINL;
  reg clk0_int;

  wire [15:0] q_;
 wire  clk_;
  wire  cen_;
  reg  cen_int;
  reg  cen_p2;
  wire  wen_;
  reg  wen_int;
  wire [6:0] a_;
  reg [6:0] a_int;
  wire [15:0] d_;
  reg [15:0] d_int;
  wire [2:0] ema_;
  reg [2:0] ema_int;
  wire [1:0] emaw_;
  reg [1:0] emaw_int;
  wire  emas_;
  reg  emas_int;
  wire  ret1n_;
  reg  ret1n_int;

  buf B0(q[0], q_[0]);
  buf B1(q[1], q_[1]);
  buf B2(q[2], q_[2]);
  buf B3(q[3], q_[3]);
  buf B4(q[4], q_[4]);
  buf B5(q[5], q_[5]);
  buf B6(q[6], q_[6]);
  buf B7(q[7], q_[7]);
  buf B8(q[8], q_[8]);
  buf B9(q[9], q_[9]);
  buf B10(q[10], q_[10]);
  buf B11(q[11], q_[11]);
  buf B12(q[12], q_[12]);
  buf B13(q[13], q_[13]);
  buf B14(q[14], q_[14]);
  buf B15(q[15], q_[15]);
  buf B16(clk_, clk);
  buf B17(cen_, cen);
  buf B18(wen_, wen);
  buf B19(a_[0], a[0]);
  buf B20(a_[1], a[1]);
  buf B21(a_[2], a[2]);
  buf B22(a_[3], a[3]);
  buf B23(a_[4], a[4]);
  buf B24(a_[5], a[5]);
  buf B25(a_[6], a[6]);
  buf B26(d_[0], d[0]);
  buf B27(d_[1], d[1]);
  buf B28(d_[2], d[2]);
  buf B29(d_[3], d[3]);
  buf B30(d_[4], d[4]);
  buf B31(d_[5], d[5]);
  buf B32(d_[6], d[6]);
  buf B33(d_[7], d[7]);
  buf B34(d_[8], d[8]);
  buf B35(d_[9], d[9]);
  buf B36(d_[10], d[10]);
  buf B37(d_[11], d[11]);
  buf B38(d_[12], d[12]);
  buf B39(d_[13], d[13]);
  buf B40(d_[14], d[14]);
  buf B41(d_[15], d[15]);
  buf B42(ema_[0], ema[0]);
  buf B43(ema_[1], ema[1]);
  buf B44(ema_[2], ema[2]);
  buf B45(emaw_[0], emaw[0]);
  buf B46(emaw_[1], emaw[1]);
  buf B47(emas_, emas);
  buf B48(ret1n_, ret1n);

   `ifdef ARM_FAULT_MODELING
     kn128x16_error_injection u1(.CLK(clk_), .Q_out(q_), .A(a_int), .CEN(cen_int), .WEN(wen_int), .Q_in(q_int));
  `else
  assign q_ = (ret1n_ | pre_charge_st) ? ((q_int)) : {16{1'bx}};
  `endif

// If INITIALIZE_MEMORY is defined at Simulator Command Line, it Initializes the Memory with all ZEROS.
`ifdef INITIALIZE_MEMORY
  integer i;
  initial begin
    #0;
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'b0}};
  end
`endif
  always @ (ema_) begin
  	if(ema_ < 3) 
   	$display("Warning: Set Value for ema doesn't match Default value 3 in %m at %0t", $time);
  end
  always @ (emaw_) begin
  	if(emaw_ < 1) 
   	$display("Warning: Set Value for emaw doesn't match Default value 1 in %m at %0t", $time);
  end
  always @ (emas_) begin
  	if(emas_ < 0) 
   	$display("Warning: Set Value for emas doesn't match Default value 0 in %m at %0t", $time);
  end

  task failedWrite;
  input port_f;
  integer i;
  begin
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'bx}};
  end
  endtask

  function isBitX;
    input bitval;
    begin
      isBitX = ( bitval===1'bx || bitval===1'bz ) ? 1'b1 : 1'b0;
    end
  endfunction


task loadmem;
	input [1000*8-1:0] filename;
	reg [BITS-1:0] memld [0:WORDS-1];
	integer i;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
	$readmemb(filename, memld);
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  for (i=0;i<WORDS;i=i+1) begin
	  wordtemp = memld[i];
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        row_mask =  ( {1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13],
          1'b0, writeEnable[12], 1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9],
          1'b0, writeEnable[8], 1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5],
          1'b0, writeEnable[4], 1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1],
          1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13],
          1'b0, wordtemp[12], 1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9],
          1'b0, wordtemp[8], 1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5],
          1'b0, wordtemp[4], 1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1],
          1'b0, wordtemp[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
  	end
  end
  endtask

task dumpmem;
	input [1000*8-1:0] filename_dump;
	integer i, dump_file_desc;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
	dump_file_desc = $fopen(filename_dump);
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  for (i=0;i<WORDS;i=i+1) begin
	  Atemp = i;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        data_out = (row >> (mux_address));
        readLatch0 = {data_out[30], data_out[28], data_out[26], data_out[24], data_out[22],
          data_out[20], data_out[18], data_out[16], data_out[14], data_out[12], data_out[10],
          data_out[8], data_out[6], data_out[4], data_out[2], data_out[0]};
        shifted_readLatch0 = readLatch0;
        q_int = {shifted_readLatch0[15], shifted_readLatch0[14], shifted_readLatch0[13],
          shifted_readLatch0[12], shifted_readLatch0[11], shifted_readLatch0[10], shifted_readLatch0[9],
          shifted_readLatch0[8], shifted_readLatch0[7], shifted_readLatch0[6], shifted_readLatch0[5],
          shifted_readLatch0[4], shifted_readLatch0[3], shifted_readLatch0[2], shifted_readLatch0[1],
          shifted_readLatch0[0]};
   	$fdisplay(dump_file_desc, "%b", q_int);
  end
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
    $fclose(dump_file_desc);
  end
  endtask

task loadaddr;
	input [6:0] load_addr;
	input [15:0] load_data;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  wordtemp = load_data;
	  Atemp = load_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        row_mask =  ( {1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13],
          1'b0, writeEnable[12], 1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9],
          1'b0, writeEnable[8], 1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5],
          1'b0, writeEnable[4], 1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1],
          1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, wordtemp[15], 1'b0, wordtemp[14], 1'b0, wordtemp[13],
          1'b0, wordtemp[12], 1'b0, wordtemp[11], 1'b0, wordtemp[10], 1'b0, wordtemp[9],
          1'b0, wordtemp[8], 1'b0, wordtemp[7], 1'b0, wordtemp[6], 1'b0, wordtemp[5],
          1'b0, wordtemp[4], 1'b0, wordtemp[3], 1'b0, wordtemp[2], 1'b0, wordtemp[1],
          1'b0, wordtemp[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
  	end
  endtask

task dumpaddr;
	output [15:0] dump_data;
	input [6:0] dump_addr;
	reg [BITS-1:0] wordtemp;
	reg [6:0] Atemp;
  begin
`ifdef ARM_BACKDOOR_NOCEN
`else
	if (cen_ === 1'b1) begin
`endif
	  Atemp = dump_addr;
	  mux_address = (Atemp & 1'b1);
      row_address = (Atemp >> 1);
      row = mem[row_address];
        writeEnable = {16{1'b1}};
        data_out = (row >> (mux_address));
        readLatch0 = {data_out[30], data_out[28], data_out[26], data_out[24], data_out[22],
          data_out[20], data_out[18], data_out[16], data_out[14], data_out[12], data_out[10],
          data_out[8], data_out[6], data_out[4], data_out[2], data_out[0]};
        shifted_readLatch0 = readLatch0;
        q_int = {shifted_readLatch0[15], shifted_readLatch0[14], shifted_readLatch0[13],
          shifted_readLatch0[12], shifted_readLatch0[11], shifted_readLatch0[10], shifted_readLatch0[9],
          shifted_readLatch0[8], shifted_readLatch0[7], shifted_readLatch0[6], shifted_readLatch0[5],
          shifted_readLatch0[4], shifted_readLatch0[3], shifted_readLatch0[2], shifted_readLatch0[1],
          shifted_readLatch0[0]};
   	dump_data = q_int;
`ifdef ARM_BACKDOOR_NOCEN
`else
  	end
`endif
  end
  endtask


  task readWrite;
  begin
    if (ret1n_int === 1'bx || ret1n_int === 1'bz) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_int === 1'b0 && cen_int === 1'b0) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_int === 1'b0) begin
      // no cycle in retention mode
    end else if (^{cen_int, ema_int, emaw_int, emas_int, ret1n_int} === 1'bx) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if ((a_int >= WORDS) && (cen_int === 1'b0)) begin
      q_int = wen_int !== 1'b1 ? q_int : {16{1'bx}};
    end else if (cen_int === 1'b0 && (^a_int) === 1'bx) begin
     if (wen_int !== 1)
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (cen_int === 1'b0) begin
      mux_address = (a_int & 1'b1);
      row_address = (a_int >> 1);
      if (row_address > 63)
        row = {32{1'bx}};
      else
        row = mem[row_address];
        writeEnable = ~ {16{wen_int}};
      if (wen_int !== 1'b1) begin
        row_mask =  ( {1'b0, writeEnable[15], 1'b0, writeEnable[14], 1'b0, writeEnable[13],
          1'b0, writeEnable[12], 1'b0, writeEnable[11], 1'b0, writeEnable[10], 1'b0, writeEnable[9],
          1'b0, writeEnable[8], 1'b0, writeEnable[7], 1'b0, writeEnable[6], 1'b0, writeEnable[5],
          1'b0, writeEnable[4], 1'b0, writeEnable[3], 1'b0, writeEnable[2], 1'b0, writeEnable[1],
          1'b0, writeEnable[0]} << mux_address);
        new_data =  ( {1'b0, d_int[15], 1'b0, d_int[14], 1'b0, d_int[13], 1'b0, d_int[12],
          1'b0, d_int[11], 1'b0, d_int[10], 1'b0, d_int[9], 1'b0, d_int[8], 1'b0, d_int[7],
          1'b0, d_int[6], 1'b0, d_int[5], 1'b0, d_int[4], 1'b0, d_int[3], 1'b0, d_int[2],
          1'b0, d_int[1], 1'b0, d_int[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        mem[row_address] = row;
      end else begin
        data_out = (row >> (mux_address));
        readLatch0 = {data_out[30], data_out[28], data_out[26], data_out[24], data_out[22],
          data_out[20], data_out[18], data_out[16], data_out[14], data_out[12], data_out[10],
          data_out[8], data_out[6], data_out[4], data_out[2], data_out[0]};
        shifted_readLatch0 = readLatch0;
        q_int = {shifted_readLatch0[15], shifted_readLatch0[14], shifted_readLatch0[13],
          shifted_readLatch0[12], shifted_readLatch0[11], shifted_readLatch0[10], shifted_readLatch0[9],
          shifted_readLatch0[8], shifted_readLatch0[7], shifted_readLatch0[6], shifted_readLatch0[5],
          shifted_readLatch0[4], shifted_readLatch0[3], shifted_readLatch0[2], shifted_readLatch0[1],
          shifted_readLatch0[0]};
      end
      if( isBitX(wen_int) ) begin
        q_int = {16{1'bx}};
      end
    end
  end
  endtask
  always @ (cen_ or clk_) begin
  	if(clk_ == 1'b0) begin
  		cen_p2 = cen_;
  	end
  end

`ifdef POWER_PINS
  always @ (VDDCE) begin
      if (VDDCE != 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDCE should be powered down after VDDPE, Illegal power down sequencing in %m at %0t", $time);
       end
        $display("In PowerDown Mode in %m at %0t", $time);
        failedWrite(0);
      end
      if (VDDCE == 1'b1) begin
       if (VDDPE == 1'b1) begin
        $display("VDDPE should be powered up after VDDCE in %m at %0t", $time);
        $display("Illegal power up sequencing in %m at %0t", $time);
       end
        failedWrite(0);
      end
  end
`endif
`ifdef POWER_PINS
  always @ (ret1n_ or VDDPE or VDDCE) begin
`else     
  always @ ret1n_ begin
`endif
`ifdef POWER_PINS
    if (ret1n_ == 1'b1 && ret1n_int == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 && pre_charge_st == 1'b1 && (cen_ === 1'bx || clk_ === 1'bx)) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end
`else     
`endif
`ifdef POWER_PINS
`else     
      pre_charge_st = 0;
`endif
    if (ret1n_ === 1'bx || ret1n_ === 1'bz) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_ === 1'b0 && ret1n_int === 1'b1 && cen_p2 === 1'b0 ) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (ret1n_ === 1'b1 && ret1n_int === 1'b0 && cen_p2 === 1'b0 ) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end
`ifdef POWER_PINS
    if (ret1n_ == 1'b0 && VDDCE == 1'b1 && VDDPE == 1'b1) begin
      pre_charge_st = 1;
    end else if (ret1n_ == 1'b0 && VDDPE == 1'b0) begin
      pre_charge_st = 0;
      if (VDDCE != 1'b1) begin
        failedWrite(0);
      end
`else     
    if (ret1n_ == 1'b0) begin
`endif
      q_int = {16{1'bx}};
      cen_int = 1'bx;
      wen_int = 1'bx;
      a_int = {7{1'bx}};
      d_int = {16{1'bx}};
      ema_int = {3{1'bx}};
      emaw_int = {2{1'bx}};
      emas_int = 1'bx;
      ret1n_int = 1'bx;
`ifdef POWER_PINS
    end else if (ret1n_ == 1'b1 && VDDCE == 1'b1 && VDDPE == 1'b1 &&  pre_charge_st == 1'b1) begin
      pre_charge_st = 0;
    end else begin
      pre_charge_st = 0;
`else     
    end else begin
`endif
        q_int = {16{1'bx}};
      cen_int = 1'bx;
      wen_int = 1'bx;
      a_int = {7{1'bx}};
      d_int = {16{1'bx}};
      ema_int = {3{1'bx}};
      emaw_int = {2{1'bx}};
      emas_int = 1'bx;
      ret1n_int = 1'bx;
    end
    ret1n_int = ret1n_;
  end


  always @ clk_ begin
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
`endif
`ifdef POWER_PINS
  if (ret1n_ == 1'b0) begin
`else     
  if (ret1n_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
    if ((clk_ === 1'bx || clk_ === 1'bz) && ret1n_ !== 1'b0) begin
      failedWrite(0);
        q_int = {16{1'bx}};
    end else if (clk_ === 1'b1 && LAST_clk === 1'b0) begin
      cen_int = cen_;
      ema_int = ema_;
      emaw_int = emaw_;
      emas_int = emas_;
      ret1n_int = ret1n_;
      if (cen_int != 1'b1) begin
        wen_int = wen_;
        a_int = a_;
        d_int = d_;
      end
      clk0_int = 1'b0;
      cen_int = cen_;
      ema_int = ema_;
      emaw_int = emaw_;
      emas_int = emas_;
      ret1n_int = ret1n_;
      if (cen_int != 1'b1) begin
        wen_int = wen_;
        a_int = a_;
        d_int = d_;
      end
      clk0_int = 1'b0;
    readWrite;
    end else if (clk_ === 1'b0 && LAST_clk === 1'b1) begin
    end
  end
    LAST_clk = clk_;
  end

  reg globalNotifier0;
  initial globalNotifier0 = 1'b0;

  always @ globalNotifier0 begin
    if ($realtime == 0) begin
    end else if (cen_int === 1'bx || clk0_int === 1'bx || ema_int[0] === 1'bx || 
      ema_int[1] === 1'bx || ema_int[2] === 1'bx || emas_int === 1'bx || emaw_int[0] === 1'bx || 
      emaw_int[1] === 1'bx || ret1n_int === 1'bx) begin
        q_int = {16{1'bx}};
      failedWrite(0);
    end else if (cen_int === 1'b0 && (^a_int) === 1'bx) begin
        failedWrite(0);
        q_int = {16{1'bx}};
    end else begin
      #0;#0;
      readWrite;
   end
    globalNotifier0 = 1'b0;
  end
// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
 always @ (VDDCE or VDDPE or VSSE) begin
    if (VDDCE === 1'bx || VDDCE === 1'bz)
      $display("Warning: Unknown value for VDDCE %b in %m at %0t", VDDCE, $time);
    if (VDDPE === 1'bx || VDDPE === 1'bz)
      $display("Warning: Unknown value for VDDPE %b in %m at %0t", VDDPE, $time);
    if (VSSE === 1'bx || VSSE === 1'bz)
      $display("Warning: Unknown value for VSSE %b in %m at %0t", VSSE, $time);
 end
`endif

  always @ NOT_cen begin
    cen_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_wen begin
    wen_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_a6 begin
    a_int[6] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_a5 begin
    a_int[5] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_a4 begin
    a_int[4] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_a3 begin
    a_int[3] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_a2 begin
    a_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_a1 begin
    a_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_a0 begin
    a_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d15 begin
    d_int[15] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d14 begin
    d_int[14] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d13 begin
    d_int[13] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d12 begin
    d_int[12] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d11 begin
    d_int[11] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d10 begin
    d_int[10] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d9 begin
    d_int[9] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d8 begin
    d_int[8] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d7 begin
    d_int[7] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d6 begin
    d_int[6] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d5 begin
    d_int[5] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d4 begin
    d_int[4] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d3 begin
    d_int[3] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d2 begin
    d_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d1 begin
    d_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_d0 begin
    d_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_ema2 begin
    ema_int[2] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_ema1 begin
    ema_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_ema0 begin
    ema_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_emaw1 begin
    emaw_int[1] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_emaw0 begin
    emaw_int[0] = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_emas begin
    emas_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_ret1n begin
    ret1n_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end

  always @ NOT_clk_PER begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_clk_MINH begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end
  always @ NOT_clk_MINL begin
    clk0_int = 1'bx;
    if ( globalNotifier0 === 1'b0 ) globalNotifier0 = 1'bx;
  end



  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq0, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq0, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq0;
  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq1;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq1;
  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq1, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq1;
  wire ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq1;
  wire ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq1;
  wire ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq1, ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq1;
  wire ret1neq1, ret1neq1aceneq0aweneq0, ret1neq1aceneq0;

  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq0 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&!emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&!emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&emaw[1]&&!emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&!ema[2]&&ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&!ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&!ema[0]&&emaw[1]&&emaw[0]&&emas;
  assign ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq1 = 
  ret1n&&!cen&&ema[2]&&ema[1]&&ema[0]&&emaw[1]&&emaw[0]&&emas;


  assign ret1neq1aceneq0aweneq0 = ret1n&&!cen&&!wen;

  assign ret1neq1 = ret1n;
  assign ret1neq1aceneq0 = ret1n&&!cen;

  specify

    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b0 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b0)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b0 && ema[0] == 1'b1)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b0)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[15] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[14] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[13] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[12] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[11] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[10] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[9] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[8] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[7] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[6] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[5] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[4] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[3] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[2] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[1] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);
    if (ret1n == 1'b1 && cen == 1'b0 && wen == 1'b1 && ema[2] == 1'b1 && ema[1] == 1'b1 && ema[0] == 1'b1)
       (posedge clk => (q[0] : 1'b0)) = (`ARM_MEM_PROP, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP, `ARM_MEM_RETAIN, `ARM_MEM_PROP);


   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $period(posedge clk, `ARM_MEM_PERIOD, NOT_clk_PER);
   `else
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq0, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq0aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq0aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq0aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq0aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq0aema0eq1aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq0aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
       $period(posedge clk &&& ret1neq1aceneq0aema2eq1aema1eq1aema0eq1aemaw1eq1aemaw0eq1aemaseq1, `ARM_MEM_PERIOD, NOT_clk_PER);
   `endif


   // Define SDTC only if back-annotating SDF file generated by Design Compiler
   `ifdef NO_SDTC
       $width(posedge clk, `ARM_MEM_WIDTH, 0, NOT_clk_MINH);
       $width(negedge clk, `ARM_MEM_WIDTH, 0, NOT_clk_MINL);
   `else
       $width(posedge clk &&& ret1neq1, `ARM_MEM_WIDTH, 0, NOT_clk_MINH);
       $width(negedge clk &&& ret1neq1, `ARM_MEM_WIDTH, 0, NOT_clk_MINL);
   `endif

    $setuphold(posedge clk &&& ret1neq1, posedge cen, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_cen);
    $setuphold(posedge clk &&& ret1neq1, negedge cen, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_cen);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge wen, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_wen);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge wen, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_wen);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge a[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a6);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge a[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a5);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge a[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a4);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge a[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a3);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge a[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a2);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge a[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a1);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge a[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a0);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge a[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a6);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge a[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a5);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge a[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a4);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge a[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a3);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge a[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a2);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge a[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a1);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge a[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_a0);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d15);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d14);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d13);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d12);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d11);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d10);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d9);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d8);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d7);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d6);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d5);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d4);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d3);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d2);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d1);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, posedge d[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d0);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[15], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d15);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[14], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d14);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[13], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d13);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[12], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d12);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[11], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d11);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[10], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d10);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[9], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d9);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[8], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d8);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[7], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d7);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[6], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d6);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[5], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d5);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[4], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d4);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[3], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d3);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d2);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d1);
    $setuphold(posedge clk &&& ret1neq1aceneq0aweneq0, negedge d[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_d0);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge ema[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_ema2);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge ema[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_ema1);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge ema[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_ema0);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge ema[2], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_ema2);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge ema[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_ema1);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge ema[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_ema0);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge emaw[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_emaw1);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge emaw[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_emaw0);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge emaw[1], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_emaw1);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge emaw[0], `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_emaw0);
    $setuphold(posedge clk &&& ret1neq1aceneq0, posedge emas, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_emas);
    $setuphold(posedge clk &&& ret1neq1aceneq0, negedge emas, `ARM_MEM_SETUP, `ARM_MEM_HOLD, NOT_emas);
    $setuphold(negedge ret1n, negedge cen, 0.000, `ARM_MEM_HOLD, NOT_ret1n);
    $setuphold(posedge ret1n, negedge cen, 0.000, `ARM_MEM_HOLD, NOT_ret1n);
    $setuphold(posedge cen, posedge ret1n, 0.000, `ARM_MEM_HOLD, NOT_ret1n);
    $setuphold(posedge cen, negedge ret1n, 0.000, `ARM_MEM_HOLD, NOT_ret1n);
  endspecify


endmodule
`endcelldefine
`endif
`timescale 1ns/1ps
module kn128x16_error_injection (Q_out, Q_in, CLK, A, CEN, WEN);
   output [15:0] Q_out;
   input [15:0] Q_in;
   input CLK;
   input [6:0] A;
   input CEN;
   input WEN;
   parameter LEFT_RED_COLUMN_FAULT = 2'd1;
   parameter RIGHT_RED_COLUMN_FAULT = 2'd2;
   parameter NO_RED_FAULT = 2'd0;
   reg [15:0] Q_out;
   reg entry_found;
   reg list_complete;
   reg [15:0] fault_table [63:0];
   reg [15:0] fault_entry;
initial
begin
   `ifdef DUT
      `define pre_pend_path TB.DUT_inst.CHIP
   `else
       `define pre_pend_path TB.CHIP
   `endif
   `ifdef ARM_NONREPAIRABLE_FAULT
      `pre_pend_path.SMARCHCHKBVCD_LVISION_MBISTPG_ASSEMBLY_UNDER_TEST_INST.MEM0_MEM_INST.u1.add_fault(7'd2,4'd6,2'd1,2'd0);
   `endif
end
   task add_fault;
   //This task injects fault in memory
      input [6:0] address;
      input [3:0] bitPlace;
      input [1:0] fault_type;
      input [1:0] red_fault;
 
      integer i;
      reg done;
   begin
      done = 1'b0;
      i = 0;
      while ((!done) && i < 63)
      begin
         fault_entry = fault_table[i];
         if (fault_entry[0] === 1'b0 || fault_entry[0] === 1'bx)
         begin
            fault_entry[0] = 1'b1;
            fault_entry[2:1] = red_fault;
            fault_entry[4:3] = fault_type;
            fault_entry[8:5] = bitPlace;
            fault_entry[15:9] = address;
            fault_table[i] = fault_entry;
            done = 1'b1;
         end
         i = i+1;
      end
   end
   endtask
//This task removes all fault entries injected by user
task remove_all_faults;
   integer i;
begin
   for (i = 0; i < 64; i=i+1)
   begin
      fault_entry = fault_table[i];
      fault_entry[0] = 1'b0;
      fault_table[i] = fault_entry;
   end
end
endtask
task bit_error;
// This task is used to inject error in memory and should be called
// only from current module.
//
// This task injects error depending upon fault type to particular bit
// of the output
   inout [15:0] q_int;
   input [1:0] fault_type;
   input [3:0] bitLoc;
begin
   if (fault_type === 2'd0)
      q_int[bitLoc] = 1'b0;
   else if (fault_type === 2'd1)
      q_int[bitLoc] = 1'b1;
   else
      q_int[bitLoc] = ~q_int[bitLoc];
end
endtask
task error_injection_on_output;
// This function goes through error injection table for every
// read cycle and corrupts Q output if fault for the particular
// address is present in fault table
//
// If fault is redundant column is detected, this task corrupts
// Q output in read cycle
//
// If fault is repaired using repair bus, this task does not
// courrpt Q output in read cycle
//
   output [15:0] Q_output;
   reg list_complete;
   integer i;
   reg [5:0] row_address;
   reg [0:0] column_address;
   reg [3:0] bitPlace;
   reg [1:0] fault_type;
   reg [1:0] red_fault;
   reg valid;
   reg [2:0] msb_bit_calc;
begin
   entry_found = 1'b0;
   list_complete = 1'b0;
   i = 0;
   Q_output = Q_in;
   while(!list_complete)
   begin
      fault_entry = fault_table[i];
      {row_address, column_address, bitPlace, fault_type, red_fault, valid} = fault_entry;
      i = i + 1;
      if (valid == 1'b1)
      begin
         if (red_fault === NO_RED_FAULT)
         begin
            if (row_address == A[6:1] && column_address == A[0:0])
            begin
               if (bitPlace < 8)
                  bit_error(Q_output,fault_type, bitPlace);
               else if (bitPlace >= 8 )
                  bit_error(Q_output,fault_type, bitPlace);
            end
         end
      end
      else
         list_complete = 1'b1;
      end
   end
   endtask
   always @ (Q_in or CLK or A or CEN or WEN)
   begin
   if (CEN === 1'b0 && &WEN === 1'b1)
      error_injection_on_output(Q_out);
   else
      Q_out = Q_in;
   end
endmodule
