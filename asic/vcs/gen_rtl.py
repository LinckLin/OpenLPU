#!/usr/bin/env python3
"""Generate a VCS-compatible snapshot of the frozen rtl/ (read-only) into
asic/vcs/gen/.

Two semantically-neutral transformations are applied (Verilator accepts both;
VCS O-2018.09 rejects them in `-sverilog` mode):

  1. command_processor.sv — hoist three groups of module-level `logic`
     declarations above their first use (forward reference):
       kv_base
       dma_rd_addr / dma_wr_addr / dma_rd_sel / dma_wr_sel / dma_wr_en / dma_wr_data
       op_rd_addr / op_wr_addr / op_rd_sel / op_wr_en / op_wr_sel / op_wr_data

  2. qmem.sv — merge the two `always_ff` blocks that write sram/hbm (engine
     write + backdoor write) into one process.  SV permits a single procedural
     driver per variable; the co-sim drives bd_en only during preload/dump and
     wr_en only during execution, so the two were mutually exclusive and the
     merge is bit-identical.

Zero functional change (same types, widths, statement ordering).  Everything
else is copied verbatim from rtl/ (read-only; snapshot lives under asic/vcs/gen/).
"""
from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "rtl")
DST = os.path.join(HERE, "gen")

# qcore_top's include closure (sram.sv is not instantiated by qcore_top).
NEEDED = [
    "qcore_top.sv",
    "qcore_pkg.sv",
    "qmem.sv",
    "command_processor.sv",
    "softfloat.sv",
    "matrix_engine.sv",
    "vector_engine.sv",
    "dma_engine.sv",
    "kv_addrgen.sv",
    "rope_lut.sv",
    "kv_bfeed.sv",
    "kv_quantdequant.sv",
    "sram_macros.sv",
]


def _move_before(text: str, decl: str, anchor: str, label: str) -> str:
    """Move `decl` (a multi-line block) to just before `anchor`."""
    if text.count(decl) != 1:
        sys.exit(f"[gen_rtl] {label}: expected exactly one declaration block, "
                 f"found {text.count(decl)}")
    if text.count(anchor) != 1:
        sys.exit(f"[gen_rtl] {label}: expected exactly one anchor, "
                 f"found {text.count(anchor)}")
    text = text.replace(decl, "", 1)
    text = text.replace(anchor, decl + anchor, 1)
    return text


def patch_command_processor(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()

    t = _move_before(
        t,
        "  logic [39:0] kv_base;\n",
        "  kv_addrgen u_kvg (\n",
        "kv_base",
    )

    dma_decl = (
        "  logic [39:0] dma_rd_addr, dma_wr_addr;\n"
        "  logic dma_rd_sel, dma_wr_sel, dma_wr_en;\n"
        "  logic [7:0] dma_wr_data;\n"
    )
    t = _move_before(t, dma_decl, "  dma_engine u_dma (\n", "dma_*")

    op_decl = (
        "  logic [39:0] op_rd_addr, op_wr_addr;\n"
        "  logic op_rd_sel, op_wr_en, op_wr_sel;\n"
        "  logic [7:0] op_wr_data;\n"
    )
    t = _move_before(
        t,
        op_decl,
        "  assign mem_rd_sel  = dma_active ? dma_rd_sel  : op_rd_sel;\n",
        "op_*",
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(t)


QMEM_OLD = """  // Engine synchronous write.
  always_ff @(posedge clk) begin
    if (wr_en) begin
      if (wr_sel)    hbm[wr_addr] <= wr_data;
      else           sram[wr_addr[22:0]] <= wr_data;
    end
  end

  // Backdoor read (combinational) + write (posedge).
  always_comb begin
    bd_rdata = 8'b0;
    if (bd_sel)      bd_rdata = hbm[bd_addr];
    else             bd_rdata = sram[bd_addr[22:0]];
  end

  always_ff @(posedge clk) begin
    if (bd_en) begin
      if (bd_sel)    hbm[bd_addr] <= bd_wdata;
      else           sram[bd_addr[22:0]] <= bd_wdata;
    end
  end
"""

QMEM_NEW = """  // Engine + backdoor synchronous writes merged into one process: SV permits
  // a single procedural driver per variable (Verilator tolerates the two-
  // process form).  The co-sim drives bd_en only during preload/dump and wr_en
  // only during execution, so the two are mutually exclusive and the merged
  // form is bit-identical; backdoor wins on the (unreachable) overlap.
  always_ff @(posedge clk) begin
    if (bd_en) begin
      if (bd_sel)    hbm[bd_addr] <= bd_wdata;
      else           sram[bd_addr[22:0]] <= bd_wdata;
    end else if (wr_en) begin
      if (wr_sel)    hbm[wr_addr] <= wr_data;
      else           sram[wr_addr[22:0]] <= wr_data;
    end
  end

  // Backdoor read (combinational).
  always_comb begin
    bd_rdata = 8'b0;
    if (bd_sel)      bd_rdata = hbm[bd_addr];
    else             bd_rdata = sram[bd_addr[22:0]];
  end
"""


def patch_qmem(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()
    if t.count(QMEM_OLD) != 1:
        sys.exit(f"[gen_rtl] qmem: expected exactly one write-pair block, "
                 f"found {t.count(QMEM_OLD)}")
    t = t.replace(QMEM_OLD, QMEM_NEW, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(t)


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    for name in NEEDED:
        shutil.copyfile(os.path.join(SRC, name), os.path.join(DST, name))
    patch_command_processor(os.path.join(DST, "command_processor.sv"))
    patch_qmem(os.path.join(DST, "qmem.sv"))
    print(f"[gen_rtl] wrote {len(NEEDED)} files to {DST}")


if __name__ == "__main__":
    main()
