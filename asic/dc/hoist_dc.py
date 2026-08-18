#!/usr/bin/env python3
"""hoist_dc.py — prepare rtl/ref/asicsnap sources for DC Presto (full design).

Three DC Presto / resource incompatibilities are fixed, all semantically neutral:

  1. Loop-break idiom.  The frozen softfloat.sv uses the Yosys/Verilator-friendly
     leading-one-detect loop
         for (integer k = 31; k >= 0; k = k - 1) if (mag[k]) begin lead = k; k = -1; end
     (counter assigned -1 to force exit).  Presto cannot elaborate this
     (ELAB-900 "Loop exceeded maximum iteration limit").  Fix: `k = -1;` ->
     `break;` (the counter is never read after the assignment).

  2. Declaration-after-use.  The frozen modules use legal SV "declaration
     anywhere" (e.g. command_processor.sv declares `kv_base` *after* the
     instantiation that connects it).  Presto requires module-level declarations
     to precede first use (VER-954/VER-956).  Fix: hoist module-body signal
     declarations (`logic`/`integer`/`wire`/`reg`) to the top of each module body
     (right after the `import` lines).

  3. Full-design storage/core physicalization.  matrix_engine's inferred state
     arrays are replaced by asic/matrix_engine_sram.sv, which instantiates nine
     SMIC28 SRAM macros behind a live state/control shell.  Its 128x128
     arithmetic array and vector_engine's runtime lane core remain physical-core
     black boxes; otherwise DC would unroll the softfloat datapaths.  The CP
     instruction array becomes a bb_sram black box (1 sync write + 1
     combinational read).

The frozen modules use brace-less bodies (Verilog-2001 style) and 2-space-indent
module-level declarations, so the hoist keys on the exact-2-space indent.
Package files (qcore_pkg/softfloat/rope_lut) get only the loop-break fix.

Copies are written under asic/dc/gen_full/ so rtl/ and asic/gen/ stay untouched.

Usage:  python3 asic/dc/hoist_dc.py
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SRC = REPO / "rtl" / "ref" / "asicsnap"
ASIC = REPO / "asic"
DST = ASIC / "dc" / "gen_full"

DECL_RE = re.compile(r'^  (logic|integer|wire|reg|localparam)\b')
PORTS_END_RE = re.compile(r'^\s*\);\s*$')
IMPORT_RE = re.compile(r'^\s*import\b')
LOOP_BREAK_RE = re.compile(r'\bk\s*=\s*-1\s*;')

# Variable-bound sticky-bit loops (softfloat.sv) -> fixed bound + runtime guard,
# the same semantically-neutral rewrite preprocess.py applies for Yosys.
VARLOOP_REPLACEMENTS = [
    (
        "for (integer k = 0; k < lead - 23 - 1; k = k + 1) st = st | mag[k];",
        "for (integer k = 0; k < 8; k = k + 1) st = st | (mag[k] & (k < (lead - 23 - 1)));",
    ),
    (
        "for (integer k = 0; k < 23 - shf - 1; k = k + 1) sty = sty | v[k];",
        "for (integer k = 0; k < 23; k = k + 1) sty = sty | (v[k] & (k < (23 - shf - 1)));",
    ),
]


def apply_varloop(text: str) -> str:
    for old, new in VARLOOP_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def process_module(text):
    """Apply loop-break + variable-loop fixes + declaration hoist to a module."""
    original = text
    text = apply_varloop(text)
    text = LOOP_BREAK_RE.sub('break;', text)
    lines = text.split('\n')
    out = []
    i = 0
    n = len(lines)
    changed = text != original
    while i < n:
        line = lines[i]
        if not re.match(r'^module\s+\w+', line):
            out.append(line)
            i += 1
            continue
        # Module found: emit header up to and including the lone `);`.
        header = []
        while i < n:
            header.append(lines[i])
            i += 1
            if PORTS_END_RE.match(lines[i - 1]):
                break
        # Collect the body (until `endmodule`), hoisting declarations.
        body_decls = []
        body_rest = []
        j = i
        while j < n and not re.match(r'^\s*endmodule\b', lines[j]):
            ln = lines[j]
            if DECL_RE.match(ln):
                buf = [ln]
                while ';' not in ln and j + 1 < n:
                    j += 1
                    ln = lines[j]
                    buf.append(ln)
                body_decls.extend(buf)
                j += 1
                changed = True
                continue
            body_rest.append(ln)
            j += 1
        insert_at = 0
        for k, ln in enumerate(body_rest):
            if IMPORT_RE.match(ln):
                insert_at = k + 1
        body = body_rest[:insert_at] + body_decls + body_rest[insert_at:]
        out.extend(header)
        out.extend(body)
        i = j  # continue at endmodule (the loop re-emits it)
    return '\n'.join(out) + '\n', changed


def make_blackbox(text):
    """Strip a numeric-core module to a DC black box (keep module/params/ports).

    vector_engine is the co-sim functional model of the 128-lane datapath.  Its
    runtime `for (i < len)` loops do not express the physical lane pipeline and
    would make DC unroll the full softfloat datapath.  The matrix engine is no
    longer handled here: main() substitutes the macro-backed physical source.
    """
    lines = text.split('\n')
    n = len(lines)
    start = next(i for i, ln in enumerate(lines) if re.match(r'^module\s+\w+', ln))
    name = re.match(r'^module\s+(\w+)', lines[start]).group(1)
    header = []
    i = start
    while i < n:
        header.append(lines[i])
        i += 1
        if PORTS_END_RE.match(lines[i - 1]):
            break
    # Known co-sim port-width mismatches: the CP drives `len` as 16 bits while
    # the vector_engine port is declared [31:0]; Verilator widens silently, DC's
    # linker (LINK-3) does not.  Narrow the black-box port to match the CP.
    if name == 'vector_engine':
        header = [ln.replace('input  logic [31:0] len,', 'input  logic [15:0] len,')
                  for ln in header]
    guard = name.upper() + '_SV'
    bb = [
        f'`ifndef {guard}',
        f'`define {guard}',
        '',
        '// DC black box (hoist_dc.py: numeric-core macro; primitives in §10.4).',
        '(* blackbox *)',
    ]
    bb.extend(header)
    bb.append('endmodule')
    bb.append('')
    bb.append(f'`endif // {guard}')
    return '\n'.join(bb) + '\n'


def make_blackbox_sram(text):
    """Convert the behavioral SMIC28 macro models to DC black boxes.

    sram_macros.sv (kh4096x64 / ang4096x64 / kn128x16) holds Verilator-friendly
    inferred-RAM bodies (`logic [63:0] mem [0:4095]` + `always_ff`).  DC would
    infer those as 4096x64 flip-flop banks — the exact storage wall this task
    removes.  Keep each module header + port list (the macro pin interface),
    tag it `(* blackbox *)`, and strip the body, so DC links the cells against
    the compiled SMIC28 .lib (asic/dc/db/<macro>_<corner>.db) instead.
    set_dont_touch is applied in dc_top.tcl.
    """
    lines = text.split('\n')
    n = len(lines)
    out = []
    i = 0
    while i < n:
        if not re.match(r'^module\s+\w+', lines[i]):
            out.append(lines[i])
            i += 1
            continue
        out.append('(* blackbox *)')
        # module header through the lone ');'
        while i < n:
            out.append(lines[i])
            i += 1
            if PORTS_END_RE.match(lines[i - 1]):
                break
        # skip the inferred-RAM body up to endmodule, then close the stub
        while i < n and not re.match(r'^\s*endmodule\b', lines[i]):
            i += 1
        out.append('endmodule')
        i += 1
    return '\n'.join(out) + '\n'

def blackbox_imem(text):
    """Remap the command-processor instruction array to a bb_sram black box.

    `logic [127:0] imem [0:NINST-1]` would infer as a 4096x128 flip-flop bank;
    a real ASIC keeps the instruction stream in an SRAM, so the desugared copy
    instantiates bb_sram (1 sync write + 1 combinational read) instead.  The
    co-sim array in rtl/ is untouched.
    """
    # 1) declaration -> read wire (kept at its hoisted position, before first use)
    text = text.replace(
        '  logic [127:0] imem [0:NINST-1];',
        '  wire  [127:0] imem_rd;   // instruction word (black-box SRAM read)',
    )
    # 2) drop the backdoor write always_ff (the macro samples we/waddr/wdata itself)
    text = text.replace(
        "\n  // instruction memory load (testbench backdoor)\n"
        "  always_ff @(posedge clk) begin\n"
        "    if (imem_we) imem[imem_waddr] <= imem_wdata;\n"
        "  end\n",
        "\n",
    )
    # 3) all combinational reads imem[pc] -> imem_rd
    text = text.replace('imem[pc]', 'imem_rd')
    # 4) instantiate the macro at the end of the module body (after every decl,
    #    so pc / imem_rd are declared before use)
    instance = (
        "\n"
        "  // instruction memory = black-box SRAM macro (physically SRAM, not flops).\n"
        "  bb_sram #(.AW(12), .DW(128)) u_imem (\n"
        "    .clk(clk), .we(imem_we), .waddr(imem_waddr), .wdata(imem_wdata),\n"
        "    .raddr(pc[11:0]), .rdata(imem_rd)\n"
        "  );\n"
    )
    idx = text.rfind('endmodule')
    return text[:idx] + instance + text[idx:]


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    nchanged = 0
    for f in sorted(SRC.glob('*.sv')):
        name = f.name
        text = f.read_text()
        if name == 'matrix_engine.sv':
            # Physical matrix state/control shell + nine SMIC28 SRAMs.
            out_text = (ASIC / 'matrix_engine_sram.sv').read_text()
        elif name == 'vector_engine.sv':
            # Runtime-lane co-sim core -> physical vector-core black box.
            out_text = make_blackbox(text)
        elif name == 'sram_macros.sv':
            # SMIC28 SRAM macros -> black boxes (timing from compiled .lib).
            out_text = make_blackbox_sram(text)
        elif name == 'command_processor.sv':
            out_text, _ = process_module(text)   # loop-break + decl hoist
            out_text = blackbox_imem(out_text)   # instruction array -> bb_sram
        elif re.search(r'\bendmodule\b', text):
            out_text, _ = process_module(text)
        else:
            # Package files (qcore_pkg/softfloat/rope_lut): loop fixes only.
            out_text = apply_varloop(text)
            out_text = LOOP_BREAK_RE.sub('break;', out_text)
        (DST / name).write_text(out_text)
        if out_text != text:
            nchanged += 1
            print(f'desugared: {name}')
    # The state SRAM/control shell is live; only the 128x128 arithmetic array
    # stays a macro boundary (primitive timing is reported separately).
    (DST / 'matrix_compute_core.sv').write_text(
        (ASIC / 'matrix_compute_core_bb.sv').read_text()
    )
    # synth_top.sv / sram_macro.sv / bb_sram.sv are DC-clean already; copy verbatim.
    for name in ('synth_top.sv', 'sram_macro.sv', 'bb_sram.sv'):
        (DST / name).write_text((ASIC / name).read_text())
    print(f'files desugared: {nchanged}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
