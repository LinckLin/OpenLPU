#!/usr/bin/env python3
"""hoist_dc.py — prepare rtl/ref/asicsnap sources for DC Presto (full design).

Two DC Presto incompatibilities are fixed, both semantically neutral:

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

DECL_RE = re.compile(r'^  (logic|integer|wire|reg)\b')
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


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    nchanged = 0
    for f in sorted(SRC.glob('*.sv')):
        text = f.read_text()
        if re.search(r'\bendmodule\b', text):
            out_text, changed = process_module(text)
        else:
            # Package files (qcore_pkg/softfloat/rope_lut): loop fixes only.
            original = text
            out_text = apply_varloop(text)
            out_text = LOOP_BREAK_RE.sub('break;', out_text)
            changed = out_text != original
        (DST / f.name).write_text(out_text)
        if changed:
            nchanged += 1
            print(f'desugared: {f.name}')
    # synth_top.sv / sram_macro.sv are DC-clean already; copy verbatim.
    for name in ('synth_top.sv', 'sram_macro.sv'):
        (DST / name).write_text((ASIC / name).read_text())
    print(f'files desugared: {nchanged}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
