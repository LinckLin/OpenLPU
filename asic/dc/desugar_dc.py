#!/usr/bin/env python3
"""desugar_dc.py — produce DC-compatible copies of the *pipeline* synthesis RTL.

DC 2018.06-SP1's Presto HDL compiler *infinite-loops* on the Yosys-friendly
loop-break idiom

      for (int k = 23; k >= 0; k--) if (x[k]) begin lead = k[4:0]; k = -1; end

(the counter is assigned -1 to force loop exit).  Yosys 0.44's 2005-Verilog
frontend has no `break`, so asic/synth_datapath.sv uses this idiom; Presto
accepts `break` but spins on the counter reassignment (reproduced with a minimal
6-line probe module — dc_shell never returns on `k = -1`, returns instantly on
`break`).

The fix is the single, semantically-identical substitution `k = -1;` -> `break;`
(the counter is never read after the assignment in these leading-one-detect
loops).  No other transform is applied to the pipeline files.

NOTE on the full design (synth_top): its engines come from rtl/ref/asicsnap/,
which DC reads *verbatim* — DC's Presto frontend handles `return`, `import`,
`inside`, casts, unpacked-array ports and variable-bound `for` natively, so the
Yosys desugar products (asic/gen/) are NOT used for the full design (they would
introduce Presto-incompatible "declaration after statement" inside functions).

Usage:  python3 asic/dc/desugar_dc.py
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ASIC = REPO / "asic"
DST = ASIC / "dc" / "gen"

PIPELINE = [
    "synth_datapath.sv",
    "synth_mac.sv",
    "matrix_int8_pe.sv",
    "matrix_int8_pe_probe.sv",
    "matrix_int8_pe_tile.sv",
    "matrix_int8_pe_tile_probe.sv",
    "matrix_int8_pe_array.sv",
    "matrix_int8_pe_array_probe.sv",
]

LOOP_BREAK = re.compile(r"\bk\s*=\s*-1\s*;")


def desugar(text: str) -> str:
    return LOOP_BREAK.sub("break;", text)


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    changed = 0
    for name in PIPELINE:
        text = (ASIC / name).read_text()
        out = desugar(text)
        (DST / name).write_text(out)
        if out != text:
            changed += 1
            print(f"desugared: {name} (k=-1 -> break)")
    print(f"files with loop-break fix: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
