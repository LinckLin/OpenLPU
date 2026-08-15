#!/usr/bin/env python3
"""Expand `assign { a, b, ... } = { x, y, ... };` (concat-LHS) into per-field
assigns, which OpenSTA 2019's Verilog reader cannot parse.  Also rewrites
escaped identifiers containing `$func$...`/`$proc$...` noise is left intact
(OpenSTA handles backslash-escaped ids)."""
import re
import sys

p = sys.argv[1]
t = open(p).read()


def fix(m):
    lhs = [x.strip() for x in m.group(1).split(",")]
    rhs = [x.strip() for x in m.group(2).split(",")]
    out = []
    if len(lhs) == len(rhs):
        for a, b in zip(lhs, rhs):
            out.append("assign %s = %s;" % (a, b))
    else:
        # unequal widths: assign bit-by-bit is unsafe; keep as-is (rare)
        out.append(m.group(0))
    return "\n".join(out)


t2 = re.sub(r"assign\s*\{\s*([^{}]+?)\s*\}\s*=\s*\{\s*([^{}]+?)\s*\}\s*;", fix, t)
open(p, "w").write(t2)
print("concat-LHS expanded; remaining:", t2.count("assign {"))
