#!/usr/bin/env python3
"""clean_lib.py — make sky130 liberty readable by Synopsys Library Compiler 2018.

Context (P10 §10 DC flow): the skywater-pdk python converter emits a liberty that
OpenSTA/Yosys accept verbatim, but Synopsys LC 2018.06-SP1 rejects with *fatal*
LBDB-27 errors because of two constructs around the bulk-well (`VNB` nwell / `VPB`
pwell) pins:

  1. 6 latch cells (dlclkp_{1,2,4}, sdlclkp_{1,2,4}) carry an internal node pin
     `M0` whose `related_ground_pin` points at `VNB` (a *nwell* pg_pin, not a
     primary_ground).  LC rejects `related_ground_pin` unless it references a
     primary ground pin.  Fix: repoint to `VGND`.  `M0` is an internal node with
     no timing arc / power table, so this is semantically neutral.

  2. 6 level-shifter tap cells (lpflow_lsbuf_lh_{hl_,}isowell_tap_{1,2,4}) have a
     `related_bias_pin : "VNB"` inside their `pg_pin("VPWR")` that dangles — those
     cells define no `VNB` pg_pin at all.  LC reports "invalid attribute 'VNB'".
     Fix: drop the dangling line.  Bulk-well metadata only, no impact.

Both fixes touch only bulk-well / internal-node metadata.  No timing arc, power
table, cell function, or leakage value is modified, so the cleaned liberty is
bit-identical to the original for every quantity the DC flow reports
(timing / leakage / area).  OpenSTA keeps reading the *original* unmodified
liberty (its baseline is unchanged).

Usage:  python3 asic/dc/clean_lib.py <in.lib> <out.lib>
"""
import re
import sys

# Cells whose internal-node pin M0 references VNB as its ground pin.
_GROUND_FIX_CELLS = {
    "sky130_fd_sc_hd__dlclkp_1",
    "sky130_fd_sc_hd__dlclkp_2",
    "sky130_fd_sc_hd__dlclkp_4",
    "sky130_fd_sc_hd__sdlclkp_1",
    "sky130_fd_sc_hd__sdlclkp_2",
    "sky130_fd_sc_hd__sdlclkp_4",
}

# Cells whose VPWR pg_pin carries a dangling `related_bias_pin : "VNB"`.
_BIAS_DROP_CELLS = {
    "sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_1",
    "sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_2",
    "sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_4",
    "sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_1",
    "sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_2",
    "sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_4",
}

CELL_RE = re.compile(r'cell\s*\("([^"]+)"\)')


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: clean_lib.py <in.lib> <out.lib>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cur_cell = None
    fixed_ground = 0
    dropped_bias = 0
    out = []
    for line in lines:
        m = CELL_RE.search(line)
        if m:
            cur_cell = m.group(1)
        if cur_cell in _GROUND_FIX_CELLS and 'related_ground_pin : "VNB"' in line:
            line = line.replace(
                'related_ground_pin : "VNB"', 'related_ground_pin : "VGND"'
            )
            fixed_ground += 1
        if cur_cell in _BIAS_DROP_CELLS and 'related_bias_pin : "VNB"' in line:
            dropped_bias += 1
            continue
        out.append(line)

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(
        f"cleaned {src} -> {dst}: repointed {fixed_ground} related_ground_pin, "
        f"dropped {dropped_bias} dangling related_bias_pin"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
