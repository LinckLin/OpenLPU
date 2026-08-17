# kh4096x64 — SMIC28 SRAM macro generation record

- **Macro**: `kh4096x64` (instance name = module name)
- **Compiler**: SM18CA001 `sram_sp_hsd_svt_mvt`, single-port SRAM
- **Source package**: `/home/public/PDK/SMIC28/SRAM_Ccompiler_ARM20240823/NO1__SMIC 28HKCP Single Port SRAM Compiler (SM18CA001)/arm/smic/28hkcp/sram_sp_hsd_svt_mvt/r0p1`
- **Compiler version**: r0p1 (release r0p1-02eac0), GUI 8.5.10
- **Configuration**: words=4096 bits=64 mux=8 mvt=BASE write_mask=off pipeline=off bmux=off redundancy=off ser=none power_type=otc
- **Corners generated (NLDM)**: `tt_ctypical_0p90v_0p90v_25c` (0.90 V), `ssg_cworstt_0p81v_0p81v_125c` (0.81 V)

## Environment (no-space compatibility tree)

The compiler package path contains spaces and the bundled `bifrun` needs the
CentOS 7 glibc 2.17 loader on Ubuntu 22. Build the shim tree with the skill
helper, then run entries with `zsh` (entries use `#!/bin/ksh -p`):

```bash
skill_dir=/home/lzl/.agents/skills/smic28-sram-compiler
shim_root=/tmp/smic28-compiler.23iJMm   # one-time, keep during generation
sram_entry=$( $skill_dir/scripts/prepare_compiler.sh sram "$shim_root/sram" )
# => /tmp/smic28-compiler.23iJMm/sram/bin/sram_sp_hsd_svt_mvt
```

`zsh "$sram_entry" -version` prints `GUI version 8.5.10`.

## Generation commands

All commands run from the macro build directory
(`/tmp/sram_macros_build/kh4096x64`). Outputs land in the current directory.

```bash
S=/tmp/smic28-compiler.23iJMm/sram/bin/sram_sp_hsd_svt_mvt
CFG="-instname kh4096x64 -words 4096 -bits 64 -mux 8 -mvt BASE \
     -write_mask off -pipeline off -bmux off -redundancy off -ser none"

# 1. Verilog simulation model
zsh "$S" verilog $CFG

# 2. NLDM Liberty, two corners
zsh "$S" liberty $CFG -libertyviewstyle nldm -libname kh4096x64 \
     -corners tt_ctypical_0p90v_0p90v_25c,ssg_cworstt_0p81v_0p81v_125c

# 3. Physical views — generate sequentially (shared temp names per instance)
for view in lef-fp gds2 lvs; do
  zsh "$S" "$view" $CFG -keeplogs
done
```

## Outputs

| View | File |
| --- | --- |
| Verilog sim model | `kh4096x64.v` |
| NLDM Liberty (tt 0.90 V 25 C) | `kh4096x64_tt_ctypical_0p90v_0p90v_25c.lib` |
| NLDM Liberty (ssg 0.81 V 125 C) | `kh4096x64_ssg_cworstt_0p81v_0p81v_125c.lib` |
| LEF footprint | `kh4096x64.lef` |
| GDS2 layout | `kh4096x64.gds2` |
| LVS netlist | `kh4096x64.cdl` |
| Antenna CLF | `kh4096x64_antenna.clf` |

## Notes

- EMA pins always present: `EMA[2:0]`, `EMAW[1:0]`, `EMAS`; recommended binding
  `EMA=011` (0.9/1.0/1.1 V), `EMAW=01`, `EMAS=0`.
- Verilog uses `ifdef POWER_PINS` to switch between a power-port module
  (`VDDCE, VDDPE, VSSE, Q, CLK, CEN, WEN, A, D, EMA, EMAW, EMAS, RET1N`) and the
  default behavioral module without power ports
  (`Q, CLK, CEN, WEN, A, D, EMA, EMAW, EMAS, RET1N`).
- `lvs` prints two benign `bif2cdl - WARNING: bbif SUBCKT rm1w/rmxw redefined`
  messages; output `.cdl` is complete (top `.SUBCKT kh4096x64 VDDCE VDDPE VSSE Q[63] ...`).
