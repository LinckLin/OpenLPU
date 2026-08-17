# ang4096x64 — SMIC28 SRAM macro generation record

- **Macro**: `ang4096x64` (instance name = module name)
- **Compiler**: SM18CA001 `sram_sp_hsd_svt_mvt`, single-port SRAM
- **Source package**: `/home/public/PDK/SMIC28/SRAM_Ccompiler_ARM20240823/NO1__SMIC 28HKCP Single Port SRAM Compiler (SM18CA001)/arm/smic/28hkcp/sram_sp_hsd_svt_mvt/r0p1`
- **Compiler version**: r0p1 (release r0p1-02eac0), GUI 8.5.10
- **Configuration**: words=4096 bits=64 mux=8 mvt=BASE write_mask=off pipeline=off bmux=off redundancy=off ser=none power_type=otc
  (identical configuration to `kh4096x64`; separate instance for stage-2 use)
- **Corners generated (NLDM)**: `tt_ctypical_0p90v_0p90v_25c` (0.90 V), `ssg_cworstt_0p81v_0p81v_125c` (0.81 V)

## Environment (no-space compatibility tree)

```bash
skill_dir=/home/lzl/.agents/skills/smic28-sram-compiler
shim_root=/tmp/smic28-compiler.23iJMm
sram_entry=$( $skill_dir/scripts/prepare_compiler.sh sram "$shim_root/sram" )
# => /tmp/smic28-compiler.23iJMm/sram/bin/sram_sp_hsd_svt_mvt
```

## Generation commands

Run from `/tmp/sram_macros_build/ang4096x64`.

```bash
S=/tmp/smic28-compiler.23iJMm/sram/bin/sram_sp_hsd_svt_mvt
CFG="-instname ang4096x64 -words 4096 -bits 64 -mux 8 -mvt BASE \
     -write_mask off -pipeline off -bmux off -redundancy off -ser none"

zsh "$S" verilog $CFG
zsh "$S" liberty $CFG -libertyviewstyle nldm -libname ang4096x64 \
     -corners tt_ctypical_0p90v_0p90v_25c,ssg_cworstt_0p81v_0p81v_125c
for view in lef-fp gds2 lvs; do
  zsh "$S" "$view" $CFG -keeplogs
done
```

## Outputs

| View | File |
| --- | --- |
| Verilog sim model | `ang4096x64.v` |
| NLDM Liberty (tt 0.90 V 25 C) | `ang4096x64_tt_ctypical_0p90v_0p90v_25c.lib` |
| NLDM Liberty (ssg 0.81 V 125 C) | `ang4096x64_ssg_cworstt_0p81v_0p81v_125c.lib` |
| LEF footprint | `ang4096x64.lef` |
| GDS2 layout | `ang4096x64.gds2` |
| LVS netlist | `ang4096x64.cdl` |
| Antenna CLF | `ang4096x64_antenna.clf` |

## Notes

- Same port set and EMA binding as `kh4096x64` (see `kh4096x64/GEN.md`).
- Top `.SUBCKT ang4096x64 VDDCE VDDPE VSSE Q[63] ...`.
