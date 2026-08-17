# kn128x16 — SMIC28 register-file macro generation record

- **Macro**: `kn128x16` (instance name = module name)
- **Compiler**: SM18CD001 `rf_sp_hse_svt_mvt`, single-port register file
- **Source package**: `/home/public/PDK/SMIC28/SRAM_Ccompiler_ARM20240823/NO3__SMIC 28HKCP Single Port Reg File Compiler (SM18CD001)/arm/smic/28hkcp/rf_sp_hse_svt_mvt/r0p1`
- **Compiler version**: r0p1 (release r0p1-03eac0), GUI 8.5.10
- **Configuration**: words=128 bits=16 mux=2 mvt=BASE write_mask=off pipeline=off bmux=off redundancy=off ser=none power_type=otc
- **Corners generated (NLDM)**: `tt_ctypical_0p90v_0p90v_25c` (0.90 V), `ssg_cworstt_0p81v_0p81v_125c` (0.81 V)

> 128×16 (words=128 bits=16 mux=2) was accepted directly by the compiler; no
> fallback to 128×32 was needed.

## Environment (no-space compatibility tree)

```bash
skill_dir=/home/lzl/.agents/skills/smic28-sram-compiler
shim_root=/tmp/smic28-compiler.23iJMm
rfsp_entry=$( $skill_dir/scripts/prepare_compiler.sh rfsp "$shim_root/rfsp" )
# => /tmp/smic28-compiler.23iJMm/rfsp/bin/rf_sp_hse_svt_mvt
```

## Generation commands

Run from `/tmp/sram_macros_build/kn128x16`.

```bash
R=/tmp/smic28-compiler.23iJMm/rfsp/bin/rf_sp_hse_svt_mvt
CFG="-instname kn128x16 -words 128 -bits 16 -mux 2 -mvt BASE \
     -write_mask off -pipeline off -bmux off -redundancy off -ser none"

zsh "$R" verilog $CFG
zsh "$R" liberty $CFG -libertyviewstyle nldm -libname kn128x16 \
     -corners tt_ctypical_0p90v_0p90v_25c,ssg_cworstt_0p81v_0p81v_125c
for view in lef-fp gds2 lvs; do
  zsh "$R" "$view" $CFG -keeplogs
done
```

## Outputs

| View | File |
| --- | --- |
| Verilog sim model | `kn128x16.v` |
| NLDM Liberty (tt 0.90 V 25 C) | `kn128x16_tt_ctypical_0p90v_0p90v_25c.lib` |
| NLDM Liberty (ssg 0.81 V 125 C) | `kn128x16_ssg_cworstt_0p81v_0p81v_125c.lib` |
| LEF footprint | `kn128x16.lef` |
| GDS2 layout | `kn128x16.gds2` |
| LVS netlist | `kn128x16.cdl` |
| Antenna CLF | `kn128x16_antenna.clf` |

## Notes

- SM18CD001 default `name_case` is **lower**: verilog/liberty/CDL use lowercase
  pins (`q, clk, cen, wen, a, d, ema, emaw, emas, ret1n`), unlike the
  upper-case SM18CA001 macros.
- EMA pins always present: `ema[2:0]`, `emaw[1:0]`, `emas`; recommended binding
  `ema=011` (0.9/1.0/1.1 V), `emaw=01`, `emas=0`.
- Verilog uses `ifdef POWER_PINS` (power ports `VDDCE, VDDPE, VSSE` present only
  when defined); default module is `kn128x16 (q, clk, cen, wen, a, d, ema, emaw, emas, ret1n)`.
- Top `.subckt kn128x16 VDDCE VDDPE VSSE q[15] ... d[0] ema[2] ... ret1n`.
