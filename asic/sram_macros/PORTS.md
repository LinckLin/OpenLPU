# SRAM macro port list (for RTL wiring)

Generated 2026-08-17 with SMIC28 ARM memory compilers (SM18CA001 / SM18CD001, r0p1).
Port names/directions are taken from the generated Verilog simulation models and
cross-checked against the NLDM Liberty pin lists and LVS CDL top-level SUBCKTs.

Power pins (`VDDCE`, `VDDPE`, `VSSE`) are `inout` and appear only when the
simulator defines `POWER_PINS`. For RTL instantiation use the default modules
**without** power ports listed below.

## EMA binding (all macros)

| Pin | Width | Binding |
| --- | --- | --- |
| EMA | 3 | `011` (0.9 V) |
| EMAW | 2 | `01` |
| EMAS | 1 | `0` |

## kh4096x64 — SM18CA001 single-port SRAM 4096×64 (mux=8)

Module (default): `kh4096x64 (Q, CLK, CEN, WEN, A, D, EMA, EMAW, EMAS, RET1N);`

| Pin | Dir | Width | Notes |
| --- | --- | --- | --- |
| Q | output | 64 | read data |
| CLK | input | 1 | clock |
| CEN | input | 1 | chip enable, active low |
| WEN | input | 1 | write enable, active low |
| A | input | 12 | address (4096 words) |
| D | input | 64 | write data |
| EMA | input | 3 | extra margin adjust |
| EMAW | input | 2 | EMA width |
| EMAS | input | 1 | EMA select |
| RET1N | input | 1 | retention enable, active low |
| VDDCE / VDDPE / VSSE | inout | 1 | power (POWER_PINS only) |

## ang4096x64 — SM18CA001 single-port SRAM 4096×64 (mux=8)

Identical port set and widths to `kh4096x64`:

`ang4096x64 (Q, CLK, CEN, WEN, A, D, EMA, EMAW, EMAS, RET1N);`

| Pin | Dir | Width |
| --- | --- | --- |
| Q | output | 64 |
| CLK | input | 1 |
| CEN | input | 1 |
| WEN | input | 1 |
| A | input | 12 |
| D | input | 64 |
| EMA | input | 3 |
| EMAW | input | 2 |
| EMAS | input | 1 |
| RET1N | input | 1 |
| VDDCE / VDDPE / VSSE | inout | 1 (POWER_PINS only) |

## kn128x16 — SM18CD001 single-port register file 128×16 (mux=2)

Module (default): `kn128x16 (q, clk, cen, wen, a, d, ema, emaw, emas, ret1n);`

Note: SM18CD001 default `name_case` is **lowercase**.

| Pin | Dir | Width | Notes |
| --- | --- | --- | --- |
| q | output | 16 | read data |
| clk | input | 1 | clock |
| cen | input | 1 | chip enable, active low |
| wen | input | 1 | write enable, active low |
| a | input | 7 | address (128 words) |
| d | input | 16 | write data |
| ema | input | 3 | extra margin adjust |
| emaw | input | 2 | EMA width |
| emas | input | 1 | EMA select |
| ret1n | input | 1 | retention enable, active low |
| VDDCE / VDDPE / VSSE | inout | 1 | power (POWER_PINS only) |

## RTL wiring summary (port name by macro)

| Signal | kh4096x64 | ang4096x64 | kn128x16 |
| --- | --- | --- | --- |
| read data | `Q` | `Q` | `q` |
| clock | `CLK` | `CLK` | `clk` |
| chip enable (active low) | `CEN` | `CEN` | `cen` |
| write enable (active low) | `WEN` | `WEN` | `wen` |
| address | `A` | `A` | `a` |
| write data | `D` | `D` | `d` |
| EMA (tie 011) | `EMA` | `EMA` | `ema` |
| EMAW (tie 01) | `EMAW` | `EMAW` | `emaw` |
| EMAS (tie 0) | `EMAS` | `EMAS` | `emas` |
| retention (tie 1) | `RET1N` | `RET1N` | `ret1n` |
