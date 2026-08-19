# 从零复现指南

> 本文给出 QCore 全链路（golden → qsim → qrun → RTL co-sim → ASIC）的逐段复现命令与预期输出。
> 环境：Python 3.10、numpy、ml_dtypes、torch、transformers；Verilator 4.038（RTL/FPGA）；
> Synopsys DC + 本地 SMIC28 28HKCP（当前 ASIC），Yosys/OpenSTA + sky130（legacy）。

## 0. 前置：模型与依赖

```bash
# 模型：config.json + model.safetensors（默认从 HF 本地缓存解析）
#   目标路径 ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/
huggingface-cli download Qwen/Qwen3-0.6B
```

各命令默认解析的模型路径为 `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`（含
`config.json` + `model.safetensors`），或显式 `--model-dir DIR`。

### 一键验收（run_all_acceptance.sh）

全链路逐条覆盖 README「组件索引表」（qsim / M5 / compiler / runtime / golden3 / RTL co-sim /
FPGA / ASIC），一键执行并出汇总表（组件 / 命令 / 结果 / 耗时(s)；任一 FAIL → exit 1）：

```bash
bash run_all_acceptance.sh            # 默认全量（含 golden3 ~8 min + co-sim ~30 min；runtime 仅 m4 证据校验）
bash run_all_acceptance.sh --quick    # 快速（跳过 golden3 / co-sim / m4 全量）
bash run_all_acceptance.sh --full     # 默认全量 + python3 qrun/m4.py 显式全量重跑（小时级）
```

> **--full 用法（L2）**：`--full` 显式全量重跑 `python3 qrun/m4.py`（默认 qbin 逻辑已同步：
> BF16 三条用 `--qbin`（默认 `/tmp/qwen3-0.6b-bf16.qbin`），INT8 用 `--qbin-int8`（默认
> `/tmp/qwen3-0.6b.qbin`））。故 `--full` 前须先 `qforge compile --dtype bf16` 产出 BF16 qbin
> （见 §3.1），否则 BF16 容器缺失时装载器报错（非静默）。

结果取值 `PASS` / `FAIL` / `SKIP(quick)` / `SKIP(工具缺失)`；ASIC 段（verilator lint / yosys synth /
OpenSTA）按工具链在位性自动判定，缺失不伪造 PASS。

## 1. Golden（P1 参考 + M7 三向 golden）

### 1.0 golden 再生成（完整验收子集）

`golden/` 可由 `ref/gen_golden.py` 再生成（gitignored，≈5.2 GB）。生成器已参数化：`MODEL_DIR`
（默认 `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`）与 `DEVICE`（默认 `cuda`）环境变量
可覆盖，`--model-dir` / `--device` 命令行参数同效。完整验收子集 = `prefill_seq128` +
`decode_seq1_cache{0,512,1024,2048,4096,8192}` + `linear_wq_pf` / `linear_wq_dc`：

```bash
# 前置：本地 HF 缓存含 Qwen3-0.6B（§0：huggingface-cli download Qwen/Qwen3-0.6B）
python3 ref/gen_golden.py            # 默认全量（--full 显式指定亦等价）
MODEL_DIR=/path/to/Qwen3-0.6B DEVICE=cpu python3 ref/gen_golden.py --full
```

`golden/` 已内置 Qwen3-0.6B 中间 tensor/KV/logits dump（`decode_seq1_cache*`、`prefill_seq128`、
`linear_wq_*`）。三向 golden（15 op 实例 + 逐层 hidden）由 RTL 测试台驱动：

```bash
cd rtl/tb && python3 run_golden3.py
```

预期输出（摘要，全文见 `docs/p8/golden3-report.md`）：

```text
15/15 实例三向全过（M4 口径 |y|≥1e-3 ≤1 ULP，tiny 例外记录）
逐层 hidden 三向一致（L00 / L13 / L27）—— 0 ULP
```

结果 JSON：`docs/p8/golden3-results.json`。

## 2. qsim（ISA 模拟器：功能级 + 时序级）

### 2.1 ISA 字段 / 汇编 round-trip

```bash
python3 qsim/test_isa_fields.py
```

预期输出（实际运行）：

```text
PASS test_33_instructions_frozen
PASS test_assembler_disassembler_roundtrip
PASS test_dma_field_layout
PASS test_dtype_acc_codes
PASS test_engine_tag_mismatch_flagged
PASS test_engine_tag_ranges
PASS test_header_bit_layout
PASS test_kv_field_layout
PASS test_matrix_field_layout
PASS test_reserved_opcode_rejected
PASS test_roundtrip_all_instructions

11/11 field-level assertion groups passed
```

### 2.2 时序模型（M5 验收）

```bash
python3 qsim/timing_p6.py
```

预期输出（实际运行，尾段）：

```text
=== M5 acceptance (write-roofline 80%) ===
  ctx 4096: target   2560000 cyc | scheduled   2078544 cyc | PASS (margin 18.81%, 481.1 tok/s)
  ctx 8192: target   4860000 cyc | scheduled   3919152 cyc | PASS (margin 19.36%, 255.2 tok/s)

=== prefill 128-block per layer: matrix 63488 + vector 55377 = 118865 cyc (compute-bound; QUANT hidden 8192)
```

v0 时序模型（`python3 qsim/timing.py`）另有逐层五桶分解，见 `docs/p3/sim-report.md`。

## 3. qrun（运行时）：qbin 重建 + 逐 token 生成

### 3.1 /tmp qbin 重建

```bash
python3 -m qforge compile Qwen/Qwen3-0.6B --dtype int8 -o /tmp/qwen3-0.6b.qbin
```

产出 ≈579 MiB（607,289,992 B）qbin：magic `NLPU` / version `1` / flags `0x6`（INT8 + dual-mode）、
141 张量、权重 595,984,384 B、scale 9,312,256 B、PF/DC 程序各 29,522 条 128-bit 指令、
ENDQ 哨兵 + 长度校验（`docs/p4/qforge.md` §2）。

BF16 容器（权重自包含，qrun BF16 路径数据源）：

```bash
python3 -m qforge compile Qwen/Qwen3-0.6B --dtype bf16 -o /tmp/qwen3-0.6b-bf16.qbin
```

产出 ≈1.19 GiB qbin：flags `0x4`（BF16 + dual-mode）、141 张量 `dtype=BF16`（scale 段省略）、
权重 1,191,968,768 B、PF/DC 程序为空（qrun 装载期重新生成 BF16 程序，`docs/p4/qforge.md` §2.1）。
BF16 qbin round-trip：`read_qbin` 后 141 张量逐字节与文件体一致（位级一致）。

M3 全量校验脚本（默认也把 qbin 构建到 `/tmp/qwen3-0.6b-m3.qbin`）：

```bash
python3 qforge/verify_m3.py            # (a) 全模型加载+round-trip；(b) 6 类 × PF/DC 量化误差
```

预期输出（摘要）：`(a) round_trip_ok=true`；`(b) 6 classes x PF/DC: ALL PASS`；结果写
`docs/p4/m3-results.json`（全文见 `docs/p4/qforge.md` §3、`docs/p4/quant-error-report.md`）。

### 3.2 CLI 运行（qsim 后端）

```bash
python3 -m qrun /tmp/qwen3-0.6b.qbin --prompt "Explain attention" --max-new 20
```

预期输出（摘要）：

```text
[qrun] dtype=int8 prompt_tokens=... build=...s generate=...s
[qrun] new tokens: [ ... ]
[qrun] decoded: ...
```

### 3.3 M4 验收驱动（四条，需 torch + HF 参照）

```bash
python3 qrun/m4.py [--model-dir DIR] [--qbin BF16_PATH] [--qbin-int8 INT8_PATH] \
                   [--weights-from-hf] [--only 1,2,3,4]
```

- `--qbin` 默认 `/tmp/qwen3-0.6b-bf16.qbin`（BF16 容器，case 1-3）；`--qbin-int8` 默认
  `/tmp/qwen3-0.6b.qbin`（case 4）。BF16 权重默认从容器装载；`--weights-from-hf` 显式回退
  safetensors。装载器校验容器 flags dtype==BF16，INT8 容器拒绝 bf16 请求（报错不静默）。
- 预期输出（摘要，全文见 `docs/p5/m4-report.md` §2）：BF16 短 ctx 20/20、中 ctx 8/8、
  4K 5/5、8K argmax 一致；INT8 交叉一致 2/10（记录为需评审项，不粉饰）。

## 4. RTL co-sim

```bash
# lint 全部模块
cd rtl && for f in qcore_pkg softfloat kv_addrgen matrix_engine vector_engine dma_engine sram qmem command_processor qcore_top; do
  verilator --lint-only -Wno-fatal -Wno-WIDTH --top-module $f $f.sv || echo "FAIL $f"; done

# 编译 co-sim 可执行（Verilator 4.038）
cd rtl/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
  --top-module qcore_top -I.. ../qcore_top.sv sim_main.cpp --Mdir obj_dir

# 运行验收驱动（全尺寸 M2a linear + vector + KV）
python3 rtl/tb/run_cosim.py
```

预期输出（摘要，全文见 `docs/p7/rtl-report.md` §7.3）：

```text
[PASS] PF BF16 trace=True max_ulp=3.0 ulp_normal=1.0 cycles=15741
[PASS] DC BF16 trace=True max_ulp=1.0 ulp_normal=0.0 cycles=1593
[PASS] PF INT8 trace=True max_ulp=0.0 ulp_normal=0.0 cycles=15475
[PASS] DC INT8 trace=True max_ulp=0.0 ulp_normal=0.0 cycles=1835
```

单层 golden（L00，14/14）：`python3 rtl/tb/run_layer_golden.py`。

## 5. FPGA（板卡无关接口层 + 集成 smoke）

```bash
# 集成 smoke（编译 + 运行，M6 判据）
cd fpga/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
  --top-module qcore_fpga_top -I../../rtl -I.. ../../fpga/qcore_fpga_top.sv \
  sim_fpga_main.cpp --Mdir obj_dir
python3 run_fpga_smoke.py
```

预期输出（全文见 `docs/p9/porting.md` §3）：

```text
[PASS] VADD             trace=True max_ulp=0.0 cycles=5 (expected 5)
[PASS] RMSNORM          trace=True max_ulp=0.0 cycles=20 (expected 20)
[PASS] KV_APPEND_LOAD   trace=True max_ulp=0.0 cycles=211 (expected 211)
[PASS] GEMM_BF16        trace=True max_ulp=0.0 cycles=266 (expected 266)
P9 FPGA smoke: ALL PASS
```

## 6. ASIC（SMIC28 当前基线 + sky130 legacy）

当前 SMIC28 流程先从本地商业 PDK 生成/映射标准单元与 SRAM `.db`；这些商业视图不进入
仓库。矩阵状态壳的 Verilator 测试使用与真宏同端口、同 1-cycle read 语义的行为模型：

```bash
# 九个 kh4096x64 的 bank/address、单口冲突、首元素预取与四种数值模式
bash asic/run_matrix_sram_check.sh

# 物理阵列 dual-MAC PE：完整 INT8 乘法域、bubble、重载与模 2^32 累加
bash asic/run_matrix_int8_pe_check.sh

# 16x16 结构 tile：二维 skew scoreboard（RTL）
bash asic/run_matrix_int8_pe_tile_check.sh

# 准备本地 SMIC28 标准单元和 SRAM 库（详见 asic/sram_macros/*/GEN.md）
bash asic/smc28/setup_smic28.sh

# 当前全顶层双角 compile_ultra；矩阵状态/控制壳 + 9 个真 SRAM，算术核保持宏边界
DC_TECH=smic28 bash asic/dc/run_dc.sh tt_025C_1v80 synth_top
DC_TECH=smic28 bash asic/dc/run_dc.sh ss_100C_1v60 synth_top

# 代表数据通路与 BF16 MAC 的 1 ns ideal-clock 探针
DC_TECH=smic28 bash asic/dc/run_dc.sh tt_025C_1v80 synth_datapath
DC_TECH=smic28 bash asic/dc/run_dc.sh ss_100C_1v60 mac_bf16

# dual-MAC PE 的 registered-boundary TT/SS 探针；标签与已提交报告一致
DC_TECH=smic28 DC_LABEL=pe1c bash asic/dc/run_dc.sh tt_025C_1v80 matrix_int8_pe
DC_TECH=smic28 DC_LABEL=pe1c bash asic/dc/run_dc.sh ss_100C_1v60 matrix_int8_pe

# 0.9 ns 裕量映射（可选）及 1.0 ns TT 映射网表功能交叉检查
DC_TECH=smic28 DC_PERIOD=0.9 DC_LABEL=pe1c_p090 \
  bash asic/dc/run_dc.sh tt_025C_1v80 matrix_int8_pe
DC_TECH=smic28 DC_PERIOD=0.9 DC_LABEL=pe1c_p090 \
  bash asic/dc/run_dc.sh ss_100C_1v60 matrix_int8_pe
bash asic/run_matrix_int8_pe_gate_check.sh

# 16x16 tile 的 SMIC28 TT/SS 分层综合（1 ns probe；报告约 11.8/23.0 min CPU）
DC_TECH=smic28 DC_LABEL=tile1c \
  bash asic/dc/run_dc.sh tt_025C_1v80 matrix_int8_pe_tile
DC_TECH=smic28 DC_LABEL=tile1c \
  bash asic/dc/run_dc.sh ss_100C_1v60 matrix_int8_pe_tile

# TT tile 映射网表 smoke（官方 HDC30P140 model，Icarus）
bash asic/run_matrix_int8_pe_tile_gate_check.sh
```

以下是保留的 sky130 legacy 开源复现路径。sky130 liberty（`sky130_fd_sc_hd__*.lib`，
5 corner）来自 skywater-pdk 官方 `*.lib.json` 经其 `python-skywater-pdk` 转换器生成：

`open_pdks` 集成流是其一键复现路径：

```bash
# open_pdks 集成流：产出 5-corner liberty 至 $PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/
git clone https://github.com/RTimothyEdwards/open_pdks
cd open_pdks && ./configure --enable-sky130-pdk && make
# 也可直接用 skywater-pdk-libs-sky130_fd_sc_hd 的 cells/*/*.lib.json + python-skywater-pdk 转换器
```

Synopsys DC 侧需先 `clean_lib.py` 语义中性清理再经 Library Compiler 转 `.db`
（详见 `docs/p10/asic-report.md` §10.2；`asic/dc/db/` 可再生成）：

```bash
bash asic/dc/build_db.sh            # clean_lib.py → lc_shell read_lib/write_lib（CORNERS/LIB_DIR 可配置）
```

```bash
# 快照 → 可综合派生（语义中性 desugar）
python3 asic/preprocess.py

# FP 基元综合（8 项 FP 基元 + 两种 MAC）
bash asic/run_synth.sh tt_025C_1v80
bash asic/run_mac_synth.sh tt_025C_1v80

# STA（多 corner，需 OpenSTA + sky130 liberty）
STA=/path/to/sta
for c in tt_025C_1v80 ss_n40C_1v28 ss_100C_1v60 ff_100C_1v65; do
  LIB=.../sky130_fd_sc_hd__$c.lib $STA -no_splash -exit asic/sta.tcl
done
```

预期输出（摘要，全文见 `docs/p10/asic-report.md` §0–§3/§9–§10）：

```text
SMIC28：矩阵状态壳定向测试 4/4 PASS；DC 识别 9 个矩阵 kh4096x64，并报告专属输入/读出路径
SMIC28：代表 synth_datapath / mac_bf16 的 tt/ss 1 ns ideal-clock 探针均 MET
SMIC28：dual-MAC PE 核心/registered probe/TT 映射网表均 PASS；1.0 ns 与 0.9 ns 双角探针 MET
SMIC28：16x16 tile RTL **2720 checks**、TT gate **1168 checks** 均 PASS；TT/SS tile 1 ns probe MET
legacy sky130：P10b 数据通路 tt Fmax ≈ 129 MHz；历史 ss/ff corner 见报告 §3
结论：以上均为 pre-layout 综合/STA 口径；64-tile 完整阵列、整芯片 CTS/寄生 signoff 尚未闭合
```
