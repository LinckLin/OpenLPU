# QCore 项目交接文档（HANDOVER）

> 最近更新：2026-08-19（北京时间）。交接人：AI 工程师会话（主会话 + subagent 评审循环）。
> 本文件 = 后续任何人（人/模型）接手本仓库的**唯一入口**：先读本文件，再读 §11 指向的文档。

---

## 0. 速览（TL;DR）

| 项 | 值 |
|---|---|
| 项目 | **QCore**——LLM 推理个人级加速平台：Hugging Face Qwen 一键编译部署到自有编译器 / ISA / 模拟器 / Runtime / RTL / ASIC |
| 开源仓库 | `github.com/LinckLin/OpenLPU`（origin `git@github.com:LinckLin/OpenLPU.git`，分支 `main`） |
| 前序基线 HEAD | `7d48d9f`（Track 2.2a 已推送；本文件记录后续 Track 2.2b 增量） |
| 状态一句话 | 全栈功能闭环 + 开源 + 0.6B 全路径与 HF 逐位一致；ASIC 已切 SMIC28（28nm）基线：控制平面 tt/ss 为 **636.9/471.7 MHz**，代表数据通路与 BF16 MAC 的 tt/ss 均闭合 **1 GHz ideal-clock 探针**；matrix 状态已接入 9 个真 SRAM（0.707050 mm² 宏面积），算术核 Liberty 与整芯片 CTS/寄生 signoff 尚未完成 |
| 工作纪律 | **任何新计划必须先经 subagent 评审循环至「评审一致，可执行」再执行**；数字必须引用冻结规格原值；验收不过留在当前节点；新范围只登记 backlog |

---

## 1. 项目定位与总体架构

**目标模型（冻结）**：Qwen3-0.6B / 8B（`docs/spec-src/01-target-model.md` / `01b-target-model-0.6b.md`）。首代硬约束：**不做训练、不做多卡/分布式、不做 70B、不做 MoE、不做多模态、不做 batch>1**（D13；gen-2 重评）。

**分层架构**：

```
qforge 编译器前端（QNN IR → qbin 141 张量） → Q-MLIR pass（qnn→qisa）
  → Q-ISA（33 条 128-bit 定长指令，冻结）
  → qsim 时序模拟器（executor + timing_p6 天花板模型）
  → qrun Runtime + QMetal（HBM/SRAM 显式内存管理）
  → QCore RTL（rtl/：command_processor + matrix_engine + vector_engine +
     dma_engine + kv_quantdequant + kv_bfeed + SRAM 宏实例 + qcore_top）
  → ASIC 流程（asic/dc/ = Synopsys DC；asic/smc28/ = SMIC28 setup；
     legacy sky130 流程保留）
```

**关键目录**：

| 路径 | 内容 |
|---|---|
| `docs/spec.md` + `docs/spec-src/00-05` | 规格六分册（冻结，含全部裁决 D1-D18） |
| `docs/perf-research/` | 性能研究全链：roadmap / audit / flashattn / noc / decision/（DECISION + 全部实测报告 + 质量基线 JSON） |
| `docs/p1..p10/` | 各里程碑报告（roofline、M2a、ASIC 报告等） |
| `ref/` | PyTorch 参考模型（0.6B 全路径，与 HF greedy 20/20 逐位一致）；`gen_golden.py` 参数化 golden 再生成 |
| `golden/`（gitignored，5.2 GB） | 2540 op 三级 golden 目录 |
| `compiler/` | qbin/isa 编码/降级（lowering.py 等） |
| `qsim/` | 模拟器 + 52 个 pytest（含 B' 功能测试） |
| `qrun/` | Runtime + 全部实测工具（fold_verify.py 门槛/天花板、windowed_kv.py、o2_kv_int8.py、quality_baseline.py、spec_alpha.py 等） |
| `qforge/` | 量化（gptq.py / rot_quant.py / w4a16_gptq.py——后两者为已验证否决方向，保留存档） |
| `rtl/` | QCore RTL + `tb/`（co-sim 驱动）+ `ref/`（qsim_baseline、asicsnap 合成快照） |
| `asic/` | `dc/`（DC 流程脚本 + reports）、`smc28/`（setup_smic28.sh）、`sram_macros/`（宏仿真模型 + GEN.md + PORTS.md） |
| `plans/` | 全部立项计划（含评审修订历史） |
| `run_all_acceptance.sh` | 总验收（quick 12 / default 14 / full 15） |

---

## 2. 已完成交付（里程碑清单）

| 里程碑 | 交付 | 验收状态 |
|---|---|---|
| **M0** | 规格冻结：Q-ISA 33 条 / QNN / Q-MLIR / qforge / qrun / QMetal / qsim 命名体系；六分册 spec；15 条审计裁决 | ✅ |
| **M1** | Qwen Reference：ref/model.py 0.6B 全路径，greedy 20/20 与 HF 逐位一致；golden 2540 op | ✅ |
| **M2a/b** | qsim 时序模型；单层 {PF,DC}×{BF16,INT8} 4/4；decode 675.5（PF）/ 468.9（DC）token/s | ✅ |
| **M3** | qforge：141 张量 qbin、6 类投影×PF/DC 12/12 量化误差、qnn→qisa MLIR pass | ✅ |
| **M4** | qrun+QMetal：BF16 E2E 20/20、8/8@1024、5/5@4K、8K argmax 一致；INT8 10/10 | ✅ |
| **M5** | 硬件感知优化：写口径 80% 达标；KV 窗口重读修正（spec 04 §2.4）；KV.LOAD 冻结（D16） | ✅ |
| **M6** | QCore RTL：128×128 全尺寸 co-sim 4/4 + 单层 14/14 | ✅ |
| **M7** | 三级 golden：15 instance + L00/L13/L27 三向 ≤1 ULP | ✅ |
| **M9/P10b** | ASIC：Yosys/OpenSTA 129 MHz(tt) 流水化；DC 数据通路 331 MHz(tt)/169(ss)；控制平面 136/72 MHz；VCS 27 用例与 Verilator 字节一致 | ✅（sky130 口径，legacy） |
| **INT8** | 全链路部署：qforge+qrun per-128-group 激活校准，10/10 交叉一致 | ✅ |
| **INT4 W4A16** | 三端数据通路 + 打包位序锁定；部署质量 3/20 未达（AWQ 回退仍 3/20） | ⚠️ 如实收尾，per-64-group 归 backlog |
| **W1** | 开源打包：README/LICENSE(Apache-2.0)/reproduction/zhihu-outline；run_all_acceptance.sh | ✅ |
| **性能路线** | 三路调研（旋转量化/KV 离群/投机解码）→ DECISION（3 轮审计）→ 三门槛实测 → B' 立项落地 → SMIC28 宏化 → 锥流水化 + 工艺重立 | ✅ 详见 §3 |

---

## 3. 性能工作链（决策驱动，全链审计记录）

### 3.1 决策链
- **三路文献调研**（`docs/perf-research/decision/`）：rot-quant.md（旋转量化：7B+ 有效、<1B 无成功案例）、kv-outlier.md（KIVI/KVQuant 处方 = K per-channel / V per-token；QK-norm 折叠 = KVQuant 免校准特例）、speculative.md（免训练草案可行性）。
- **DECISION.md**（3 轮审计一致）：主方向 = 投机解码（后降级）、副 = KV 量化、受限 = W4A16。
- **三门槛实测**：①PLD 有效 α=30.6% < 0.4 → 投机解码降 gen-2；②**INT8-K（QK-norm 折叠）+ INT4-V：ΔPPL 1.266%/1.995%、交叉 20/20 过 → B' 立项**；③GPTQ/旋转五配置全否 → W4A16 归档。

### 3.2 已落地性能项
| 项 | 数字（全部冻结口径） | 状态 |
|---|---|---|
| O1 StreamingLLM 窗口化 | W=2048：ΔPPL 0.673%@4K/1.663%@8K、20/20 交叉一致；天花板 255→**866 tok/s@8K（3.4×）**、墙钟 2.58×；W=1024@8K 否决（3.316%） | ✅ 落地 |
| B'（INT8-K 折叠 + INT4-V） | 天花板 866→**1049.3（1.212×）**；保守 INT8 K+V 1006.2；V-only 930.8 | ✅ 数据通路+旋转器全链路落地，co-sim 3/3 0 ULP |
| 旋转器（on-read RoPE 流式融合） | B-feed 融合（无 staged 写、无新指令）；k-outer 数值契约；sink/窗口/PF 三例 0 ULP | ✅（rotator-impl.md） |
| SMIC28 SRAM 宏化 | kh4096x64/kn128x16/ang4096x64（SM18CA001/SM18CD001）；DC khat 存储墙破（内存 40.7→5.2 GB） | ✅ |
| 锥流水化（14 级 register-slice） | rope_sincos 锥 68.11 ns(sky130) → 非瓶颈；SMIC28 pre 116.4/87.0 → **post 217.9/164.5 MHz（1.87×/1.89×）** | ✅ |
| 量化器流水化（Track 2.1） | 倒数 NR 迭代逐操作切分 + quant mul/clip 两级流水；tt **4.59→1.57 ns，217.9→636.9 MHz（2.92×）**，ss **6.08→2.12 ns，164.5→471.7 MHz（2.87×）**；双角 top-10 均无 kv_quantdequant | ✅ tt/ss 全顶层重综合完成 |
| 数据通路 SMIC28 重立（Track 2.2a） | `synth_datapath` / `mac_bf16` 切 HDC30P140 RVT；tt/ss arrival **0.98–0.99 ns，1 ns 探针全 MET**；面积 0.0031/0.0019 mm² 量级 | ✅ 四次 compile_ultra 完成，物理裕量待 signoff |
| matrix 状态 RAM 宏化（Track 2.2b） | 4×acc/partial + 4×C seed + 1×scale = **9 个 kh4096x64**；逻辑/物理容量 208/288 KiB（72.22%）；宏面积 **0.707050 mm²**；tagged writeback queue + 首元素预取 | ✅ 四模式定向测试；TT 输入/Q scoped timing MET，SS Q MET 但输入最差 **-0.18 ns**；算术核仍为黑盒 |
| D18 工艺切换 | ASIC 基线 = SMIC28（28HKCP HDC30P140 RVT + SMIC28 宏）；sky130 全流程保留为 legacy | ✅ |

### 3.3 关键未兑现（如实）
- **B' 的 1049.3 是天花板模型数字，尚未 wall-clock 实测兑现**（qrun 端到端实跑未做）。
- 量化后指令发射 721,895 inst/token 是新的软墙（B6）。

---

## 4. 冻结规格与数字口径（权威清单，引用时必须用原值）

| 量 | 冻结值 | 出处 |
|---|---|---|
| INT8 峰值 | **32.77 TMAC/s**（128×128×2 op×1 GHz） | spec 04 |
| INT4 峰值 | 65.54 TMAC/s；BF16 = INT8/4 = 8.19 TMAC/s | spec 04 |
| HBM 带宽（sustained） | 读 720 / 写 240 GB/s | spec 03 |
| SRAM 带宽 | 读 512 / 写 256 B/cyc；bank = 字节地址 [7:4] | spec 03 |
| DC 模式带宽短缺 | **27.3×** | docs/p1/roofline.md |
| KV/ token（BF16） | 8B **147,456 B**；0.6B **114,688 B** | spec 04 §2.4 |
| decode 权重读（INT8） | 8B **7.57 GB** / 0.6B **0.596 GB** | docs/p1/roofline.md |
| B' 天花板锚点 | 26,317 cyc/层（HBM）= (15,730,944+2052×1568)/720；B-feed 25,104 ≤ 26,317；1049.3 = 1e9/(28×26,317+216,087) | fold-verify / timing_p6 |
| 窗口化天花板 | W=2048、R=S+W=2052 → 866.0 tok/s | qsim/timing_p6 |
| 质量基线 | **PPL@4K=32.2、@8K=27.43；HellaSwag acc_norm=0.55** | quality-baseline.json |
| 门槛判据 | ΔPPL ≤2%；交叉一致 ≥8/10；co-sim BF16 ≤1 ULP + trace/cycles 一致 | DECISION / M 系列 |
| 负通道统计（0.6B） | 折叠 scale 3.404% 负通道（min −3.86） | fold-verify |

---

## 5. 验证基础设施（怎么跑）

```bash
cd /home/lzl/project/newlpu

# 总验收（W1）
./run_all_acceptance.sh quick      # 12 PASS + 2 SKIP
./run_all_acceptance.sh            # 默认全量 14 项
./run_all_acceptance.sh full       # 默认全量 + m4，15 项

# qsim 全部测试（52 个）
python3 -m pytest qsim/ -q

# B' 专用 co-sim（sink/window/PF 三例，0 ULP 判据）
python3 rtl/tb/run_cosim_bprime.py

# SMIC28 matrix 状态宏壳（四数值模式 + 单口冲突 + CP 首元素预取）
bash asic/run_matrix_sram_check.sh

# 默认回归子集（vector 22 + KV 2；ROPE_pos40960 为既有 boundary note 8 ULP）
python3 - <<'EOF'   # 或读 run_cosim.py 内 run_vector_tests/run_kv_tests
import importlib.util
spec = importlib.util.spec_from_file_location("rc", "rtl/tb/run_cosim.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
rc.run_vector_tests(); rc.run_kv_tests()
EOF

# 主 co-sim（M2a 全尺寸用例既有时长 >15 min，属正常）
python3 rtl/tb/run_cosim.py

# B' 天花板 + 门槛（1049.3 复现）
python3 qrun/fold_verify.py --ceiling-only --out /tmp/ceiling.json

# golden 再生成（参数化）
python3 ref/gen_golden.py --model-dir <path> --device <cpu|cuda>

# Verilator 重建
cd rtl/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
  --top-module qcore_top -I.. ../qcore_top.sv sim_main.cpp --Mdir obj_dir
```

**判据解释**：co-sim 比较三件套 = trace（逐指令周期）、cycles（总数）、内存 dump（BF16 ≤1 ULP，B' 判据收紧为 0 ULP）。trace/cycles 由 `rtl/tb/cosim.py` 按冻结 per-instruction latency 常数解码生成，CP 同源计费——**墙钟周期变化对 trace 比较不可见**（功能执行与周期记账解耦）。

---

## 6. 决策登记册（D1-D18，摘要）

| 编号 | 裁决 | 状态 |
|---|---|---|
| D1-D11 | M0 审计 15 条中的规格裁决（命名/边界/判据等） | 冻结 |
| D12 | 三级 Golden Reference：PyTorch→qsim→RTL 数值一致 | 冻结 |
| D13 | 首代范围：无训练/多卡/70B/MoE/多模态/batch>1 | 冻结（gen-2 重评） |
| D14 | INT4 W4A16 打包位序（偶低/奇高半字节） | 冻结 |
| D15 | MATRIX 输出 dtype 语义（dequant→BF16；否则 srcA） | 冻结 |
| D16 | KV.LOAD 冻结：decode 单副本 + P7 内部广播总线；batch_stride_B=0 | 冻结 |
| D17 | batch=1 范围解释（spec §3.3） | 冻结 |
| D18 | **ASIC 工艺切换 sky130→SMIC28（28HKCP）**：双基线并存；公开仓库只放 RTL+脚本，PDK 产物本地化（历史已重写） | 冻结（当前基线） |

完整裁决记录在 `docs/spec.md` §3.3 与各分册。

---

## 7. 环境与工具链（本机）

- **Synopsys DC/VCS**：`DC_HOME=O-2018.06-SP1`，`LM_LICENSE_FILE=27000@bics109`；VCS 需 snps-centos7 兼容命名空间（glibc 2.17）。DC 编译超时上限建议单角落 ≥5h（compile_ultra 实测 tt >5h）。
- **SMIC28 PDK**（`/home/public/PDK/SMIC28/`）：
  - STDcell：`STDcell/SCC28NHKCP_HDC{30,35,40}P140_{LVT,RVT,ULVT,HVT}_V0p2`，已预解压（liberty/{0.8v,0.9v,1.0v} 含 basic/ccs/ecsm .lib 与 .db + lef/verilog）；当前流程用 HDC30P140 RVT；vendor .db 直接被 DC-2018 读取。
  - Memory：`SRAM_Ccompiler_ARM20240823/`（SM18CA001 单口 SRAM / SM18CE000 双口 RF / SM18CD001 单口 RF）。**必须用 skill `smic28-sram-compiler` 的 `prepare_compiler.sh` 兼容树**（no-space 路径 + bifrun 包 glibc 2.17 wrapper；直接 symlink 会导致 lef/gds2 失败）。生成用 `zsh` 显式跑入口脚本。已知限制：verilog_rtl 生成器挂起、apache_avm 仅特定 corner、ccs_t/ecsm_tv/ccs_tnv 未实现（可用 nldm/ecsm_t/ccs_tn）。
  - 宏产物稳定目录：`/home/public/PDK/SMIC28/macros_out/{kh4096x64,kn128x16,ang4096x64}`；再生成命令见各 `GEN.md`（`asic/sram_macros/<name>/GEN.md`）。
  - **宏 EMA 绑值**：`EMA[2:0]=011`（0.9V）、`EMAW[1:0]=01`、`EMAS=0`、`RET1N=1`。
  - corner 名：std cell `tt_v0p9_25c`/`ssg_v0p81_125c` ↔ 宏 `tt_ctypical_0p90v_0p90v_25c`/`ssg_cworstt_0p81v_0p81v_125c`。
- **SMIC28 DC 流程**：`asic/smc28/setup_smic28.sh`（宏再生成→.db 构建→corner 映射）；`asic/dc/run_dc.sh <corner> <design>`（`DC_TECH=sky130|smic28`，`DC_DESIGN=synth_top`）；宏库由 `build_macro_db.sh`（读 `SMIC28_MACRO_DIR`，默认 `/home/public/PDK/SMIC28/macros_out`）构建。
- **Verilator 4.038 兼容**：ARM 编译器产出的门级 .v（specify 块）与 Verilator 不兼容（UNOPTFLAT）——co-sim 用 `rtl/sram_macros.sv` 同端口行为模型（1 拍读/同步写），DC 用真 .lib。
- **GPU**：HuggingFace 模型缓存 `~/.cache/huggingface/`（quality-baseline 曾用 cuda；4B/8B 验证前需核实显存）。

---

## 8. 安全与合规（必读）

1. **P0 历史教训**：本地 VCS 环境快照（`.daidir/`）曾携带用户 `~/.zshrc` 中的 Anthropic token 进入仓库 → 已 gitignore `*.daidir/`、本地脱敏、远程克隆验证干净，并已建议用户**轮换该凭据**。**提交前必须执行密钥扫描**：`grep -rlE "sk-ant-[A-Za-z0-9]{10,}|Bearer [A-Za-z0-9]" <新文件>`。
2. **PDK 合规（D18）**：SMIC28 商业 PDK 产物（.lib/.lef/.gds2/.cdl/.db）**严禁入公开仓库**。已执行：git rm + filter-repo 历史重写（d8fb867→46a316e 等）+ force-push；`git log --all --full-history` 全历史扫描无任何 PDK 产物（仅项目自写 `mem_stub.lib`）。`.gitignore` 已覆盖 `.lib/.lef/.gds2/.cdl/.clf` 等。**git bundle 备份**在历史重写前制作（完整旧历史含 PDK 文件的归档，勿公开）。
3. 仓库只保留：RTL、流程脚本、宏 `.v` 仿真模型 + GEN.md + PORTS.md、全部报告文本。

---

## 9. 当前状态与未闭合项（backlog）

**当前基线数字（SMIC28，最终 report_timing）**：

| 口径 | tt | ss | 关键路径 |
|---|---|---|---|
| quant-pipeline（当前 RTL） | **1.57 ns → 636.9 MHz** | **2.12 ns → 471.7 MHz** | tt：`u_cp/len_reg[3]→hbm_addr[33]`；ss：`u_cp/u_dma/row_reg[2]→hbm_addr[39]`；B-feed 分别 1.55/2.09 ns 紧随；面积 0.266/0.269 mm² |
| synth_datapath（SMIC28） | **0.98 ns，1 ns MET** | **0.98 ns，1 ns MET** | tt：add/sub 级；ss：输入到 i32→fp32 首级；面积 0.003085/0.003237 mm² |
| mac_bf16（SMIC28） | **0.99 ns，1 ns MET** | **0.98 ns，1 ns MET** | tt：mul 中间级；ss：add 尾级；面积 0.001906/0.001961 mm² |
| rope post-pipeline（上一对照） | 4.59 ns → **217.9 MHz** | 6.08 ns → **164.5 MHz** | kv_quantdequant：`s_bits_reg[5]/[1]→hbm_wdata[0]`；面积 0.272/0.273 mm² |
| pre-pipeline（对照） | 8.59 ns → 116.4 MHz | 11.49 ns → 87.0 MHz | rope_sincos 锥 8.56/11.43 ns 紧随 |
| legacy（sky130 逻辑 + SMIC28 宏，跨工艺可达性口径） | 14.7 MHz（68.11 ns） | 未跑 | 历史参考 |

本轮 Track 2.2b 的双角 DC 报告为
`asic/dc/reports/synth_top_smic28_tt_025C_1v80_mxram.rpt` 与
`asic/dc/reports/synth_top_smic28_ss_100C_1v60_mxram.rpt`。两份报告均为
`compile=1`、1 ns ideal-clock 的 pre-layout 综合探针，顶层共 11 个宏/黑盒；矩阵九宏的
层次局部宏面积为 **707049.9141 µm²（0.707050 mm²）**，顶层 macro/black-box 合计为
**787126.751709 µm²**。

| corner | 全局最差 data arrival / slack | Fmax probe | total cell area | dynamic / leakage | 矩阵 SRAM 输入 scoped | 矩阵 Q 读 scoped |
|---|---|---:|---:|---|---|---|
| `tt_025C_1v80` | 1.47 ns / **-0.47 ns**（`C_reg[31][4] → hbm_addr[39]`） | **680.3 MHz** | 978016.837384 µm² | 121.2793 / 2.1169 mW | 0.80 ns / **+0.01 ns MET** | 0.48 ns / **+0.50 ns MET** |
| `ss_100C_1v60` | 1.93 ns / **-0.95 ns**（B-feed `s_ang2_reg[2][24] → s_r1_reg[2][16]`） | **518.1 MHz** | 980001.631439 µm² | 101.7342 / 16.7783 mW | 0.97 ns / **-0.18 ns VIOLATED** | 0.63 ns / **+0.34 ns MET** |

这说明宏化后的矩阵状态/控制壳和 Q 读回路径已经可由 DC 直接观察，但 SS 的 SRAM
输入 setup 仍未闭合；全局瓶颈仍在控制/B-feed。计算核 Liberty、vector/阵列物理核、
floorplan/CTS/布线及寄生 STA 仍是 signoff 前置条件，不能把本轮数字写成整芯片 1 GHz
收敛。

当前 tt/ss 报告分别为
`asic/dc/reports/synth_top_smic28_tt_025C_1v80_kvqd_pipe.rpt` 与
`asic/dc/reports/synth_top_smic28_ss_100C_1v60_kvqd_pipe_ss.rpt`。tt top-10 包含 6 条
`len_reg→hbm_addr`（1.57 ns）与 4 条 B-feed 寄存器路径（1.55 ns）；ss top-10 包含 3 条
DMA/长度寄存器到 `hbm_addr`（2.12 ns）与 7 条 B-feed 寄存器路径（2.09 ns）。双角均不再
包含 kv_quantdequant。相对 rope post-pipeline 同角基线，tt/ss Fmax 分别提升 2.92×/2.87×，
总 cell area 分别为 266379.2/268726.7 μm²；ss 同角面积下降 1.41%，低努力功耗估计为
dynamic 91.339 mW、leakage 16.380 mW。该数字沿用 1 ns ideal-clock 综合探针口径；报告仍有
B-feed `clk` 30804 loads 的 high-fanout 估算告警，不能替代 CTS/布局后 STA。

本轮功能验证：`pytest qsim/` **52 passed**；B' co-sim **3/3 PASS、0 ULP、trace/cycles 一致**；
随机与全零 K/V 定向用例的 HBM 写回 196 bytes 均与 executor 字节级一致；matrix 宏壳
四模式定向测试全过；quick 总验收 **12 PASS / 0 FAIL / 2 SKIP**（quick 规定跳过
golden3/full co-sim，B' co-sim 已另行验证 3/3）。

**未闭合项（全部有明确下一步）**：

| 项 | 性质 | 下一步 |
|---|---|---|
| B' 4B/8B 重检 | D10 义务 | fold_verify 同口径复测；模型下载/显存前置核实 |
| quant-pipeline signoff STA | tt/ss 全顶层综合已过，物理口径待闭合 | 补 CTS、互连寄生与布局后 STA，处理 B-feed 高扇出时钟 |
| matrix/vector 计算核物理集成 | matrix 状态 RAM 已真宏化；128×128 array 与 vector core 仍为黑盒 | 生成/导入计算核 Liberty 与物理视图，补 vector 状态 RAM，再做 floorplan/CTS/布线 |
| B' wall-clock 实测 | 兑现缺口 | qrun 端到端 vs 866 实测加速比 |
| 指令发射墙 721,895 inst/token | 软墙 | 描述符内批量（无新指令） |
| INT4 W4A16 质量 3/20 | 质量缺口 | per-64-group / 混合精度（排队） |
| 分页 KV、batch>1、QCore×N+NoC（O7）、MoE | gen-2 | 排队；D13 重评时启动 |

---

## 10. 下一阶段任务（详细，见交接时的规划）

**Track 1 — 数值/质量闭环（P0）**：①B' 4B/8B 重检（D10）：fold_verify 同种子同样本复测 ΔPPL/交叉/负通道统计；门槛裁决（过→全家族结论；不过→B' 仅 0.6B 口径 + per-head 混合精度备用）。②8B 冻结口径重算（B' 对 8B 的增益重推，预期 <1.212×）。

**Track 2 — Fmax 续攻（P1）**：①量化器路径 register-slice 与双角全顶层重综合 **已完成**（tt 636.9 MHz；ss 471.7 MHz）；②代表数据通路/BF16 MAC 的 SMIC28 tt/ss 重报 **已完成**（1 ns 探针全 MET）；③matrix 状态 RAM 的 9 宏物理壳与接口 timing **已完成**（Track 2.2b）；④计算核 Liberty + floorplan/CTS/互连寄生后的 signoff STA 待做。

**Track 3 — 性能兑现（P2）**：①B' 端到端 wall-clock（窗口化+INT8-K 折叠+INT4-V+流式旋转全链路 20 token 实测 vs 866 基线的实际加速比 + 差距归因）；②指令发射墙松绑（描述符内批量）。

**Track 4 — gen-2 预研（排队）**：NoC/QCore×N 草案（O7）、分页 KV、batch>1（O8）。

**依赖顺序**：1.1 →（1.2、3.1）；2.1 → 2.2a → 2.2b → 物理 signoff；3.2 可与
2.1 并行；Track 4 待稳定后评估。**每个立项必须走 subagent 评审循环至「评审一致，
可执行」**（本仓库不可违背的工作纪律）。

---

## 11. 文档索引（深入阅读顺序）

1. `docs/spec.md`（总纲 + D1-D18 裁决）→ `docs/spec-src/00-05`（六分册）
2. `docs/p1/roofline.md`（带宽/算力口径）、`docs/p2/m2a-report.md`、`docs/p10/asic-report.md`（§10.7 双工艺对比表）
3. `docs/perf-research/roadmap.md`（路线图）→ `decision/DECISION.md`（决策链 + §6/§7 实测裁决）→ `decision/rotator-impl.md`（B' 旋转器交付）→ `decision/bprime-impl.md` → 各实测报告与 JSON
4. `README.md`（含 SMIC28 setup 环境依赖）、`reproduction.md`（复现手册）、`UPLOAD-MANIFEST.md`
5. `plans/`（全部立项计划与评审修订历史；工作流范本）

---

## 12. 已知坑位清单（接手必读）

1. **脉冲握手设计**：`kv_bfeed` 的 `row_valid` 曾为粘性（置 1 不拉低）导致 CP 复制上一 k 的 row（B[k]=B[k−1] 偏斜）。规则：FSM 默认清零 + 完成脉冲 + 请求方复制后解除——新增任何 valid/req 握手照此模板。
2. **DC 并发抢 CPU**：多个 compile_ultra 并行会使各自收敛极慢（pre 首轮 ~7h 被 SIGKILL）。同机跑 DC 时串行或限 2 个。
3. **快照混用**：`rtl/ref/asicsnap/` 是 DC 合成快照（gitignored），改动 RTL 后必须重新同步，否则 DC 综合的是旧代码（曾导致 pre 与 post 数字相同才暴露）。
4. **ARM 门级 .v 与 Verilator**：specify 块不兼容 → co-sim 必须用行为模型（`rtl/sram_macros.sv`），DC 用真 .lib。
5. **测试 staging 地址重叠**：`run_cosim_bprime.py` 中 K/V/Q/softmax staging 区域部分重叠（写入序保证双方一致，测试仍有效但数据有互相覆盖）；新增测试时避免沿用此布局。
6. **window 假阳性教训**：STORE_BLOCK 必须写 `pos_base` 对应位置，否则 RTL 与 executor 同读零而 0 ULP 不验证任何东西。
7. **compiler 兼容树**：SMIC28 SRAM 编译器路径含空格 + bifrun glibc 问题 → 必须走 skill 的 `prepare_compiler.sh`；入口脚本 `#!/bin/ksh -p` 需 `zsh` 显式执行。
8. **executor CD 重构回归**：`_matrix` 里 `cd_mode/scale_dtype/scale_base` 曾因重构丢失定义——改 CD 相关代码时同步核验 executor 与 RTL 的 CD[20]/[19]/[18:0] 语义一致。
9. **DC Presto 严格性**：混合阻塞/非阻塞赋值对同一信号（b_slice）在 Verilator 只告警、DC 直接 VER-134 致命——新代码一律非阻塞。

---

*本文件是活文档：任何里程碑/裁决/基线数字变更后必须同步更新对应小节。*
