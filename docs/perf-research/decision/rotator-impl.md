# B'-旋转器流式融合落地 — 实施报告（rotator-impl）

> 目标（plans/rotator-plan.md v3，round-3 评审一致）：兑现 B'（INT8-K QK-norm 折叠 +
> INT4-V）的 **1049 tok/s** 条件口径 —— 把 KV 去量化 + on-the-fly RoPE 融入 attention 的
> B 操作数 feed，退役 staged BF16 写路径，无新指令。

## 0. 结论摘要

| 判定 | 结果 |
|---|---|
| **双口径收尾** | staged ≈627 tok/s 口径**退役**；流式（B-feed 融合）1049.3 为**主口径** |
| **时序模型** | `qrun/fold_verify.py --ceiling-only` 复现 **1049.3**（B-feed 项 25,104 ≤ HBM 26,317，HBM 保持 bound）✓ |
| **B-feed RTL 微硬件** | `rtl/kv_bfeed.sv`（新）+ command_processor B-feed 状态机 + kv_quantdequant LOAD 侧退役；lint 通过、Verilator 构建通过 |
| **B-feed 数值对齐** | ✅ **≤1 ULP**：sink/window/PF 三例 ALL PASS、0 ULP、trace/cycles 一致（K 旋转偏差根因 = row_valid 握手粘性，见 §3.1） |
| **默认 BF16 回归** | ✅ vector 22 例 + KV 2 例 = 24/24 PASS（23 例 0 ULP + 1 例 boundary note 8 ULP，ROPE_pos40960 既有宽松判据）（B-feed 仅 CD[31]=1 生效；M2a 全尺寸用例既有时长 >15min 未跑） |
| **qsim 回归** | ✅ 52 passed（含 B' 5 例；修复 CD 重构引入的 cd_mode/scale_dtype/scale_base 三处回归） |
| **测试迁移** | `run_cosim_bprime.py` LOAD 用例已改写为 B-feed 断言（含 sink 绝对位置/负通道/窗口真实数据/PF 四例）；`test_bprime_kv.py` 5 例为 executor 功能层双轨测试 |
| **DC Fmax 复测** | ✅ **存储墙已破、Fmax 已量化**：`khat[0:16383]`→`kh4096x64` 宏 + 行生产串行化（每周期 1×64b 读）后 compile_ultra 跑通（内存 40.7→5.2 GB）；**tt Fmax ≈ 14.7 MHz**，关键路径 = `rope_sincos` 组合锥 68.11 ns（逻辑侧，见 §3.2③；跨工艺口径，vs 基线 136 MHz） |

**落地判定：** B' 数据通路 + 流式旋转器（dequant+on-the-fly RoPE 融合进 attention B 侧
读路径）**三端落地且 ≤1 ULP**；时序口径 1049.3 由修正模型复现；默认回归不破坏。

## 1. 已落地项

### 1.1 CD/指令位分配（plan §3.2，逐位钉死）
- `CD[31]` = KV_QUANT，`CD[30]` = ROTATE_K，`CD[29:21]` = KV_IDX（`layer*8+head`，
  完成计划未显式分配的 k_norm/scale slab 寻址，**需评审项 ①**）。
- BMM 指令 reserved `[20:5]` = `pos_base[15:0]`（绝对位置 0..40959）。
- 双段基址：sink 段 `pos_base=0`、窗口段 `pos_base=pos−W+1`（与 windowed_kv.py 双
  KV.LOAD 结构一致）。
- 随件修订：`docs/spec-src/02-isa.md` §6.1（reserved→字段语义）+ §10（非零检查豁免）。
- 33 条指令计数不变（无新指令）。

### 1.2 RTL 落地
- **`rtl/kv_bfeed.sv`（新）**：B-feed dequant + on-the-fly RoPE。
  - K 路径：precompute `k_hat[128][N]`（INT8 q × 带符号折叠 `scale_c = bf16(s_q·k_norm[c])`），
    row 生产按 channel pair（`ch ^ 64`）做 HF rotate_half，绝对位置 `p = pos_base + token`。
  - V 路径：流式 INT4 nibble 去量化（`bf16(q4·s_v)`），无旋转。
  - 握手：`row_valid` 为完成脉冲（默认清零），与 CP 的 `row_req` 单周期脉冲配对（§3.1）。
- **`rtl/command_processor.sv`**：新增 `S_MX_BFEED_PRE`（K precompute）与 `S_MX_BFEED_B`
  （row 生产）状态；S_FETCH 解码 CD[31:30]/CD[29:21]/pos_base；mem_rd 多路复用接入 bfeed。
- **`rtl/kv_quantdequant.sv`**：LOAD 侧 staged 去量化（S_RDS/S_RD）**退役**，仅保留 APPEND
  量化（INT8-K 折叠 / INT4-V）。`write_mode`/`k_norm_base` 端口移除。
- **`rtl/qcore_pkg.sv`**：CD 位常量 + `POS_BASE` 位常量。

### 1.3 qsim executor 参考
- `qsim/executor.py`：`_matrix` 检测 `CD[31]` KV_QUANT → `_matrix_bprime` / `_bfeed_k` /
  `_bfeed_v`；`pos_base` 从原始指令字 `[20:5]` 读取。

## 2. 验证状态

| 层 | 用例 | 结果 |
|---|---|---|
| RTL lint / build | `verilator --lint-only` + `--cc --exe --build` | ✅ 通过，无 UNDRIVEN |
| 时序模型 | `fold_verify.py --ceiling-only` | ✅ `int8_k_fold_int4_v`=1049.3、`int8_kv_fold`=1006.2 |
| co-sim B-feed | `run_cosim_bprime.py`（sink/window/PF 三例） | ✅ 3/3 ALL PASS、0 ULP、trace/cycles 一致 |
| 默认 BF16 回归 | vector 22 例 + KV 2 例（run_cosim.py 子集） | ✅ 24/24 PASS（23 例 0 ULP + 1 例 boundary note 8 ULP） |
| qsim 回归 | `pytest qsim/` | ✅ 52 passed |
| DC Fmax | B-feed 关键路径影响（synth_top tt 复测） | ✅ 存储墙已破（khat→kh4096x64 宏 + 串行化）；compile_ultra 完成，**tt Fmax ≈ 14.7 MHz**（rope_sincos 锥 68.11 ns，逻辑侧，见 §3.2③） |

## 3. 修复记录与未闭合问题（需评审）

### 3.1 已修复（含根因）
1. **K 旋转数值偏差（曾阻塞）**：根因 = `kv_bfeed` 的 `row_valid` 握手粘性——置 1 后永不
   拉低，CP 在第二个及后续 row 请求时看到上一轮的陈旧 `row_valid`，把**上一 k 的 row**
   复制进 `b_slice`（精确解释观察到的 `B[k]=B[k−1]` 偏斜与 token 值错位）。修复：
   `row_valid` 改为完成脉冲（FSM 默认清零）+ CP 在复制后解除 `row_req`（单周期脉冲），
   双端握手对齐。修复后三例 0 ULP。
2. **window 例假阳性**：窗口段 STORE_BLOCK 曾写 pos 0..7 而 BMM 读 pos_base=8（双方同读
   零 → 0 ULP 不验证任何东西）。修复：`pos_start=POS_BASE`，窗口例现验证真实非零位置数据。
3. **executor CD 重构回归**：`cd_mode`/`scale_dtype`/`scale_base` 三处 NameError（B' 重构
   时丢失定义）。修复：按 CD[20]（mode）、CD[19]（scale dtype）、CD[18:0]（SRAM word addr
   ×16）语义补回。

### 3.2 未闭合
1. **CD[29:21] KV_IDX 为计划未显式分配的补全位**：plan v3 只钉死 CD[31:30]；k_norm/scale
   slab 寻址需要 `layer*8+head`，本实现占用 CD 剩余 reserved `[29:21]`（9 bit，恰好
   layer 6b + head 3b）。请评审是否接受该补全。
2. **kv_quantdequant LOAD 侧退役后 KV.LOAD 量化路径**：RTL 已退役（量化 LOAD 走 BF16 DMA
   回退）；executor 仍保留 `_kv_read` 参考（GATHER backlog，量化 GATHER 明确 out-of-scope）。
   两侧不一致已在文档层面记录。
3. **DC 关键路径影响（存储墙已破，Fmax 见下）**：原根因 = B-feed K 行生产 `for(i<128)`
   全并行展开出 256 个 `khat` 并行读口（256×16K:1 mux + 128×`rope_sincos` 组合锥），
   compile_ultra 卡 Pass 1 Mapping >45 min 不收敛（40.7 GB）。本任务（sram-macro 重构）
   修复如下：
   - **存储介质**：`khat[0:16383]` → `kh4096x64` SMIC28 单口 SRAM 宏（4096×64，每 64b
     字 = 同 ch 的 4 token×16b bf16，地址 `{ch[6:0], n[6:2]}`）；`knorm[0:127]` →
     `kn128x16` RF 宏。写路径 (b)：S_Q 循环改 n-组内层（n=4k..4k+3 连续写同一字）+
     s_q 128×2B 全量寄存（消除按 ch 重读）+ 尾组 N%4≠0 零填充（消费端仅取 j<N）。
   - **行生产串行化**：每周期 1 次 64b 读；DC（N=4）2 读/行、PF（N=128）64 读/行（ch
     与 ch^64 各一读）；旋转锥每周期 4 token（≈4×5 fp32 op 深，合成友好）；`row_valid`
     脉冲握手保持（默认清零 + 完成脉冲，CP 复制后解除 row_req）。
   - **宏接线**：kh/ang 大写引脚（Q/CLK/CEN/WEN/A/D/EMA/EMAW/EMAS/RET1N）、kn 小写
     （q/clk/cen/wen/a/d/ema/emaw/emas/ret1n）；EMA=011/EMAW=01/EMAS=0/RET1N=1 绑死。
   - **仿真模型**：编译宏的 gate-level `.v`（specify 块 + timing-check 反馈环）在
     Verilator 4.038 下 UNOPTFLAT 不可 elaborate——co-sim 用同名同引脚的行为模型
     （1 拍读、同步写，功能逐位一致）；DC 用真实 `.lib`→`.db`（宏黑盒 + set_dont_touch）。
   - **跨工艺口径声明**：SMIC28 宏（0.9V/28nm `.lib`）与 sky130 逻辑（1.8V/130nm）
     **混搭 ≠ 流片口径**——本复跑目的 = khat 墙消除后控制平面逻辑 Fmax 的**可达性评估**
     （关键路径在逻辑侧则数字有参考价值；在宏侧则如实报宏独立 timing）。
   - **DC 结果（compile_ultra 完成，tt 角落）**：elaborate + link 成功（`kh4096x64`/
     `kn128x16` 按 `.db` 链接，唯一 unresolved = 既有 8 MiB scratchpad `sram_macro` 黑盒，
     非本任务宏）；compile_ultra **越过原 khat 映射卡点**——内存由 40.7 GB 收敛至 ~5.2 GB
     （无存储墙），kv_bfeed 正常综合，跑通 Pass 1 Mapping → Mapping Optimizations →
     Global Optimization → Delay → WLM Backend 全阶段。最终 `report_timing`：
     - **关键路径 = `pos_base_reg[2] → row_reg[40][19]`，data arrival = 68.11 ns，
       Fmax = 1/68.11 ns ≈ 14.7 MHz**（vs 基线 136 MHz / 7.33 ns）。关键路径全在**逻辑侧
       rope 锥**（`rope_sincos` Cody-Waite 归约 + Hermite 插值 + rotate 乘加链，纯 sky130
       组合逻辑），**不含宏**（宏 CLK→Q = 0.41 ns，非瓶颈）。
     - **结论**：khat 存储墙已消除（内存 40.7→5.2 GB、无 256 读口爆炸），但 on-the-fly
       RoPE 的 `rope_sincos` 组合锥（68 ns）成为新的 Fmax 上限——**证实计划的阶段 2 降路径
       必要**：`ang4096x64` 位置缓存（cos/sin 预计算入宏、逐行计算改查表）是消除该锥的
       直接手段。跨工艺口径：关键路径 100% 在 sky130 逻辑侧，故 14.7 MHz 为控制平面逻辑
       Fmax 的可达性下界（宏不参与）。
     - **ss 角落未跑（验收项如实记录）**：tt 单一角落 compile_ultra 已 >5 h（同基线
       136 MHz 方法学），按计划「如超时只跑 tt 并如实说明」执行；ss Fmax 未量化。
       pre-macro 阶段的过期 ss 报告已标注 `.STALE-presto` 后缀避免误读。

## 4. 复现

```bash
cd /home/lzl/project/newlpu
python3 qrun/fold_verify.py --ceiling-only --out /tmp/ceiling.json   # 1049.3 / 1006.2
cd rtl/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
  --top-module qcore_top -I.. ../qcore_top.sv sim_main.cpp --Mdir obj_dir
cd ../.. && python3 rtl/tb/run_cosim_bprime.py                        # 3/3 ALL PASS
python3 -m pytest qsim/ -q                                            # 52 passed
# 宏 .lib -> .db 转库（SMIC28 宏 + sky130 逻辑并列；跨工艺可达性口径）
bash asic/dc/build_macro_db.sh
# DC Fmax 复测（khat 墙已破；tt + ss 两角落，compile_ultra 见 §3.2③）
./asic/dc/run_dc.sh tt_025C_1v80 synth_top && ./asic/dc/run_dc.sh ss_100C_1v60 synth_top
```
