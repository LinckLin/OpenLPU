# kv_bfeed 合成化重构 + SMIC28 SRAM 宏 — 立项计划（提案 v1，待评审）

> 背景：B'-旋转器流式融合已交付（AGREE，3/3 co-sim 0 ULP，1049.3 主口径）。唯一未闭合
> 验收项 = DC Fmax 未量化：compile_ultra 卡在 `kv_bfeed` 的 `khat[0:16383]`（16K×32）
> 平面数组——K row 生产的 `for(i<128)` 展开出 **256 个并行读口**（256×16K:1 mux +
> 128×rope_sincos 组合锥），40.7 GB 不收敛。用户指示：本机有 SMIC28 SRAM 编译器
> （`/home/public/PDK/SMIC28/SRAM_Ccompiler_ARM20240823`，已验证 headless 可用）。
> 本计划 = 用真实 SRAM 宏替代平面 khat + 行生产串行化，消除合成墙，量化 Fmax。

## 1. 问题根因（已核实）

1. **256 读口是建模捷径不是微架构**：co-sim 的 `khat` 按 `[ch][token]` 通道主序存放、
   每周期全行（128 token）并行读出喂 128 路旋转锥。时序模型（1049.3 的 B-feed 项
   25,104 ≤ 26,317）只约束**旋转输出总量**（2052×2048 B/层 @256 B/cyc = 16.4K cyc），
   不约束 co-sim 的实现顺序（co-sim 周期 ≠ 性能模型，性能由冻结常数计费）。
2. **数值契约**：矩阵引擎 k-outer 累加顺序必须保持（fp32 非结合）——本次重构不改
   引擎的 `b_slice` 消费顺序（仍按 mx_k 逐通道行），只改 khat 的存储介质与读出方式。
3. **存储真值**：khat 存的都是 bf16-exact 值（32-bit fp32 位型）→ 宏存 16-bit bf16
   无精度损失；写入路径 `bf16(x)` 与读出 `bf16_to_fp32` 保持 bit 级一致。

## 2. 方案（主选：SMIC28 宏实例化）

### 2.1 宏生成（skill smic28-sram-compiler 流程，已验证编译器 headless 可用）
| 宏 | 编译器 | 配置 | 用途 |
|---|---|---|---|
| `kh4096x64` | SM18CA001 单口 SRAM | words=4096, bits=64, mux=8, mvt=BASE, write_mask=off, pipeline=off, ser=none | khat：4096×64 = 32 KB = 128 ch × 128 token × 16b（PF 满瓦片）；地址 `{ch[6:0], n[6:2]}`（每 64b 字 4 entry） |
| `kn128x16` | SM18CD001 单口 RF | words=128, bits=16, mux=2, mvt=BASE | k_norm 折叠表（128×2B，带符号 bf16） |
| `ang4096x64` | SM18CA001 单口 SRAM | words=4096, bits=64, mux=8（或 SM18CD001） | cos/sin 位置缓存：128 位置 × 64 对（cos,sin）bf16 = 256 B/位置 × 128 = 32 KB |

**EMA 绑值（skill release-note）**：`EMA[2:0]=011`（0.9V）、`EMAW[1:0]=01`、`EMAS=0`——
集成时按此绑死，写入 kv_bfeed 重构项。
`lef-fp`/`gds2`/`lvs`（物理视图存档，本计划不流片，仅备用）。生成用 skill 的
`prepare_compiler.sh` 兼容树（no-space + bifrun glibc 2.17 wrapper）。

### 2.2 kv_bfeed 重构（rtl/kv_bfeed.sv）
2. K 预计算（S_KNORM/S_SQ/S_Q）不变，写入改宏端口。**写路径二选一（实现期钉死）**：
   (a) 每 entry RMW（读 64b 字-改 lane-回写，单口宏约 2-3 拍/entry）；(b) S_Q 循环
   重构为 n-组内层（n=4k..4k+3 一组连续写同一字）+ s_q 全量寄存（128×2B）避免重读——
   推荐 (b)，预计算拍数与现实现同阶。
3. **行生产串行化**：S_DONE 的 `for(i<128)` 全并行 → 每周期 1 次 64-bit 读。
   rotate_half 每 token 需要 ch 与 ch^64 两通道（xr/xi），按 `{ch[6:0], n[6:2]}`
   布局二者必在不同字：**每行 DC（N=4）2 读、PF（N=128）64 读**（每 64b 字 = 同 ch
   的 4 token×16b）；旋转锥每周期处理读出的 4 token（约 4×5 fp32 op 深，锥深 ≈10 op，
   合成友好）；row_valid 脉冲握手保持（上轮 P0 修复不回退）。
4. `knorm` → `kn128x16` RF 宏（写入不变）；S_KNORM 读改宏端口。
5. 角度缓存（阶段 2）：`rope_sincos` 逐位置计算 64 对 cos/sin 存入 `ang4096x64`，
   8 head 复用同一缓存（每 token 每层只算一次）；阶段 1 保留现有 rope_sincos 逐行
   计算（锥深不变），宏化作为 Fmax 不够时的降路径。

注：co-sim 周期变化与 trace 判据——`expected_trace` 由 `rtl/tb/cosim.py` 按冻结
per-instruction latency 常数解码生成（CP 同源计费），executor 不生成 trace；串行化
引起的墙钟周期变化对 trace/cycles 比较不可见，无冻结锚点受影响。

### 2.3 DC 复跑
- 宏 `.lib`（nldm）转库：`lc_build_db.tcl` 硬编码 sky130，需新增宏库变体脚本
  （读 SMIC .lib → .db）；**角落映射**：SMIC `tt_ctypical_0p90v_0p90v_25c` ↔ sky130
  `tt_025C_1v80`、SMIC `ssg_cworstt_0p81v_0p81v_125c` ↔ sky130 `ss_100C_1v60`——
  宏 .lib 在生成时即按角落出两个（`-corners`），与 sky130 逻辑库并列读入（DC 多库）。
- **诚实口径声明**：SMIC28 宏（0.9V/28nm 时序）与 sky130 逻辑（1.8V/130nm）混搭 ≠
  流片口径——本复跑目的 = **khat 墙消除后控制平面逻辑 Fmax 的可达性评估**（关键路径
  若在逻辑侧则数字有参考价值；若在宏侧则报告宏的独立 timing 数据）。报告如实标注。
- tt/ss 两角落 compile_ultra，对比基线 136 MHz(tt)/72 MHz(ss)。

## 3. 验证

1. **数值**：bprime co-sim 3/3 复跑 **0 ULP**（含 sink 绝对位置/负通道断言、window 真实
   数据、PF 例）；默认回归子集 24/24；qsim 52 不破坏。
2. **功能等价**：重构后 co-sim 的 khat 读写与重构前逐值一致（通过 0 ULP 对比证明）。
3. **合成**：verilator lint（宏仿真模型 + 顶层）；DC elaborate+link+compile_ultra 过，
   Fmax 数字 + 关键路径报告；若关键路径落在宏端口逻辑，如实报告宏 timing。
4. **可复现**：宏生成命令 + shim 路径 + 报告记录；生成物存 `asic/sram_macros/`。

## 4. 验收

- kv_bfeed 重构后 bprime co-sim 3/3 **0 ULP** + 默认回归不破坏；
- DC compile_ultra tt/ss 跑通，Fmax 数字（含跨工艺口径声明）写入 rotator-impl.md §3.2③；
- 宏生成物（verilog/liberty/lef/gds2）落盘 `asic/sram_macros/`，生成命令可复现；
- 报告更新 DECISION.md §6 与 rotator-impl.md。

## 5. 风险

| 风险 | 对策 |
|------|------|
| 宏最小配置（4096×64 mux8）编译器不支持 | 按 skill 已验证配置降级（1024×32 已验证）；32 KB 容量不足则 8 宏拼，或降 PF 瓦片 N=64（16 KB → 4 宏拼） |
| 串行化改变 co-sim 周期（433 等） | expected_trace 由 cosim.py 按冻结 latency 常数解码（CP 计费同源）；墙钟周期不入比较，无冻结锚点受影响 |
| 跨工艺混搭被误读为流片口径 | 报告 + DC 脚本注释双重声明「可达性评估」 |
| 宏读延迟（1 cyc）破坏 row_valid 握手时序 | FSM 加读等待态；bprime co-sim 0 ULP 为判据 |
| 关键路径仍在逻辑侧（rope 锥） | 报告如实；阶段 2 角度缓存宏化作为降路径 |

## 6. 需评审关注点

1. 方案主选（宏实例化 + 行生产串行化）与「数值契约保持（k-outer 顺序不变）」的论证；
2. 跨工艺混搭口径是否诚实充分（可达性评估，非流片口径）；
3. 宏配置选择（kh4096x64/kn128x16/ang4096x64）与降级路径；
4. 若一致，请声明「评审一致，可执行」。
