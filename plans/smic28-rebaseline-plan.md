# SMIC28 全流程重立（提案 v1，待评审）

> 目标（用户指令 ②）：ASIC 工艺基线切换 **SMIC28（28HKCP，0.9V）**，消除跨工艺混搭。
> 现状：DC 流程 = sky130 逻辑（1.8V/130nm）+ SMIC28 宏混搭（可达性口径）；基线
> 136/72 MHz(tt/ss) 控制平面、129 MHz Yosys、331 MHz 数据通路均为 sky130 口径。
> 本机资源：`/home/public/PDK/SMIC28/`（STDcell = SCC28NHKCP_HDC30/35/40P140
> LVT/RVT/ULVT/HVT + CDK tarball；Memory = SRAM 编译器已用；PDK = 物理验证）。

## 1. 范围与口径

1. **新基线**：控制平面（command_processor 真综合，含 kv_bfeed/kv_quantdequant）与
   数据通路（matrix_engine/vector_engine，黑盒口径同前）在 SMIC28 std cell + SMIC28
   宏下 DC compile_ultra，tt/ss 双角落（宏已按 tt_ctypical_0p90v_0p90v_25c /
   ssg_cworstt_0p81v_0p81v_125c 生成，与 std cell 库 corner 对齐核实）。
2. **旧基线保留**：sky130 流程与全部历史报告不删（legacy 口径并存），新 SMIC28 数字
   成为当前基线；docs/p10/asic-report.md + docs/spec.md 增 **D18 裁决**（工艺切换、
   理由、两口径对比表）。
3. **开源合规（关键，含历史处置）**：商业 PDK 产物（.lib/.db/.gds2/lef/cdl）**不入公开
   仓库**——RTL + 流程脚本进仓库；PDK 产物由 `setup_smic28.sh` 从
   `/home/public/PDK/SMIC28` 本地生成（gitignore 覆盖）；README 注明环境依赖。
   **历史处置（P1 裁决）**：宏 .lib/.lef 已在 commit d8fb867 推送公开 origin——主选
   (b) `git filter-repo` 历史重写 + force-push + GitHub 缓存清理（仓库年轻、2 提交，
   破坏既有 clone 成本可接受；克隆数极少）；兜底 (a) 若重写失败则保留历史、在 README
   与 D18 如实记录暴露范围。裁决结果记入 D18。
4. **生成链改造（P2）**：`build_macro_db.sh` 输入路径硬编码 `asic/sram_macros/*/*.lib`
   ——改读本地生成目录（`SMIC28_MACRO_DIR` 环境变量，**默认锚定**
   `/home/public/PDK/SMIC28/macros_out`）；新增 `asic/smc28/setup_smic28.sh`：
   宏 .lib 源 = 编译器包（`SRAM_Ccompiler_ARM20240823`，按 GEN.md 命令再生成至稳定
   本地目录，非临时 /tmp）；std cell .lib/.db 已在库树内预解压；映射 corner
   （tt_v0p9_25c / ssg_v0p81_125c ↔ 宏 tt/ssg）；构建 `asic/dc/db/` 宏库；默认路径
   写入 README（换机复现链闭合）。

## 2. 步骤

1. 探索（**预解压已核实**）：`STDcell` 十棵库树已解压（HDC30/35/40P140 ×
   LVT/RVT/ULVT/HVT），liberty/{0.8v,0.9v,1.0v} 含 basic/ccs/ecsm .lib 与 .db、
   lef/verilog 齐备——先取 HDC30P140 单库跑通；
2. lc_shell 转 .db **仅作兜底**（已随库 .db 若因新 LC 版本被 DC-2018 拒绝，从 .lib 重转）；
   corner/电压/温度与宏库对齐（std cell tt_v0p9_25c / ssg_v0p81_125c ↔ 宏
   tt_ctypical_0p90v_0p90v_25c / ssg_cworstt_0p81v_0p81v_125c）；
3. `dc_top.tcl` 双工艺参数化（DC_TECH=sky130|smic28，库路径/电压/corner 映射表）；
   Yosys/OpenSTA 若可用 SMIC28 .lib 则做交叉验证，不可用则如实记录（DC 为主）；
4. DC compile_ultra tt/ss：控制平面（含当前流水化前 RTL；与计划 A 的流水化版本先后
   各跑一次，记录 Fmax 前后对比）；
5. 报告：新基线数字 + 关键路径 + 双工艺对比表（**legacy 列标注「跨工艺可达性口径
   （sky130 逻辑 + SMIC28 宏）」**，非纯 sky130；SMIC28 新值；数据通路 331 MHz 同口径
   重报）；
6. 仓库清理：宏 .lib/.lef 移出 git（保留 .v/GEN.md/PORTS.md），.gitignore 更新。

## 3. 验收

- SMIC28 tt/ss 双角落 compile_ultra 跑通，Fmax 数字 + 关键路径如实；
- 双工艺对比表 + D18 裁决入 docs；开源仓库无商业 PDK 产物泄漏（git 扫描验证）；
- setup/生成脚本可复现（指向本地 PDK 路径）。

## 4. 风险

| 风险 | 对策 |
|------|------|
| 已随库 .db 被 DC/LC-2018 拒绝（新版 LC 构建） | 从库树内 .lib 用 lc_shell 重转（在树内，风险受控） |
| 多阈值库变体（LVT/RVT/ULVT/HVT）选择不当 | 先 HDC30P140 单库跑通；变体对比后择优并报告 |
| compile_ultra 时长（tt 上次 >5h） | 双角落异步跑；先 tt 后 ss；超时如实 |
| 历史重写失败（force-push 被拒） | 兜底 (a)：保留历史 + README/D18 如实记录暴露范围 |
| 公开仓库误入 PDK 产物 | 提交前 git 扫描 + .gitignore 验证 |
| OpenSTA 不支持 SMIC28 lib 格式 | 以 DC 为主，OpenSTA 口径弃用或如实标注 |

## 5. 需评审关注点

1. 双基线并存与 D18 裁决的处置；开源合规方案（宏 .lib 移出仓库）是否充分；
2. 探索性步骤（CDK tarball）的失败模式与兜底；
3. 与计划 A（流水化）的执行顺序（建议：SMIC28 流程先跑当前 RTL 拿新基线，再跑流水化
   版本拿 Fmax 对比）；
4. 若一致，请声明「评审一致，可执行」。
