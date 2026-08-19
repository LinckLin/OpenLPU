# 上传清单（LinckLin/newlpu，公开仓库）

## 不上传（EXCLUDE，含理由）

| 路径 | 大小 | 理由 |
|------|------|------|
| `.venv/` | 4.8 GB | 本地虚拟环境 |
| `golden/` | 5.2 GB | 可再生成产物（`ref/gen_golden.py` 完整验收子集：prefill_seq128 + decode cache 0/512/1024/2048/4096/8192 + linear_wq_pf/dc，reproduction.md §1 有命令）；超 GitHub 2 GB 上限；派生自 Qwen3-0.6B |
| `rtl/ref/asicsnap/` | 56 MB | 综合用冻结快照（cp -r rtl 可再生成，asic-report §7 已述） |
| `rtl/ref/qsim_baseline/` | 240 KB | —— 保留（co-sim 脚本 sys.path 依赖，clone 后开箱即跑） |
| `obj_*/`、`*_obj/`（rtl/tb、fpga/tb、asic、rtl/ref 下） | 15M+ | Verilator 构建产物（如 `obj_dir`、`obj_dir_o1`、`obj_cr`、`obj_kv`、`obj_sram_shrink`、`sf_obj`、`sf_ops_obj`、`sf_angles_obj` 等） |
| `asic/dc/db/`、`asic/dc/gen*/` | 33 MB | DC 生成 .db 与 desugar 树（`asic/dc/build_db.sh` 等脚本可再生成） |
| `asic/gen/` | 212 KB | preprocess desugar 产物（脚本可再生成） |
| `asic/vcs/simv`、`asic/vcs/gen/`、`asic/vcs/csrc*` | ~1 MB | VCS 编译产物与派生快照 |
| `*.daidir/`（如 `asic/vcs/simv.daidir/`） | ~1 MB | VCS 运行时目录，内含环境快照（可能携带凭据） |
| `alib-52/` | 7.9 MB | ABC 综合 alib 产物 |
| `abc.history`、`.history_sta`、`.pytest_cache/` | 小 | 工具痕迹 |
| `*.pvl`、`*.pvk`、`*.syn`、`*.mr`、`*.svf`、`lc_output.txt` | 小 | Synopsys/EDA 工具痕迹 |
| `*.log`、`*.jou` | 小 | 工具日志 |
| 全部 `__pycache__/`、`*.pyc` | 小 | Python 缓存 |

## 上传（INCLUDE）

- 根：README.md、LICENSE、PLAN.md、UPLOAD-MANIFEST.md、run_all_acceptance.sh
- 源码：compiler/、qforge/、qrun/、qsim/、ref/（生成器脚本，不含 golden 数据）
- 硬件：rtl/（源 + tb 驱动，不含 obj_*/_obj/asicsnap）、fpga/（源 + tb，不含 obj_*/_obj）
- 流程：asic/（脚本 + tcl/sh/py/sv/lib + netlist 门级网表 + mem_stub，不含 db/gen/simv/csrc/daidir）
- 文档：docs/（全部报告与 spec 六分册）、plans/（12 份计划含评审记录；含 matrix 物理计算核推进计划）

## 敏感性检查结论

- 凭据模式扫描（hf_/ghp_/sk-/AKIA/password=）零命中；无 .env、无硬编码 token。
- `*.daidir/`（VCS 运行时目录）内含环境快照，可能携带 `ANTHROPIC_AUTH_TOKEN` /
  `VSCODE_GIT_IPC_AUTH_TOKEN` 等凭据——已由 .gitignore 排除；推送前主会话将警示轮换该 Anthropic 凭据。
- 模型权重在 ~/.cache（仓库外），不涉及。
