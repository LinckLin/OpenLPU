#!/usr/bin/env bash
# run_all_acceptance.sh — QCore 全量验收脚本（八组件，逐条对照 README「组件索引表」）。
#
# 用法：
#   bash run_all_acceptance.sh            默认全量：八组件全跑，runtime 仅做 m4 证据文件校验
#   bash run_all_acceptance.sh --quick    快速模式：跳过 golden3 / co-sim / m4 全量重跑
#   bash run_all_acceptance.sh --full     默认全量 + runtime 显式全量重跑（python3 qrun/m4.py，小时级）
#   bash run_all_acceptance.sh --help
#
# 汇总表列：组件 / 命令 / 结果 / 耗时(s)。
# 结果取值：PASS | FAIL | SKIP(quick) | SKIP(工具缺失)。
# ASIC 段（verilator lint / yosys synth / OpenSTA）按工具链在位性自动判定，缺失则不伪造 PASS。
# 任一 FAIL → exit 1。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

usage() {
  cat <<'EOF'
QCore 全量验收脚本（八组件，对照 README 组件索引表）

用法:
  bash run_all_acceptance.sh            默认全量（八组件；runtime 仅 m4 证据校验）
  bash run_all_acceptance.sh --quick    快速（跳过 golden3 / co-sim / m4 全量）
  bash run_all_acceptance.sh --full     默认全量 + m4.py 显式全量重跑（小时级）

结果取值: PASS | FAIL | SKIP(quick) | SKIP(工具缺失)
任一 FAIL → exit 1。
EOF
}

QUICK=0
FULL=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --full)  FULL=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $a" >&2; usage >&2; exit 2 ;;
  esac
done

# ---- 工具链探测（ASIC 段；缺失则 SKIP(工具缺失)） ----
VERILATOR_BIN="$(command -v verilator 2>/dev/null || true)"
YOSYS_BIN="${YOSYS:-$(command -v yosys 2>/dev/null || true)}"
[ -z "$YOSYS_BIN" ] && [ -x "$HOME/.eda/yosys-src/yosys" ] && YOSYS_BIN="$HOME/.eda/yosys-src/yosys"
STA_BIN="${STA:-$(command -v sta 2>/dev/null || true)}"
[ -z "$STA_BIN" ] && [ -x "$HOME/.eda/opensta/usr/bin/sta" ] && STA_BIN="$HOME/.eda/opensta/usr/bin/sta"
CORNER=tt_025C_1v80
TT_LIB="${TT_LIB:-$HOME/.eda/liberty/sky130_fd_sc_hd__${CORNER}.lib}"

# ---- 结果记录 ----
RESULT_FILE="$(mktemp -t qcore_accept.XXXXXX)"

now_ns() { date +%s%N; }
fmt_secs() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.1f", (b-a)/1e9}'; }
trunc() {  # $1=字符串 $2=最大字节数；超长补省略号，避免长命令撑破汇总表对齐
  local s="$1" n="$2"
  if [ "${#s}" -gt "$n" ]; then
    printf '%s…' "${s:0:$((n - 1))}"
  else
    printf '%s' "$s"
  fi
}

record() {  # $1=组件 $2=命令 $3=结果 $4=耗时(s)
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULT_FILE"
}

# run_step <组件> <日志文件> -- <命令...>
run_step() {
  local comp="$1" log="$2"
  shift 2
  [ "${1:-}" = "--" ] && shift
  local cmd="$*"
  local t0 t1 rc
  t0=$(now_ns)
  if "$@" >"$log" 2>&1; then rc=0; else rc=$?; fi
  t1=$(now_ns)
  local secs; secs=$(fmt_secs "$t0" "$t1")
  if [ "$rc" -eq 0 ]; then
    record "$comp" "$cmd" "PASS" "$secs"
  else
    record "$comp" "$cmd" "FAIL" "$secs"
  fi
  return "$rc"
}

# ---- 预检：Verilator 可执行（RTL co-sim / golden3 与 FPGA smoke 依赖） ----
if [ ! -x rtl/tb/obj_dir/Vqcore_top ]; then
  echo "[build] rtl/tb/obj_dir/Vqcore_top 缺失，按 reproduction §4 构建 ..."
  (cd rtl/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
    --top-module qcore_top -I.. ../qcore_top.sv sim_main.cpp --Mdir obj_dir) \
    || echo "[warn] RTL 构建失败（golden3 / co-sim 将 FAIL）"
fi
if [ ! -x fpga/tb/obj_dir/Vqcore_fpga_top ]; then
  echo "[build] fpga/tb/obj_dir/Vqcore_fpga_top 缺失，按 reproduction §5 构建 ..."
  (cd fpga/tb && verilator --cc --exe --build -j 16 -O2 -Wno-fatal -Wno-WIDTH \
    --top-module qcore_fpga_top -I../../rtl -I.. ../../fpga/qcore_fpga_top.sv \
    sim_fpga_main.cpp --Mdir obj_dir) \
    || echo "[warn] FPGA 构建失败（fpga smoke 将 FAIL）"
fi

LOG_DIR="$(mktemp -d -t qcore_log.XXXXXX)"

echo "== QCore 全量验收 =="
if [ "$QUICK" -eq 1 ]; then
  echo "模式: quick（跳过 golden3 / co-sim / m4 全量）"
elif [ "$FULL" -eq 1 ]; then
  echo "模式: full（含 python3 qrun/m4.py 显式全量重跑，小时级）"
else
  echo "模式: 默认全量（含 golden3 + co-sim；m4 仅证据校验）"
fi
echo

# ---- 1. qsim：pytest 两文件（ISA 字段 + VECTOR/KV 数值） ----
run_step "qsim" "$LOG_DIR/qsim_pytest.log" -- \
  python3 -m pytest qsim/test_isa_fields.py qsim/test_vector_kv.py -q
echo "[qsim pytest] $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 2. qsim：test_m2a（非 pytest 用例，显式单跑，4 case 判据） ----
run_step "qsim" "$LOG_DIR/qsim_m2a.log" -- python3 qsim/test_m2a.py
echo "[qsim m2a]    $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 3. M5：timing_p6（双 target PASS：ctx 4096 与 ctx 8192） ----
t0=$(now_ns)
python3 qsim/timing_p6.py >"$LOG_DIR/timing_p6.log" 2>&1
rc=$?
t1=$(now_ns)
secs=$(fmt_secs "$t0" "$t1")
p4096=$(grep -cE 'ctx 4096:.*PASS' "$LOG_DIR/timing_p6.log" 2>/dev/null); p4096=${p4096:-0}
p8192=$(grep -cE 'ctx 8192:.*PASS' "$LOG_DIR/timing_p6.log" 2>/dev/null); p8192=${p8192:-0}
if [ "$rc" -eq 0 ] && [ "$p4096" -ge 1 ] && [ "$p8192" -ge 1 ]; then
  record "M5" "python3 qsim/timing_p6.py" "PASS" "$secs"
else
  record "M5" "python3 qsim/timing_p6.py" "FAIL" "$secs"
fi
echo "[M5 timing]   $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 4. compiler：verify_m3（12 例 = 6 类 × PF/DC + 结构） ----
run_step "compiler" "$LOG_DIR/verify_m3.log" -- python3 qforge/verify_m3.py
echo "[compiler m3] $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 5. runtime：m4 默认证据文件校验（docs/p5/m4-report.md BF16 四判据） ----
t0=$(now_ns)
EVID="docs/p5/m4-report.md"
ok=0
if [ -f "$EVID" ] \
  && grep -qF '20/20' "$EVID" \
  && grep -qF '8/8' "$EVID" \
  && grep -qF '5/5' "$EVID" \
  && grep -qF 'argmax 一致：**True**' "$EVID"; then
  ok=1
fi
t1=$(now_ns)
secs=$(fmt_secs "$t0" "$t1")
if [ "$ok" -eq 1 ]; then
  record "runtime" "grep docs/p5/m4-report.md (BF16 四判据)" "PASS" "$secs"
else
  record "runtime" "grep docs/p5/m4-report.md (BF16 四判据)" "FAIL" "$secs"
fi
echo "[runtime m4]  $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 6. runtime：m4 全量重跑（仅 --full；--quick 优先跳过） ----
if [ "$FULL" -eq 1 ]; then
  if [ "$QUICK" -eq 1 ]; then
    record "runtime" "python3 qrun/m4.py" "SKIP(quick)" "0"
  else
    run_step "runtime" "$LOG_DIR/m4_full.log" -- python3 qrun/m4.py
  fi
fi

# ---- 7. golden3（三级 golden） ----
if [ "$QUICK" -eq 1 ]; then
  record "golden3" "python3 rtl/tb/run_golden3.py" "SKIP(quick)" "0"
  echo "[golden3]     SKIP(quick)"
else
  run_step "golden3" "$LOG_DIR/golden3.log" -- python3 rtl/tb/run_golden3.py
  echo "[golden3]     $(tail -1 "$RESULT_FILE" | cut -f3)"
fi

# ---- 8. RTL：co-sim ----
if [ "$QUICK" -eq 1 ]; then
  record "RTL" "python3 rtl/tb/run_cosim.py" "SKIP(quick)" "0"
  echo "[RTL co-sim]  SKIP(quick)"
else
  run_step "RTL" "$LOG_DIR/cosim.log" -- python3 rtl/tb/run_cosim.py
  echo "[RTL co-sim]  $(tail -1 "$RESULT_FILE" | cut -f3)"
fi

# ---- 9. FPGA：集成 smoke ----
run_step "FPGA" "$LOG_DIR/fpga_smoke.log" -- python3 fpga/tb/run_fpga_smoke.py
echo "[fpga smoke]  $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 10. FPGA：SRAM 缩编（sram_check，256/128 KiB/bank） ----
run_step "FPGA" "$LOG_DIR/sram_check.log" -- bash asic/run_sram_check.sh
echo "[sram 缩编]   $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 11. ASIC：matrix 状态 SRAM 真宏接口与数值定向测试 ----
run_step "ASIC" "$LOG_DIR/matrix_sram_check.log" -- bash asic/run_matrix_sram_check.sh
echo "[matrix SRAM] $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 12. ASIC：verilator --lint-only（rtl 顶层） ----
if [ -n "$VERILATOR_BIN" ]; then
  run_step "ASIC" "$LOG_DIR/asic_lint.log" -- \
    verilator --lint-only -Wno-fatal -Wno-WIDTH -Irtl --top-module qcore_top rtl/qcore_top.sv
else
  record "ASIC" "verilator --lint-only (rtl 顶层)" "SKIP(工具缺失)" "0"
fi
echo "[asic lint]   $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 13. ASIC：综合（Yosys + sky130） ----
if [ -n "$YOSYS_BIN" ]; then
  run_step "ASIC" "$LOG_DIR/asic_synth.log" -- bash asic/run_synth.sh "$CORNER"
else
  record "ASIC" "bash asic/run_synth.sh $CORNER" "SKIP(工具缺失)" "0"
fi
echo "[asic synth]  $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 14. ASIC：OpenSTA（多 corner STA 的单 corner 探测；读 synth_datapath 网表） ----
if [ -n "$STA_BIN" ] && [ -f "$TT_LIB" ]; then
  t0=$(now_ns)
  LIB="$TT_LIB" "$STA_BIN" -no_splash -exit asic/sta.tcl >"$LOG_DIR/asic_sta.log" 2>&1
  rc=$?
  t1=$(now_ns)
  secs=$(fmt_secs "$t0" "$t1")
  if [ "$rc" -eq 0 ]; then
    record "ASIC" "OpenSTA ($STA_BIN, $CORNER)" "PASS" "$secs"
  else
    record "ASIC" "OpenSTA ($STA_BIN, $CORNER)" "FAIL" "$secs"
  fi
else
  record "ASIC" "OpenSTA ($CORNER)" "SKIP(工具缺失)" "0"
fi
echo "[asic sta]    $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 15. ASIC：MAC 综合（bf16/int8 同趟） ----
if [ -n "$YOSYS_BIN" ]; then
  run_step "ASIC" "$LOG_DIR/asic_mac_synth.log" -- bash asic/run_mac_synth.sh "$CORNER"
else
  record "ASIC" "bash asic/run_mac_synth.sh $CORNER" "SKIP(工具缺失)" "0"
fi
echo "[asic mac]    $(tail -1 "$RESULT_FILE" | cut -f3)"

# ---- 汇总 ----
echo
echo "== 验收汇总 =="
printf '%-9s | %-60s | %-16s | %s\n' "组件" "命令" "结果" "耗时(s)"
printf '%-9s-+-%-60s-+-%-16s-+-%s\n' "---------" "------------------------------------------------------------" "----------------" "--------"
while IFS=$'\t' read -r comp cmd result secs; do
  printf '%-9s | %-60s | %-16s | %s\n' "$comp" "$(trunc "$cmd" 60)" "$result" "$secs"
done < "$RESULT_FILE"

PASS_N=$(awk -F'\t' '$3=="PASS"{n++} END{print n+0}' "$RESULT_FILE")
FAIL_N=$(awk -F'\t' '$3=="FAIL"{n++} END{print n+0}' "$RESULT_FILE")
SKIP_N=$(awk -F'\t' '$3 ~ /^SKIP/{n++} END{print n+0}' "$RESULT_FILE")
echo
echo "合计: $PASS_N PASS / $FAIL_N FAIL / $SKIP_N SKIP"
echo "日志目录: $LOG_DIR"

rm -f "$RESULT_FILE"
if [ "$FAIL_N" -gt 0 ]; then
  echo "存在 FAIL → exit 1"
  exit 1
fi
exit 0
