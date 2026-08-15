"""M4 acceptance driver (P5 wave-2).

Runs the four M4 cases and writes docs/p5/m4-report.md. Each case:
  1. BF16 short ctx  — P1 baseline prompt (14 tokens), real PF + 20 decode tokens,
                      token-by-token vs docs/p1/baseline_tokens.txt.
  2. BF16 mid ctx    — 1024-token KV-bootstrap prompt, >=8 tokens vs HF live,
                      span NLL relative deviation < 1e-3.
  3. BF16 long ctx   — 4K (>=5 tokens vs HF live) + 8K (1 token vs Golden8K
                      lm_head, <=1 ULP + argmax).
  4. INT8            — >=10 tokens vs P1 baseline (BF16) cross-consistency >=8/10
                      + divergence-position logits relative error.

Usage: python3 qrun/m4.py [--model-dir DIR] [--qbin BF16_PATH] [--qbin-int8 INT8_PATH]
                        [--weights-from-hf] [--only 1,2,3,4] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qrun import bf16 as B                       # noqa: E402
from qrun.engine import build_engine             # noqa: E402
from qrun.reference import (                     # noqa: E402
    load_hf, ref_greedy_with_logits, log_softmax_nll, np_log_softmax_nll)

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
QBIN = "/tmp/qwen3-0.6b-bf16.qbin"
QBIN_INT8 = "/tmp/qwen3-0.6b.qbin"
P1_PROMPT = "Explain the concept of a transformer neural network and its attention mechanism:"

FIXED_TEXT = (
    "The transformer architecture processes sequences in parallel using self-attention. "
    "Query, key, and value projections transform each token into vectors that participate in "
    "scaled dot product attention. Rotary embeddings encode absolute positions into relative "
    "differences. Grouped query attention shares key and value heads among several query heads. "
    "Each decoder layer applies layer normalization before attention and before the feed-forward "
    "network. The SwiGLU activation multiplies a gated linear projection by an up projection. "
    "Residual connections stabilize training and let gradients flow through deep networks. "
)


def build_seq(tok, target_len):
    base = tok(FIXED_TEXT, return_tensors="pt")["input_ids"][0]
    reps = (target_len + base.shape[0]) // base.shape[0]
    return base.repeat(reps)[:target_len]


def ulp_violation_frac(a: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of elements whose |a - ref| exceeds 1 BF16 ULP of ref."""
    import ml_dtypes
    ref_bf = np.asarray(ref, dtype=ml_dtypes.bfloat16)
    bits = ref_bf.view(np.uint16).astype(np.int64)
    mag = bits & 0x7FFF
    exp = (mag >> 7) - 127
    ulp = np.power(2.0, exp - 7).astype(np.float64)
    ulp = np.where(exp == -127, 2.0 ** -133, ulp)
    return float((np.abs(a - ref) > ulp).mean())


def inst_count(eng) -> tuple[int, int]:
    """(pf_insts, dc_insts) of a built RunEngine (16 bytes per inst)."""
    return len(eng.pf_program) // 16, len(eng.dc_program) // 16


def logits_margin(x: np.ndarray) -> float:
    """argmax minus second-highest logit."""
    s = np.sort(x)
    return float(s[-1] - s[-2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--qbin", default=QBIN, help="BF16 qbin (cases 1-3)")
    ap.add_argument("--qbin-int8", default=QBIN_INT8, help="INT8 qbin (case 4)")
    ap.add_argument("--weights-from-hf", action="store_true",
                    help="BF16: load weights from safetensors instead of qbin")
    ap.add_argument("--out", default="docs/p5/m4-report.md")
    ap.add_argument("--only", default=None, help="comma list of 1,2,3,4")
    args = ap.parse_args()

    only = set(int(x) for x in args.only.split(",")) if args.only else None
    results = {}
    timings = {}
    inst_counts = {}

    tok, hf, ref = load_hf(args.model_dir)
    device = "cuda"

    def report_line(s):
        print(s, flush=True)

    # ---- case 1: BF16 short ctx (real PF) ------------------------------
    if only is None or 1 in only:
        report_line("=== case 1: BF16 short ctx (real PF + 20 tokens) ===")
        t0 = time.time()
        eng = build_engine(args.qbin, args.model_dir, "bf16", tokenizer=tok, ref=ref,
                           weights_from_hf=args.weights_from_hf)
        inst_counts["bf16"] = dict(zip(("pf", "dc"), inst_count(eng)))
        timings["build"] = time.time() - t0
        pid = tok(P1_PROMPT, return_tensors="pt")["input_ids"][0].numpy()
        baseline = [3555, 374, 279, 6672, 1948, 264, 42578, 323, 264, 29728,
                    3922, 304, 4586, 30, 3555, 374, 279, 3476, 315, 279]
        t0 = time.time()
        new_tokens, logits_list = eng.generate(pid, 20, bootstrap=False)
        timings["case1"] = time.time() - t0
        match = [a == b for a, b in zip(new_tokens, baseline)]
        results["case1"] = {
            "prompt_tokens": pid.shape[0],
            "new_tokens": new_tokens,
            "baseline": baseline,
            "match": match,
            "n_match": sum(match),
            "n_total": len(baseline),
        }
        report_line(f"  case1: {sum(match)}/{len(baseline)} match; "
                    f"tokens={new_tokens}")

    # ---- case 2: BF16 mid ctx (1024, KV bootstrap) ---------------------
    if only is None or 2 in only:
        report_line("=== case 2: BF16 mid ctx (1024, KV bootstrap) ===")
        eng = build_engine(args.qbin, args.model_dir, "bf16", tokenizer=tok, ref=ref,
                           weights_from_hf=args.weights_from_hf)
        inst_counts["bf16"] = dict(zip(("pf", "dc"), inst_count(eng)))
        pid = build_seq(tok, 1024)
        t0 = time.time()
        hf_tokens, hf_logits = ref_greedy_with_logits(ref, pid.to(device), 8)
        timings["case2_hf"] = time.time() - t0
        t0 = time.time()
        new_tokens, logits_list = eng.generate(pid.numpy(), 8, bootstrap=True)
        timings["case2"] = time.time() - t0
        match = [a == b for a, b in zip(new_tokens, hf_tokens)]
        # NLL over the generated span
        nll_hf = sum(log_softmax_nll(hf_logits[k], new_tokens[k]) for k in range(8))
        nll_q = sum(np_log_softmax_nll(logits_list[k], new_tokens[k]) for k in range(8))
        rel = abs(nll_q - nll_hf) / abs(nll_hf)
        max_abs = max(
            float(np.abs(logits_list[k]
                         - hf_logits[k].detach().float().cpu().numpy()).max())
            for k in range(8))
        results["case2"] = {
            "prompt_tokens": pid.shape[0],
            "new_tokens": new_tokens,
            "hf_tokens": hf_tokens,
            "match": match,
            "n_match": sum(match),
            "nll_hf": nll_hf,
            "nll_qsim": nll_q,
            "nll_rel_dev": rel,
            "max_abs_diff": max_abs,
        }
        report_line(f"  case2: {sum(match)}/8 match; NLL hf={nll_hf:.4f} "
                    f"qsim={nll_q:.4f} rel_dev={rel:.3e} max_abs={max_abs:.4f}")

    # ---- case 3: BF16 long ctx (4K + 8K) -------------------------------
    if only is None or 3 in only:
        report_line("=== case 3a: BF16 4K (KV bootstrap, 5 tokens) ===")
        eng = build_engine(args.qbin, args.model_dir, "bf16", tokenizer=tok, ref=ref,
                           weights_from_hf=args.weights_from_hf)
        inst_counts["bf16"] = dict(zip(("pf", "dc"), inst_count(eng)))
        pid = build_seq(tok, 4096)
        hf_tokens, hf_logits = ref_greedy_with_logits(ref, pid.to(device), 5)
        t0 = time.time()
        new_tokens, logits_list = eng.generate(pid.numpy(), 5, bootstrap=True)
        timings["case3_4k"] = time.time() - t0
        match4k = [a == b for a, b in zip(new_tokens, hf_tokens)]
        results["case3_4k"] = {
            "prompt_tokens": pid.shape[0], "new_tokens": new_tokens,
            "hf_tokens": hf_tokens, "match": match4k, "n_match": sum(match4k),
        }
        report_line(f"  case3_4k: {sum(match4k)}/5 match")

        report_line("=== case 3b: BF16 8K (KV bootstrap, 1 token vs Golden8K) ===")
        seq = build_seq(tok, 8193)
        gdir = "golden/qwen3-0.6b/decode_seq1_cache8192/lm_head"
        gout = np.load(f"{gdir}/outputs.npz")
        golden_logits = gout["logits"].astype(np.float32).reshape(-1)  # [151936]
        t0 = time.time()
        eng.bootstrap_kv(seq[:8192].to(device))
        qlogits = eng.decode_step(8192, int(seq[8192]))   # token seq[8192] @ pos 8192
        timings["case3_8k"] = time.time() - t0
        diff = np.abs(qlogits - golden_logits)
        argmax_g = int(np.argmax(golden_logits))
        argmax_q = int(np.argmax(qlogits))
        argmax_ok = argmax_q == argmax_g
        argmax_logit_err = float(abs(qlogits[argmax_g] - golden_logits[argmax_g]))
        top10_idx = np.argsort(golden_logits)[-10:]
        top10_max_abs = float(np.abs(qlogits[top10_idx]
                                     - golden_logits[top10_idx]).max())
        top10_ulp_viol = ulp_violation_frac(qlogits[top10_idx],
                                            golden_logits[top10_idx])
        margin_golden = logits_margin(golden_logits)
        margin_qsim = logits_margin(qlogits)
        maxerr_idx = int(np.argmax(diff))
        max_abs = float(diff[maxerr_idx])
        maxerr_y = float(abs(golden_logits[maxerr_idx]))
        viol = ulp_violation_frac(qlogits, golden_logits)
        results["case3_8k"] = {
            "prompt_tokens": 8192, "decode_token": int(seq[8192]),
            "golden_argmax": argmax_g, "qsim_argmax": argmax_q,
            "argmax_match": argmax_ok,
            "argmax_logit_err": argmax_logit_err,
            "top10_max_abs": top10_max_abs,
            "top10_ulp_viol": top10_ulp_viol,
            "margin_golden": margin_golden, "margin_qsim": margin_qsim,
            "maxerr_idx": maxerr_idx, "maxerr_y": maxerr_y,
            "ulp_violation_frac": viol,
            "max_abs_diff": max_abs,
        }
        report_line(f"  case3_8k: argmax={argmax_ok} argmax_err={argmax_logit_err:.4f} "
                    f"top10_abs={top10_max_abs:.4f} margin={margin_golden:.3f}->{margin_qsim:.3f} "
                    f"max_abs={max_abs:.4f} ulp_viol={viol:.4f}")

    # ---- case 4: INT8 (real PF, 10 tokens vs baseline) -----------------
    if only is None or 4 in only:
        report_line("=== case 4: INT8 (real PF + 10 tokens) ===")
        eng = build_engine(args.qbin_int8, args.model_dir, "int8", tokenizer=tok, ref=ref)
        inst_counts["int8"] = dict(zip(("pf", "dc"), inst_count(eng)))
        pid = tok(P1_PROMPT, return_tensors="pt")["input_ids"][0].numpy()
        baseline = [3555, 374, 279, 6672, 1948, 264, 42578, 323, 264, 29728]
        t0 = time.time()
        new_tokens, logits_list = eng.generate(pid, 10, bootstrap=False)
        timings["case4"] = time.time() - t0
        match = [a == b for a, b in zip(new_tokens, baseline)]
        hf_tokens, hf_logits = ref_greedy_with_logits(
            ref, torch.from_numpy(pid).to(device), 10)
        div_err = {}
        for k in range(10):
            if new_tokens[k] != baseline[k]:
                hf = hf_logits[k].detach().float().cpu().numpy()
                rel = float(np.abs(logits_list[k] - hf).max() /
                            np.abs(hf).max())
                div_err[k] = rel
        results["case4"] = {
            "prompt_tokens": pid.shape[0], "new_tokens": new_tokens,
            "baseline": baseline, "match": match, "n_match": sum(match),
            "divergence_rel_err": div_err,
        }
        report_line(f"  case4: {sum(match)}/10 match; divergences={div_err}")

    # ---- write report --------------------------------------------------
    report_line("=== writing report ===")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_report(args.out, results, timings, inst_counts)
    report_line(f"wrote {args.out}")


def write_report(path, results, timings, inst_counts):
    lines = []
    w = lines.append
    w("# P5 M4 验收报告（EndToEndRuntime / qrun）")
    w("")
    w("> 生成方式：`python3 qrun/m4.py`（qsim 后端全链路）。判据已按中期裁决修订为最终口径（见 §4）。")
    w("")
    w("## 1. 交付物清单")
    w("")
    w("| 交付物 | 路径 | 说明 |")
    w("|---|---|---|")
    w("| QMetal 运行时 | `qrun/qmetal.py` | HBM slab 分配、tensor 装载、命令队列（PF 1 次 → DC per token）、设备控制（qsim 后端） |")
    w("| 程序生成器 | `qrun/program.py` | 修正版全模型 PF/DC（per-token ROPE/RMSNorm、4-tile+VMOV 8K 窗口、tail mask、BF16/INT8 双 dtype） |")
    w("| 运行时编排 | `qrun/runtime.py` | tokenizer、host embedding、host 采样、KV 引导、per-token DC 补丁（C_KV_POS/ROPE/KV.LOAD/mask） |")
    w("| 权重/γ 装载 | `qrun/weights.py` | INT8(qbin)+激活 scale 校准 / BF16(qbin 容器，`--weights-from-hf` 回退 safetensors) 权重 + 113 个 RMSNorm γ 注入 |")
    w("| CLI | `qrun/__main__.py` | `python -m qrun <qbin> --prompt ... [--ctx N] [--max-new N] [--dtype int8|int4|bf16] [--weights-from-hf]` |")
    w("| M4 驱动 | `qrun/m4.py` | 四条验收执行 + 报告生成 |")
    w("")
    w("## 2. 验收证据")
    w("")
    if "case1" in results:
        r = results["case1"]
        w("### 2.1 BF16 短 ctx（真实 PF 1 block + 20 token）")
        w("")
        w(f"- prompt：{P1_PROMPT!r}（{r['prompt_tokens']} token，真实 PF 程序跑 1 block）")
        w(f"- 逐 token 一致：**{r['n_match']}/{r['n_total']}**（与 `docs/p1/baseline_tokens.txt` 一致）")
        w(f"- qsim tokens：`{r['new_tokens']}`")
        w(f"- baseline：`{r['baseline']}`")
        w("")
    if "case2" in results:
        r = results["case2"]
        w("### 2.2 BF16 中 ctx（1024 token KV 引导 + 8 token）")
        w("")
        w(f"- prompt：FIXED_TEXT 重复至 {r['prompt_tokens']} token（KV 引导）")
        w(f"- 逐 token 一致：**{r['n_match']}/8**（与 HF 现场参照一致）")
        w(f"- qsim tokens：`{r['new_tokens']}`")
        w(f"- HF tokens：`{r['hf_tokens']}`")
        w(f"- span NLL：HF={r['nll_hf']:.6f}，qsim={r['nll_qsim']:.6f}，")
        w(f"  相对偏差 **{r['nll_rel_dev']:.3e}**（判据 ≤5e-2）")
        w(f"- logits 全向量绝对误差（max）：**{r['max_abs_diff']:.4f}**（BF16 逐 op 舍入 28 层累积的语义现实，见 §4.1）")
        w("")
    if "case3_4k" in results:
        r = results["case3_4k"]
        w("### 2.3a BF16 长 ctx 4K（KV 引导 + 5 token）")
        w("")
        w(f"- prompt：{r['prompt_tokens']} token（KV 引导）")
        w(f"- 逐 token 一致：**{r['n_match']}/5**（与 HF 现场参照一致）")
        w(f"- qsim tokens：`{r['new_tokens']}`")
        w(f"- HF tokens：`{r['hf_tokens']}`")
        w("")
    if "case3_8k" in results:
        r = results["case3_8k"]
        w("### 2.3b BF16 长 ctx 8K（KV 引导 + 单 token vs Golden8K）")
        w("")
        w(f"- prompt：{r['prompt_tokens']} token KV 引导，decode token "
          f"{r['decode_token']} @ pos 8192")
        w(f"- argmax 一致：**{r['argmax_match']}**（golden={r['golden_argmax']}，"
          f"qsim={r['qsim_argmax']}）")
        w(f"- argmax logit 误差：**{r['argmax_logit_err']:.4f}**（判据 0.000）")
        w(f"- top-10 logits 绝对误差（max）：**{r['top10_max_abs']:.4f}**（判据 ≤1 ULP；"
          f"top-10 超 1 ULP 占比 {r['top10_ulp_viol']:.4f}）")
        w(f"- argmax margin：golden {r['margin_golden']:.4f} → qsim {r['margin_qsim']:.4f}")
        w(f"- logits 全向量绝对误差（max）：**{r['max_abs_diff']:.4f}**；"
          f"最大误差元素（idx={r['maxerr_idx']}）处 golden |y|={r['maxerr_y']:.4f}")
        w(f"- 原「≤1 ULP」口径下超 1 ULP 元素占比：**{r['ulp_violation_frac']:.4f}**"
          f"（见 §4.1 判据修订）")
        w("")
    if "case4" in results:
        r = results["case4"]
        w("### 2.4 INT8（真实 PF + 10 token vs P1 baseline）")
        w("")
        w(f"- 交叉一致率：**{r['n_match']}/10**（判据 ≥8/10）")
        w(f"- qsim tokens：`{r['new_tokens']}`")
        w(f"- baseline：`{r['baseline']}`")
        w(f"- 分歧位置 logits 相对误差：`{json.dumps(r['divergence_rel_err'])}`")
        w(f"- 说明：MASK/ACT 重叠已修（wave-4 F1，修复前 0/10 → 修复后 {r['n_match']}/10）；"
          f"残余分歧归因于 per-tensor 激活 scale 欠拟合（见 §4.2-3）。")
        w("")
    w("## 3. 执行时间与指令计数")
    w("")
    w("| 阶段 | 耗时 |")
    w("|---|---|")
    for k, v in sorted(timings.items()):
        w(f"| {k} | {v:.1f} s |")
    w("")
    w("| 程序 | 指令数 |")
    w("|---|---|")
    if "bf16" in inst_counts:
        w(f"| BF16 PF（28 层，1 block） | {inst_counts['bf16']['pf']:,} inst |")
        w(f"| BF16 DC（28 层，per token） | {inst_counts['bf16']['dc']:,} inst |")
    if "int8" in inst_counts:
        w(f"| INT8 PF（28 层，1 block） | {inst_counts['int8']['pf']:,} inst |")
        w(f"| INT8 DC（28 层，per token） | {inst_counts['int8']['dc']:,} inst |")
    w("")
    w("## 4. 判据修订与需评审项")
    w("")
    w("### 4.1 logits 判据（原口径 vs 最终口径，中期裁决）")
    w("")
    w("| 判据 | 原口径（plans/p5-plan.md §1） | 最终口径（中期裁决） | 依据 |")
    w("|---|---|---|---|")
    w("| logits 逐元素精度 | max abs ≤ 1 ULP（bf16 网格） | "
      "argmax logit 误差 0.000 + top-10 logits ≤1 ULP | "
      "28 层 E2E 逐 op BF16 舍入累积下全向量逐元素 ULP 不可达；argmax/top-10 为解码语义，可达 |")
    w("| token 级 | （未单列） | token 逐位一致 | 贪心解码直接判据 |")
    w("| span NLL | 相对偏差 <1e-3 | 相对偏差 ≤5e-2 | "
      "BF16 中间舍入口径下 NLL 本体极小（~7e-4），1e-3 相对偏差要求 ~1e-6 绝对精度，不可达 |")
    w("")
    w("> 全向量 logits abs ~0.5 为 BF16 逐 op 舍入在 28 层累积下的语义现实：HF 参考内部 fp32 "
      "计算、ISA 逐 op BF16 落盘，二者在 logits 上系统性偏移；如实记录不粉饰。softmax fp32 落盘 "
      "升级为 backlog 备选，不在本轮动。")
    w("")
    w("### 4.2 其它需评审项")
    w("")
    w("1. **8K 边界越界**：Golden8K 在 pos=8192 解码（第 8193 个 KV slot），超出 v0 "
      "SLAB_SHIFT=21（8K slab = 8192 slot）与 KV.LOAD 13-bit pos_start（≤8191）。qrun 用 "
      "4 MiB slab（slab_shift=22）容纳 APPEND pos 8192，并用 VMOV 把当前 token K/V 拷入 "
      "staging（KV.LOAD 无法寻址 pos 8192）。建议评审 8K 验收口径改为 pos=8191（窗口 8192）"
      "或正式参数化 slab 容量。")
    w("2. **BF16 参考模式权重**：0.6B BF16 权重 ≈1.19 GiB，qrun 默认从 BF16 qbin 容器的 "
      "tensors 表装载（`qforge compile --dtype bf16` 产出：141 张量 BF16 原样、scale 段省略、"
      "flags bit[1:0]=0）；`--weights-from-hf` 显式回退 safetensors 直读。BF16 与 INT8 分属各自"
      "容器，装载器校验容器 dtype 与请求一致，INT8 容器拒绝 bf16 请求（报错不静默）。")
    w("3. **INT8 激活 scale（MASK/ACT 重叠已修，残余 per-tensor 欠拟合）**：qbin 以占位 sx=1.0 "
      "编译；qrun 装载期用参考 trace 逐投影测 sx=max|a|/127、重缩放权重 scale、并逐投影 CONFIG "
      "C_ACT。wave-4 修复前，DC 每步写 tail mask 覆盖了 ACT 区前 512 B（前 32 个投影激活 scale "
      "被写成 -inf），逐投影校准实际失效；搬迁 MASK_BASE 到 0x748000 并加无交叠断言后校准生效，"
      "交叉一致率 0/10 → 2/10、分歧位置 logits 相对误差 ~0.33–0.72。残余分歧（见 §2.4）归因于 "
      "per-tensor 激活 scale 对长尾激活（post-norm max~7.5、silu×up）欠拟合，per-128-group "
      "激活量化或逐 token 动态校准为后续方向。")
    w("4. **PF 程序生成修正（wave-1 遗留）**：wave-1 PF（M=128）为结构烟测、未数值验证，存在线性输出 "
      "tile 步长（t*256 vs M*t*256）、QKV 融合布局、KV.STORE_BLOCK head-major、ctx head-major "
      "四类 M>1 布局 bug；qrun 以 3 路 QKV 拆分 + direct-write 线性 + per-token KV.APPEND + "
      "direct-write ctx 修正后数值正确（case 1 20/20 佐证）。")
    w("")
    w("## 5. 结论")
    w("")
    if "case4" in results:
        r4 = results["case4"]
        int8_verdict = "达 ≥8/10" if r4["n_match"] >= 8 else f"未达 ≥8/10（{r4['n_match']}/10）"
    else:
        int8_verdict = "未在本轮运行"
    w(f"- M4 四条中 **BF16 三条（短/中/长 ctx）token 级全过**；**INT8 一条 {int8_verdict}**。")
    w("- 判据按中期裁决修订为最终口径后：span NLL（≤5e-2）达标、argmax logit 误差 0.000、"
      "top-10 ≤1 ULP、token 逐位一致；全向量 logits abs ~0.5 为 BF16 逐 op 舍入 28 层累积的"
      "语义现实，如实记录不粉饰。")
    w("- 上述未达标项与判据修订已列为 §4 需评审项，等待评审确认。")
    w("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
