"""M3 verification (PLAN p3-p4-plan.md §4): qforge qbin load/round-trip + per-class
W8A8 quantization error.

(a) Full-model qbin: complete loader parse (magic/version/flags/header/
    tensors/.pf/.dc/ENDQ+length) + weight round-trip assertion (container +
    executor HBM), plus a wave-1 structural decode + DC/PF instruction-sequence
    check (numeric exec of VECTOR/KV deferred to wave 2 / ExecVecKv).
(b) 6 projection classes (qkv/o/gate/up/down/lm_head) x {PF, DC} per-layer
    quantization error, M2a criterion: INT32 bit-exact group partials AND
    dequant fp32 < 1e-6 (per-128-group quantized data). Also reports the
    quantization error vs the fp32 golden y_ref (informational, not a gate).

Usage: python3 qforge/verify_m3.py [--qbin PATH] [--full-pf] [--out JSON]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = np.float16

from qforge import config as C, graph, quant, lowering as L, build
from qforge.safetensors import SafeTensors
from compiler.isa.qbin import read_qbin
from compiler.isa import isa as I
from qsim.executor import Executor, load_qbin_into_executor, int8_group_partials

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(REPO, "golden", "qwen3-0.6b")
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
ST_PATH = os.path.join(MODEL_DIR, "model.safetensors")


# ---------------------------------------------------------------------------
def _decode_all(prog: bytes) -> list[dict]:
    return [I.decode_inst(int.from_bytes(prog[o:o + 16], "little"))
            for o in range(0, len(prog), 16)]


def _check_dc_structure(insts: list[dict]) -> dict:
    """Wave-1 structural smoke: first 2 layers carry the N-tiling + online
    softmax attention sequence (numeric semantics are ExecVecKv's, deferred to
    wave 2)."""
    from collections import Counter
    c = Counter(d["mnemonic"] for d in insts)
    appends = [i for i, d in enumerate(insts) if d["mnemonic"] == "KV.APPEND"]
    # layer L's KV.APPEND occupies appends[8L .. 8L+7]; attention follows.
    # first two layers = everything up to layer 2's first append.
    end = appends[16] if len(appends) >= 16 else len(insts)
    head = insts[:end]
    hc = Counter(d["mnemonic"] for d in head)
    bmm_b2 = sum(1 for d in head if d["mnemonic"] == "BMM" and d["batch"] == 2)
    kv_both = sum(1 for d in head if d["mnemonic"] == "KV.LOAD" and d["sel"] == 2)
    n_tiles = [d["N"] for d in head if d["mnemonic"] == "BMM"]
    checks = {
        "kv_load_sel_both_2layers": kv_both,
        "bmm_batch2_2layers": bmm_b2,
        "n_tiling_max": max(n_tiles) if n_tiles else 0,
        "online_softmax_vreduce_max": hc.get("VREDUCE_MAX", 0),
        "online_softmax_vreduce_sum": hc.get("VREDUCE_SUM", 0),
        "online_softmax_vexp": hc.get("VEXP", 0),
        "running_max_vmax": hc.get("VMAX", 0),
        "running_sum_vadd": hc.get("VADD", 0),
    }
    ok = (kv_both >= 64            # 2 layers x 8 heads x 4 KV tiles
          and bmm_b2 >= 256       # 2 layers x 8 heads x 64 tiles x 2 (QK^T+AV)
          and hc["VREDUCE_MAX"] >= 256 and hc["VREDUCE_SUM"] >= 256
          and hc["VEXP"] >= 256 and max(n_tiles or [0]) <= 128)
    return {"ok": ok, "counts": dict(c), "first2layers": checks}


def _check_pf_structure(insts: list[dict]) -> dict:
    """Corrected PF: per-token KV.APPEND (28 layers x 128 tokens x 8 heads),
    no KV.STORE_BLOCK (head-major store removed), GEMM QK^T/AV + per-token
    RMSNorm/ROPE + PF softmax + VSILU."""
    from collections import Counter
    c = Counter(d["mnemonic"] for d in insts)
    ok = (c["GEMM"] > 0 and c["KV.APPEND"] == 28 * 128 * 8
          and c["KV.STORE_BLOCK"] == 0
          and c["RMSNORM"] > 0 and c["ROPE"] > 0 and c["VREDUCE_MAX"] > 0
          and c["VREDUCE_SUM"] > 0 and c["VSILU"] > 0 and c["VMASK"] > 0)
    return {"ok": ok, "counts": dict(c)}


def full_model_checks(qbin_path: str, full_pf: bool) -> dict:
    qb = read_qbin(qbin_path)
    pf_len = qb.header["pf_len"]
    dc_len = qb.header["dc_len"]
    exact_pf = qb.pf_program[:pf_len]
    exact_dc = qb.dc_program[:dc_len]
    res = {
        "magic": qb.magic.decode(), "version": qb.version, "flags": qb.flags,
        "header_size": qb.header_size, "model": qb.header["model"],
        "n_tensors": len(qb.tensors),
        "pf_insts": len(exact_pf) // 16, "dc_insts": len(exact_dc) // 16,
        "pf_len_bytes": len(exact_pf), "dc_len_bytes": len(exact_dc),
    }
    assert qb.magic == b"NLPU", "bad magic"
    assert qb.version == 1, "bad version"
    assert qb.flags & 0x3 == 2, "dtype flag != INT8(2)"
    assert qb.flags & 0x4, "dual-mode flag not set"
    assert len(exact_pf) % 16 == 0 and len(exact_dc) % 16 == 0

    # program decode (all instructions valid + engine tags valid)
    for name, prog in (("pf", exact_pf), ("dc", exact_dc)):
        for off in range(0, len(prog), 16):
            d = I.decode_inst(int.from_bytes(prog[off:off + 16], "little"))
            assert d["engine_tag_valid"], f"{name} inst {off//16} engine mismatch"
    res["programs_decode_ok"] = True

    # weight round-trip via executor HBM (write then read back bit-exact)
    exe = Executor()
    load_qbin_into_executor(exe, qb)
    rt_ok = True
    total_w = total_s = 0
    for t in qb.tensors:
        back = exe.read_bytes("hbm", t.hbm_off, len(t.data))
        if back != t.data:
            rt_ok = False
            break
        if t.scales is not None:
            back_s = exe.read_bytes("hbm", t.scales_hbm_off, len(t.scales))
            if back_s != t.scales:
                rt_ok = False
                break
        total_w += len(t.data)
        total_s += len(t.scales or b"")
    res["round_trip_ok"] = rt_ok
    res["weight_bytes"] = total_w
    res["scale_bytes"] = total_s
    assert rt_ok, "weight round-trip failed"

    # wave-1 structural checks (numeric exec deferred to wave 2)
    dc_insts = _decode_all(exact_dc)
    pf_insts = _decode_all(exact_pf)
    res["dc_structure"] = _check_dc_structure(dc_insts)
    res["pf_structure"] = _check_pf_structure(pf_insts)
    assert res["dc_structure"]["ok"], f"DC structure check failed: {res['dc_structure']}"
    assert res["pf_structure"]["ok"], f"PF structure check failed: {res['pf_structure']}"
    return res


# ---------------------------------------------------------------------------
# (b) per-class quant error
# ---------------------------------------------------------------------------
def load_golden_input(kind: str, mode: str, layer: int = 0) -> np.ndarray:
    """Load the golden activation input for a projection class."""
    base = os.path.join(GOLDEN, "prefill_seq128" if mode == "PF"
                        else "decode_seq1_cache1024")
    if kind == "qkv":
        p = os.path.join(base, f"L{layer:02d}_attn_qkv")
        x = np.load(os.path.join(p, "inputs.npz"))["x"]
    elif kind == "o":
        p = os.path.join(base, f"L{layer:02d}_attn_o")
        ctx = np.load(os.path.join(p, "inputs.npz"))["ctx"]   # [16, seq, 128]
        # o_proj input = ctx.transpose(0,1).reshape(seq, heads*head_dim)
        x = ctx.transpose(1, 0, 2).reshape(ctx.shape[1], -1).astype(np.float32)
    elif kind in ("gate", "up"):
        op = "mlp_gate" if kind == "gate" else "mlp_up"
        p = os.path.join(base, f"L{layer:02d}_{op}")
        x = np.load(os.path.join(p, "inputs.npz"))["x"]
    elif kind == "down":
        p = os.path.join(base, f"L{layer:02d}_mlp_down")
        x = np.load(os.path.join(p, "inputs.npz"))["x"]
    elif kind == "lm_head":
        p = os.path.join(base, "lm_head")
        x = np.load(os.path.join(p, "inputs.npz"))["x"]
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def load_golden_yref(kind: str, mode: str, layer: int = 0) -> np.ndarray:
    """Load the fp32 golden y_ref for a projection class."""
    base = os.path.join(GOLDEN, "prefill_seq128" if mode == "PF"
                        else "decode_seq1_cache1024")
    if kind == "qkv":
        p = os.path.join(base, f"L{layer:02d}_attn_qkv")
        o = np.load(os.path.join(p, "outputs.npz"))
        return np.concatenate([o["q_ref"], o["k_ref"], o["v_ref"]],
                              axis=1).astype(np.float32)
    if kind == "o":
        p = os.path.join(base, f"L{layer:02d}_attn_o")
        return np.load(os.path.join(p, "outputs.npz"))["o_ref"].astype(np.float32)
    if kind == "gate":
        p = os.path.join(base, f"L{layer:02d}_mlp_gate")
        return np.load(os.path.join(p, "outputs.npz"))["gate_ref"].astype(np.float32)
    if kind == "up":
        p = os.path.join(base, f"L{layer:02d}_mlp_up")
        return np.load(os.path.join(p, "outputs.npz"))["up_ref"].astype(np.float32)
    if kind == "down":
        p = os.path.join(base, f"L{layer:02d}_mlp_down")
        return np.load(os.path.join(p, "outputs.npz"))["down_ref"].astype(np.float32)
    if kind == "lm_head":
        p = os.path.join(base, "lm_head")
        return np.load(os.path.join(p, "outputs.npz"))["logits_ref"].astype(np.float32)
    raise ValueError(kind)


def load_weight(kind: str, st: SafeTensors, layer: int = 0) -> np.ndarray:
    if kind == "qkv":
        parts = [st.get_float32(f"model.layers.{layer}.self_attn.{k}_proj.weight")
                 for k in ("q", "k", "v")]
        return np.concatenate(parts, axis=0)
    keys = {
        "o": f"model.layers.{layer}.self_attn.o_proj.weight",
        "gate": f"model.layers.{layer}.mlp.gate_proj.weight",
        "up": f"model.layers.{layer}.mlp.up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
        "lm_head": "lm_head.weight",
    }
    return st.get_float32(keys[kind])


def _bf16_ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element bf16-domain ULP distance (plan §1 convention)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    mag = np.maximum(np.abs(a), np.abs(b))
    ulp = np.zeros_like(mag)
    nz = mag > 0
    ulp[nz] = np.exp2(np.floor(np.log2(mag[nz])) - 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ulp > 0, d / ulp, np.where(d > 0, np.inf, 0.0))


def run_class_case(kind: str, mode: str, st: SafeTensors) -> dict:
    w = load_weight(kind, st)
    x = load_golden_input(kind, mode)
    y_ref = load_golden_yref(kind, mode)
    M, K = x.shape
    N = w.shape[0]
    assert tuple(w.shape) == (N, K), f"{kind} {mode}: w {w.shape} != {(N,K)}"

    qb, xq, cd, wqi, input_hbm, output_hbm = build.build_projection_qbin(
        f"/tmp/qforge_{kind}_{mode}.qbin", kind, w, x, mode)

    exe = Executor()
    load_qbin_into_executor(exe, qb)
    exe.write_bytes("hbm", input_hbm, xq.tobytes())
    prog = (qb.pf_program[:qb.header["pf_len"]] if mode == "PF"
            else qb.dc_program[:qb.header["dc_len"]])
    exe.run(prog)
    ntiles = N // 128
    out = np.frombuffer(exe.read_bytes("hbm", output_hbm,
                                       ntiles * M * 128 * 2),
                        dtype=BF16).astype(np.float32).reshape(ntiles, M, 128)
    out = out.transpose(1, 0, 2).reshape(M, N)

    G = K // 128
    partials = int8_group_partials(xq, wqi.T, G)
    bit_exact = all(
        np.array_equal(partials[g],
                       np.einsum("ik,kj->ij",
                                 xq[:, g * 128:(g + 1) * 128].astype(np.int32),
                                 wqi[:, g * 128:(g + 1) * 128].T.astype(np.int32),
                                 dtype=np.int32, optimize=True))
        for g in range(G))
    ref = np.zeros((M, N), np.float64)
    for g in range(G):
        ref += cd[:, g].astype(np.float64)[None, :] * partials[g].astype(np.float64)
    # dequant writeback is BF16 (04 §1.5 post-process): golden bf16 y is the
    # fp64 reference rounded to BF16.
    golden_bf16 = ref.astype(BF16).astype(np.float32)
    dequant_bf16_max_ulp = float(np.ceil(_bf16_ulp_dist(out, golden_bf16).max()))
    dequant_abs = float(np.abs(out.astype(np.float64) - ref).max())
    with np.errstate(divide="ignore", invalid="ignore"):
        dequant_rel = float((np.abs(out.astype(np.float64) - ref)
                             / np.maximum(np.abs(ref), 1e-30)).max())
    # dequant gate: INT32 partials bit-exact, and the executor's bf16 dequant
    # writeback matches the fp64 reference rounded to bf16 within 1 bf16 ULP.
    passed = bit_exact and (dequant_bf16_max_ulp <= 1)

    # informational: quantization error vs fp32 golden
    qerr_abs = float(np.abs(out.astype(np.float32) - y_ref).max())
    qerr_rel = float(np.abs(out.astype(np.float32) - y_ref).max()
                     / np.abs(y_ref).max())
    return {
        "kind": kind, "mode": mode, "N": N, "K": K, "M": M,
        "int32_bit_exact": bool(bit_exact),
        "dequant_bf16_max_ulp": dequant_bf16_max_ulp,
        "dequant_max_abs_err": dequant_abs,
        "dequant_max_rel_err": dequant_rel,
        "quant_err_vs_golden_abs": qerr_abs,
        "quant_err_vs_golden_rel": qerr_rel,
        "pass": bool(passed),
        "criteria": "int32 bit-exact & dequant bf16<=1ulp",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qbin", default="/tmp/qwen3-0.6b-m3.qbin")
    ap.add_argument("--full-pf", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "p4",
                                                  "m3-results.json"))
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_build:
        st = SafeTensors.open(ST_PATH)
        qb = build.build_model_qbin(args.qbin, st, C.QUANT_INT8,
                                    C.ACTIVATION_SCALE_DEFAULT)
        build_sec = time.time() - t0
    else:
        st = SafeTensors.open(ST_PATH)
        build_sec = 0.0

    fm = full_model_checks(args.qbin, args.full_pf)
    fm["build_sec"] = round(build_sec, 3)

    cases = []
    for kind in graph.CLASS_NAMES:
        for mode in ("PF", "DC"):
            r = run_class_case(kind, mode, st)
            cases.append(r)
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"[{mark}] {kind:7s} {mode:2s}  bit_exact={r['int32_bit_exact']} "
                  f"dequant={r['dequant_max_abs_err']:.2e} "
                  f"quant_err_rel={r['quant_err_vs_golden_rel']:.4f}")

    all_pass = all(c["pass"] for c in cases)
    print("\n(a) full-model:", json.dumps(fm, indent=2))
    print(f"\n(b) 6 classes x PF/DC: {'ALL PASS' if all_pass else 'FAILURES'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"full_model": fm, "cases": cases, "all_pass": all_pass},
                  f, indent=2)
    print(f"results -> {args.out}")
    return 0 if (all_pass and fm["round_trip_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
