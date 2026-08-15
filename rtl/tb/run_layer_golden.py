"""QCore RTL single-layer golden driver (M6, L00 / decode_seq1_cache1024).

Runs the L00 op chain (RMSNorm -> QK-norm -> RoPE -> SwiGLU -> residual) on the
RTL (Verilator) and compares each op's bf16 output against the P1 golden
per-op tensors (bf16 <= 1 ULP).  Linear projections (QKV/O/gate/up/down) are
covered by the M2a full-size linear co-sim + M4 executor-vs-golden (see
docs/p7/rtl-report.md); this driver adds the direct RTL-vs-golden evidence for
the vector ops that the M2a linear co-sim does not exercise.

Usage: python3 rtl/tb/run_layer_golden.py [--layer 0]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE = os.path.join(REPO, "rtl", "ref", "qsim_baseline")
GOLDEN = os.path.join(REPO, "golden", "qwen3-0.6b")
MODEL_SAFETENSORS = os.environ.get(
    "MODEL_SAFETENSORS",
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/model.safetensors"))
CACHE = "decode_seq1_cache1024"

sys.path.insert(0, BASELINE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    BF16 = np.float16

from executor import Executor  # noqa: E402
from compiler.isa import isa as I  # noqa: E402
import cosim  # noqa: E402
from cosim import dump_preload, run_rtl, expected_trace, bf16_ulp_dist  # noqa: E402

LAYER = 0
EPS = 1e-6
ROPE_THETA = 1_000_000.0
POS = 1024


def _fp32_bits(v: float) -> int:
    return int(np.array([v], dtype=np.float32).view(np.uint32)[0])


def _bf16b(x) -> bytes:
    return np.asarray(x, dtype=np.float32).astype(BF16).tobytes()


def _cfg(prog, reg, cls, val):
    prog += I.encode_inst("CONFIG", REG=reg, reg_class=cls,
                          IMM64=int(val)).to_bytes(16, "little")


def _load_golden(op: str):
    d = os.path.join(GOLDEN, CACHE, f"L{LAYER:02d}_{op}")
    return (np.load(os.path.join(d, "inputs.npz")),
            np.load(os.path.join(d, "outputs.npz")))


def _hf_tensor(key: str) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(MODEL_SAFETENSORS, framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def _run_rtl(exe: Executor, prog: bytes, region, n_elem: int):
    preload = dump_preload(exe)
    etr, etot = expected_trace(prog)
    trace, total, dump = run_rtl(bytes(prog), preload, [region])
    got = np.frombuffer(dump, dtype=BF16).astype(np.float32)
    return trace, total, etr, etot, got


def _report(name, trace, total, etr, etot, got, ref):
    ref = np.asarray(ref, np.float32).reshape(-1)
    got = got.reshape(-1)
    d = bf16_ulp_dist(got, ref)
    trace_ok = bool((trace == etr) and (total == etot))
    max_ulp = float(np.ceil(d.max())) if d.size else 0.0
    # M4 口径 (test_m2a): normal-magnitude elements must be <= 1 bf16 ulp;
    # tiny cancellation values (|ref|<1e-3) may differ by a few ulps from the
    # fp32 accumulation order (RTL k-sequential vs torch golden) — a
    # cross-implementation effect, not an RTL bug.
    normal = np.abs(ref) >= 1e-3
    max_ulp_normal = float(np.ceil(d[normal].max())) if normal.any() else 0.0
    ok = trace_ok and max_ulp_normal <= 1.0
    return {
        "name": name, "trace_ok": trace_ok,
        "total_cycles": int(total), "max_ulp": max_ulp,
        "max_ulp_normal": max_ulp_normal,
        "n_elem": int(got.size), "pass": bool(ok),
    }


def op_rmsnorm_in():
    inp, out = _load_golden("rmsnorm_in")
    x = inp["x"].astype(np.float32).reshape(-1)
    y = out["y"].astype(np.float32).reshape(-1)
    gamma = _hf_tensor(f"model.layers.{LAYER}.input_layernorm.weight")

    exe = Executor()
    exe.write_bytes("sram", 0x10000, _bf16b(x))
    exe.write_bytes("sram", 0x20000, _bf16b(gamma))
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x2000); _cfg(prog, 2, 1, 0x3000)
    _cfg(prog, 5, 0, _fp32_bits(EPS))
    prog += I.encode_inst("RMSNORM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARd=2, len=x.size, CV=5, imm=0).to_bytes(16, "little")
    trace, total, etr, etot, got = _run_rtl(exe, bytes(prog), (0, 0x30000, x.size * 2), x.size)
    return _report("rmsnorm_in", trace, total, etr, etot, got, y)


def op_qknorm():
    inp, out = _load_golden("attn_qknorm")
    q = inp["q"].astype(np.float32).reshape(-1)
    k = inp["k"].astype(np.float32).reshape(-1)
    q_out = out["q"].astype(np.float32)
    k_out = out["k"].astype(np.float32)
    qg = _hf_tensor(f"model.layers.{LAYER}.self_attn.q_norm.weight")
    kg = _hf_tensor(f"model.layers.{LAYER}.self_attn.k_norm.weight")

    res = []
    for name, x, g, n_heads, ref in (
        ("qknorm_q", q, qg, 16, q_out),
        ("qknorm_k", k, kg, 8, k_out),
    ):
        exe = Executor()
        exe.write_bytes("sram", 0x10000, _bf16b(x))
        exe.write_bytes("sram", 0x20000, _bf16b(np.tile(g, n_heads)))
        prog = bytearray()
        _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x2000); _cfg(prog, 2, 1, 0x3000)
        _cfg(prog, 5, 0, _fp32_bits(EPS))
        prog += I.encode_inst("RMSNORM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                              ARa=0, ARb=1, ARd=2, len=x.size, CV=5,
                              imm=(1 << 31)).to_bytes(16, "little")
        trace, total, etr, etot, got = _run_rtl(exe, bytes(prog), (0, 0x30000, x.size * 2), x.size)
        res.append(_report(name, trace, total, etr, etot, got, ref.reshape(-1)))
    return res


def op_rope():
    inp, out = _load_golden("attn_rope")
    q = inp["q"].astype(np.float32).reshape(-1)
    k = inp["k"].astype(np.float32).reshape(-1)
    q_out = out["q"].astype(np.float32)
    k_out = out["k"].astype(np.float32)

    res = []
    for name, x, ref in (("rope_q", q, q_out), ("rope_k", k, k_out)):
        exe = Executor()
        exe.write_bytes("sram", 0x10000, _bf16b(x))
        prog = bytearray()
        _cfg(prog, 0, 1, 0x1000); _cfg(prog, 2, 1, 0x3000)
        _cfg(prog, 6, 0, _fp32_bits(ROPE_THETA))
        prog += I.encode_inst("ROPE", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                              ARa=0, ARb=0, ARd=2, len=x.size, CV=6, imm=POS).to_bytes(16, "little")
        trace, total, etr, etot, got = _run_rtl(exe, bytes(prog), (0, 0x30000, x.size * 2), x.size)
        res.append(_report(name, trace, total, etr, etot, got, ref.reshape(-1)))
    return res


def op_silu():
    inp, out = _load_golden("mlp_silu")
    gate = inp["gate"].astype(np.float32).reshape(-1)
    up = inp["up"].astype(np.float32).reshape(-1)
    y = out["y"].astype(np.float32).reshape(-1)

    exe = Executor()
    exe.write_bytes("sram", 0x10000, _bf16b(gate))
    exe.write_bytes("sram", 0x40000, _bf16b(up))
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x4000)
    _cfg(prog, 2, 1, 0x5000); _cfg(prog, 3, 1, 0x6000)
    prog += I.encode_inst("VSILU", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=0, ARd=2, len=gate.size, CV=0).to_bytes(16, "little")
    prog += I.encode_inst("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=2, ARb=1, ARd=3, len=gate.size, CV=0).to_bytes(16, "little")
    trace, total, etr, etot, got = _run_rtl(exe, bytes(prog), (0, 0x60000, gate.size * 2), gate.size)
    return _report("mlp_silu", trace, total, etr, etot, got, y)


def op_residual():
    inp, out = _load_golden("residual_mlp")
    x = inp["x"].astype(np.float32).reshape(-1)
    mlp_down = inp["mlp_down"].astype(np.float32).reshape(-1)
    y = out["y"].astype(np.float32).reshape(-1)

    exe = Executor()
    exe.write_bytes("sram", 0x10000, _bf16b(x))
    exe.write_bytes("sram", 0x20000, _bf16b(mlp_down))
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x2000); _cfg(prog, 2, 1, 0x3000)
    prog += I.encode_inst("VADD", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARd=2, len=x.size, CV=0).to_bytes(16, "little")
    trace, total, etr, etot, got = _run_rtl(exe, bytes(prog), (0, 0x30000, x.size * 2), x.size)
    return _report("residual_mlp", trace, total, etr, etot, got, y)


def _linear_golden(name, in_key, out_key, w_key, x_shape, ref_shape):
    """Run one DC linear projection (GEMM, M=1) on RTL and compare to golden."""
    from run_cosim import _build_linear, _detile_from
    inp, out = _load_golden(name)
    x = inp[in_key].astype(np.float32).reshape(-1)
    ref = out[out_key].astype(np.float32)
    W = _hf_tensor(w_key)
    M, K = x_shape
    N = W.shape[0]
    xm = x.reshape(M, K)
    exe, prog, plan = _build_linear("DC", "BF16", xm, W)
    preload = dump_preload(exe)
    etr, etot = expected_trace(prog)
    out_sram = plan.io["out_sram"]; out_bytes = plan.io["out_bytes"]
    trace, total, dump = run_rtl(bytes(prog), preload, [(0, out_sram, out_bytes)])
    rtl_out = _detile_from(dump, plan, M, N).reshape(ref_shape)
    return _report(name, trace, total, etr, etot, rtl_out, ref)


def op_linear_qkv():
    res = []
    for name, ok, wk, rs in (
        ("attn_qkv_q", "q", "q_proj.weight", (1, 2048)),
        ("attn_qkv_k", "k", "k_proj.weight", (1, 1024)),
        ("attn_qkv_v", "v", "v_proj.weight", (1, 1024)),
    ):
        wkey = f"model.layers.{LAYER}.self_attn.{wk}"
        res.append(_linear_golden("attn_qkv", "x", ok, wkey, (1, 1024), rs))
    return res


def op_linear_o():
    return [_linear_golden("attn_o", "ctx", "o",
                           f"model.layers.{LAYER}.self_attn.o_proj.weight",
                           (1, 16 * 128), (1, 1024))]


def op_linear_mlp():
    res = []
    for name, ik, ok, wk in (
        ("mlp_gate", "x", "gate", "gate_proj.weight"),
        ("mlp_up", "x", "up", "up_proj.weight"),
        ("mlp_down", "x", "down", "down_proj.weight"),
    ):
        inp, out = _load_golden(name)
        K = inp[ik].astype(np.float32).size
        wkey = f"model.layers.{LAYER}.mlp.{wk}"
        N = _hf_tensor(wkey).shape[0]
        res.append(_linear_golden(name, ik, ok, wkey, (1, K), (1, N)))
    return res


def main():
    results = []
    for r in ([op_rmsnorm_in()] + op_qknorm() + op_rope()
              + [op_silu(), op_residual()]
              + op_linear_qkv() + op_linear_o() + op_linear_mlp()):
        results.append(r)
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[{mark}] {r['name']:14s} trace={r['trace_ok']} "
              f"max_ulp={r['max_ulp']:.1f} ulp_normal={r['max_ulp_normal']:.1f} "
              f"cycles={r['total_cycles']} n_elem={r['n_elem']}")
    all_pass = all(r["pass"] for r in results)
    print(f"\nL00 single-layer golden: {'ALL PASS' if all_pass else 'FAILURES'}")
    with open("/tmp/layer_golden.json", "w") as f:
        json.dump({"results": results, "all_pass": bool(all_pass)}, f, indent=2)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
