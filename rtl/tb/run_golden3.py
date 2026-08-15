"""P8 three-way golden driver (M7): PyTorch <-> qsim <-> RTL.

For each of the 15 op instances at L00 / decode_seq1_cache1024, run the same
ISA program on (a) the qsim reference Executor and (b) the Verilator RTL, and
compare both against the P1 golden (PyTorch bf16) output.  Criterion = M4:
|y| >= 1e-3 must be <= 1 bf16 ULP; tiny (|y| < 1e-3) elements are recorded as
exceptions.  Per-layer hidden-state three-way is run for L00 / L13 / L27
(residual_mlp output = layer output hidden).

Three-way columns:
  qsim<->PyTorch : Executor output  vs golden bf16
  RTL<->qsim     : RTL output       vs Executor output (also trace/cycle exact)
  RTL<->PyTorch  : RTL output       vs golden bf16

Writes docs/p8/golden3-results.json and prints the ULP table.
"""
from __future__ import annotations

import json
import os
import struct
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
from cosim import (dump_preload, run_rtl, expected_trace,  # noqa: E402
                   bf16_ulp_dist)
from run_cosim import _build_linear as _cosim_build_linear, _detile_from  # noqa: E402

LAYER = 0
EPS = 1e-6
ROPE_THETA = 1_000_000.0
POS = 1024

# model card (01b)
H, HD = 1024, 128
QH, KVH, GQA = 16, 8, 2
CACHE_LEN = 1025                    # 1024 cache + 1 current token
ATTN_SCALE = np.float32(HD ** -0.5)
N_TILE = 128

# attention SRAM layout (byte addresses, 16B aligned)
Q_BASE = 0x10000                    # q   [16, 128]        head stride 256
K_BASE = 0x20000                    # k   [8, 1025, 128]   head stride 262400
V_BASE = 0x20000                    # v   (same shape as k)
P_BASE = 0x10000                    # probs [16, 1025]     head stride 2064
SC_BASE = 0x10000                   # scores [16, 1025]    head stride 2064
S_BASE = 0x240000                   # attn_score output scores
C_BASE = 0x240000                   # attn_ctx output ctx [16, 128] head stride 256
HEAD_STRIDE_S = 2064                # 1025 bf16 (2050 B) padded to 129 words
K_HEAD_STRIDE = CACHE_LEN * HD * 2  # 262400 B


def _fp32_bits(v: float) -> int:
    return int(np.array([v], dtype=np.float32).view(np.uint32)[0])


def _bf16b(x) -> bytes:
    return np.asarray(x, dtype=np.float32).astype(BF16).tobytes()


def _cfg(prog: bytearray, reg: int, cls: int, val: int):
    prog += I.encode_inst("CONFIG", REG=reg, reg_class=cls,
                          IMM64=int(val)).to_bytes(16, "little")


def _stride(row: int, batch: int = 0) -> int:
    return (row << 16) | batch


def _w(byte_addr: int) -> int:
    assert byte_addr % 16 == 0
    return byte_addr // 16


def _load_golden(op: str, layer: int = LAYER):
    d = os.path.join(GOLDEN, CACHE, f"L{layer:02d}_{op}")
    return (np.load(os.path.join(d, "inputs.npz")),
            np.load(os.path.join(d, "outputs.npz")))


def _hf_tensor(key: str) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(MODEL_SAFETENSORS, framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def _write_padded_heads(exe: Executor, base: int, x: np.ndarray, stride: int):
    """Write x [n_heads, n_cols] (fp32) to SRAM at `base`, each head at a
    `stride`-byte pitch (padding between heads)."""
    n_heads, n_cols = x.shape
    buf = bytearray(n_heads * stride)
    for h in range(n_heads):
        buf[h * stride:h * stride + n_cols * 2] = _bf16b(x[h])
    exe.write_bytes("sram", base, bytes(buf))


# --------------------------------------------------------------------------- #
# three-way runner
# --------------------------------------------------------------------------- #
def _stats(d: np.ndarray, ref: np.ndarray) -> dict:
    """ULP stats of `d` vs `ref` under the M4 criterion."""
    ref = np.asarray(ref, np.float32).reshape(-1)
    d = np.asarray(d, np.float32).reshape(-1)
    u = bf16_ulp_dist(d, ref)
    normal = np.abs(ref) >= 1e-3
    tiny = ~normal
    max_ulp = float(np.ceil(u.max())) if u.size else 0.0
    max_ulp_normal = float(np.ceil(u[normal].max())) if normal.any() else 0.0
    max_ulp_tiny = float(np.ceil(u[tiny].max())) if tiny.any() else 0.0
    n_tiny = int(tiny.sum())
    return {"max_ulp": max_ulp, "max_ulp_normal": max_ulp_normal,
            "n_tiny": n_tiny, "max_ulp_tiny": max_ulp_tiny,
            "pass": max_ulp_normal <= 1.0, "n_elem": int(ref.size)}


def _threeway(exe: Executor, prog: bytes, regions, qsim_reader, rtl_reader,
              ref, name: str) -> dict:
    preload = dump_preload(exe)
    etr, etot = expected_trace(prog)
    trace, total, dump = run_rtl(bytes(prog), preload, regions)
    rtl = rtl_reader(dump)
    exe.run(bytes(prog))
    qsim = qsim_reader(exe)
    ref = np.asarray(ref, np.float32).reshape(-1)
    rtl = np.asarray(rtl, np.float32).reshape(-1)
    qsim = np.asarray(qsim, np.float32).reshape(-1)
    trace_ok = bool((trace == etr) and (total == etot))
    qp = _stats(qsim, ref)
    rq = _stats(rtl, qsim)
    rp = _stats(rtl, ref)
    # RTL<->qsim must also be cycle/trace exact (co-sim contract).
    rq["trace_ok"] = trace_ok
    rq["pass"] = rq["pass"] and trace_ok
    return {"name": name, "trace_ok": trace_ok,
            "total_cycles": int(total),
            "qsim_pytorch": qp, "rtl_qsim": rq, "rtl_pytorch": rp}


def _vec_reader(byte_addr: int, n: int):
    def r(exe_or_dump):
        if isinstance(exe_or_dump, Executor):
            return np.frombuffer(exe_or_dump.read_bytes("sram", byte_addr, n * 2),
                                 dtype=BF16).astype(np.float32)
        return np.frombuffer(exe_or_dump, dtype=BF16).astype(np.float32)
    return r


# --------------------------------------------------------------------------- #
# op builders (each returns exe, prog, regions, qsim_reader, rtl_reader, ref)
# --------------------------------------------------------------------------- #
def _build_rmsnorm(op: str, gamma_key: str, layer: int):
    inp, out = _load_golden(op, layer)
    x = inp["x"].astype(np.float32).reshape(-1)
    y = out["y"].astype(np.float32).reshape(-1)
    gamma = _hf_tensor(gamma_key.format(layer))
    exe = Executor()
    exe.write_bytes("sram", 0x10000, _bf16b(x))
    exe.write_bytes("sram", 0x20000, _bf16b(gamma))
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x2000); _cfg(prog, 2, 1, 0x3000)
    _cfg(prog, 5, 0, _fp32_bits(EPS))
    prog += I.encode_inst("RMSNORM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARd=2, len=x.size, CV=5, imm=0).to_bytes(16, "little")
    return exe, bytes(prog), [(0, 0x30000, x.size * 2)], \
        _vec_reader(0x30000, x.size), _vec_reader(0, x.size), y


def _build_residual(op: str, layer: int):
    inp, out = _load_golden(op, layer)
    x = inp["x"].astype(np.float32).reshape(-1)
    d = inp["attn_o" if op == "residual_attn" else "mlp_down"].astype(np.float32).reshape(-1)
    y = out["y"].astype(np.float32).reshape(-1)
    exe = Executor()
    exe.write_bytes("sram", 0x10000, _bf16b(x))
    exe.write_bytes("sram", 0x20000, _bf16b(d))
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x2000); _cfg(prog, 2, 1, 0x3000)
    prog += I.encode_inst("VADD", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARd=2, len=x.size, CV=0).to_bytes(16, "little")
    return exe, bytes(prog), [(0, 0x30000, x.size * 2)], \
        _vec_reader(0x30000, x.size), _vec_reader(0, x.size), y


def _build_linear(op: str, in_key: str, out_key: str, w_key: str,
                  x_shape, ref_shape, layer: int):
    inp, out = _load_golden(op, layer)
    x = inp[in_key].astype(np.float32).reshape(-1)
    ref = out[out_key].astype(np.float32)
    W = _hf_tensor(w_key.format(layer))
    M, K = x_shape
    N = W.shape[0]
    xm = x.reshape(M, K)
    exe, prog, plan = _cosim_build_linear("DC", "BF16", xm, W)
    out_sram = plan.io["out_sram"]; out_bytes = plan.io["out_bytes"]

    def qsim_reader(e):
        raw = e.read_bytes("sram", out_sram, out_bytes)
        return _detile_from(raw, plan, M, N).reshape(ref_shape)

    def rtl_reader(dump):
        return _detile_from(dump, plan, M, N).reshape(ref_shape)

    return exe, prog, [(0, out_sram, out_bytes)], qsim_reader, rtl_reader, ref


def _build_rope(layer: int):
    inp, out = _load_golden("attn_rope", layer)
    q = inp["q"].astype(np.float32).reshape(-1)
    k = inp["k"].astype(np.float32).reshape(-1)
    q_out = out["q"].astype(np.float32)
    k_out = out["k"].astype(np.float32)
    exe = Executor()
    exe.write_bytes("sram", 0x10000, _bf16b(np.concatenate([q, k])))
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 2, 1, 0x3000)
    _cfg(prog, 6, 0, _fp32_bits(ROPE_THETA))
    prog += I.encode_inst("ROPE", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=0, ARd=2, len=q.size + k.size, CV=6,
                          imm=POS).to_bytes(16, "little")
    ref = np.concatenate([q_out.reshape(-1), k_out.reshape(-1)])
    return exe, bytes(prog), [(0, 0x30000, (q.size + k.size) * 2)], \
        _vec_reader(0x30000, q.size + k.size), _vec_reader(0, q.size + k.size), ref


def _build_qknorm(layer: int):
    inp, out = _load_golden("attn_qknorm", layer)
    q = inp["q"].astype(np.float32).reshape(-1)
    k = inp["k"].astype(np.float32).reshape(-1)
    q_out = out["q"].astype(np.float32)
    k_out = out["k"].astype(np.float32)
    qg = _hf_tensor(f"model.layers.{layer}.self_attn.q_norm.weight")
    kg = _hf_tensor(f"model.layers.{layer}.self_attn.k_norm.weight")
    exe = Executor()
    exe.write_bytes("sram", 0x10000, _bf16b(np.concatenate([q, k])))
    exe.write_bytes("sram", 0x20000, _bf16b(np.concatenate([np.tile(qg, 16), np.tile(kg, 8)])))
    prog = bytearray()
    _cfg(prog, 0, 1, 0x1000); _cfg(prog, 1, 1, 0x2000); _cfg(prog, 2, 1, 0x3000)
    _cfg(prog, 5, 0, _fp32_bits(EPS))
    prog += I.encode_inst("RMSNORM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                          ARa=0, ARb=1, ARd=2, len=q.size + k.size, CV=5,
                          imm=(1 << 31)).to_bytes(16, "little")
    ref = np.concatenate([q_out.reshape(-1), k_out.reshape(-1)])
    return exe, bytes(prog), [(0, 0x30000, (q.size + k.size) * 2)], \
        _vec_reader(0x30000, q.size + k.size), _vec_reader(0, q.size + k.size), ref


def _build_silu(layer: int):
    inp, out = _load_golden("mlp_silu", layer)
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
    return exe, bytes(prog), [(0, 0x60000, gate.size * 2)], \
        _vec_reader(0x60000, gate.size), _vec_reader(0, gate.size), y


def _build_attn_score(layer: int):
    """scores[16, 1025] = (q[16,128] @ k_rep^T) * attn_scale, tiled N=128.

    GQA: for each kv head g, batch=2 q heads share k[g] (batch_stride_B=0).
    """
    inp, out = _load_golden("attn_score", layer)
    q = inp["q"].astype(np.float32)          # [16,1,128]
    k = inp["k"].astype(np.float32)          # [8,1025,128]
    ref = out["scores"].astype(np.float32)   # [16,1,1025]

    exe = Executor()
    exe.write_bytes("sram", Q_BASE, _bf16b(q.reshape(-1)))
    exe.write_bytes("sram", K_BASE, _bf16b(k.reshape(-1)))
    prog = bytearray()
    _cfg(prog, 0, 0, _stride(HD * 2, HD * 2))               # C0: A row=256 batch=256
    _cfg(prog, 1, 0, _stride(HD * 2, 0))                    # C1: B row=256 batch=0
    _cfg(prog, 2, 0, _stride(HEAD_STRIDE_S, HEAD_STRIDE_S))  # C2: C row/batch=2064
    _cfg(prog, 5, 0, _fp32_bits(ATTN_SCALE))
    n_tiles = (CACHE_LEN + N_TILE - 1) // N_TILE            # 9 (8x128 + 1x1)
    for g in range(KVH):
        for t in range(n_tiles):
            tile_w = N_TILE if t < n_tiles - 1 else CACHE_LEN - t * N_TILE
            a_w = _w(Q_BASE + 2 * g * HD * 2)
            b_w = _w(K_BASE + g * K_HEAD_STRIDE + t * N_TILE * HD * 2)
            c_w = _w(S_BASE + 2 * g * HEAD_STRIDE_S + t * N_TILE * 2)
            _cfg(prog, 0, 1, a_w); _cfg(prog, 1, 1, b_w); _cfg(prog, 2, 1, c_w)
            prog += I.encode_inst(
                "BMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                ARa=0, ARb=1, ARc=2, M=1, N=tile_w, K=HD, batch=GQA,
                CA=0, CB=1, CC=2, CD=0, acc_init=1, bsrc=0, dequant=0,
                transpose_A=0, transpose_B=1).to_bytes(16, "little")
    # VSCALE the full scores buffer (16 x 2064 B = 16512 elems) in <=4096 chunks
    total_elems = QH * HEAD_STRIDE_S // 2
    off = 0
    while off < total_elems:
        n = min(4096, total_elems - off)
        _cfg(prog, 0, 1, _w(S_BASE + off * 2))
        prog += I.encode_inst("VSCALE", srcA=I.DT_BF16, srcB=I.DT_BF16,
                              acc=I.ACC_FP32, ARa=0, ARb=0, ARd=0, len=n,
                              CV=5, imm=0).to_bytes(16, "little")
        off += n

    def qsim_reader(e):
        out = np.empty((QH, CACHE_LEN), dtype=np.float32)
        for h in range(QH):
            raw = e.read_bytes("sram", S_BASE + h * HEAD_STRIDE_S, CACHE_LEN * 2)
            out[h] = np.frombuffer(raw, dtype=BF16).astype(np.float32)
        return out.reshape(-1)

    return exe, bytes(prog), [(0, S_BASE, QH * HEAD_STRIDE_S)], \
        qsim_reader, _score_rtl_reader, ref.reshape(-1)


def _score_rtl_reader(dump):
    out = np.empty((QH, CACHE_LEN), dtype=np.float32)
    for h in range(QH):
        raw = dump[h * HEAD_STRIDE_S:h * HEAD_STRIDE_S + CACHE_LEN * 2]
        out[h] = np.frombuffer(raw, dtype=BF16).astype(np.float32)
    return out.reshape(-1)


def _build_attn_softmax(layer: int):
    """probs[16,1025] = softmax(scores) — per-head, bf16-per-op ISA datapath.

    Uses separate input/scratch/output buffers (no in-place vector op) to avoid
    the CP read-after-write hazard on the last element observed at len=1025
    (see report § finding)."""
    inp, out = _load_golden("attn_softmax", layer)
    scores = inp["scores"].astype(np.float32)   # [16,1,1025]
    mask = inp["mask"].astype(np.float32)       # [1,1025]
    ref = out["probs"].astype(np.float32)       # [16,1,1025]
    assert np.all(mask == 0), "decode mask must be all-zero"

    exe = Executor()
    _write_padded_heads(exe, SC_BASE, scores.reshape(QH, CACHE_LEN), HEAD_STRIDE_S)
    ES_BASE = 0x300000                          # VSUB output
    EX_BASE = 0x320000                          # VEXP output
    MAXV = 0x260000; SUMV = 0x261000; RINV = 0x262000
    prog = bytearray()
    _cfg(prog, 8, 0, 1)  # C8 = 1 group (per-head reduce)
    for h in range(QH):
        hw = _w(SC_BASE + h * HEAD_STRIDE_S)    # scores (input)
        ew = _w(ES_BASE + h * HEAD_STRIDE_S)    # es = scores - max
        xw = _w(EX_BASE + h * HEAD_STRIDE_S)    # ex = exp(es)
        # 1. max
        _cfg(prog, 0, 1, hw); _cfg(prog, 2, 1, _w(MAXV))
        prog += I.encode_inst("VREDUCE_MAX", srcA=I.DT_BF16, srcB=I.DT_BF16,
                              acc=I.ACC_FP32, ARa=0, ARb=0, ARd=2, len=CACHE_LEN,
                              CV=8, imm=0).to_bytes(16, "little")
        # 2. sub (broadcast max) -> es
        _cfg(prog, 0, 1, hw); _cfg(prog, 1, 1, _w(MAXV)); _cfg(prog, 2, 1, ew)
        prog += I.encode_inst("VSUB", srcA=I.DT_BF16, srcB=I.DT_BF16,
                              acc=I.ACC_FP32, ARa=0, ARb=1, ARd=2, len=CACHE_LEN,
                              CV=1, imm=0).to_bytes(16, "little")
        # 3. exp -> ex
        _cfg(prog, 0, 1, ew); _cfg(prog, 2, 1, xw)
        prog += I.encode_inst("VEXP", srcA=I.DT_BF16, srcB=I.DT_BF16,
                              acc=I.ACC_FP32, ARa=0, ARb=0, ARd=2, len=CACHE_LEN,
                              CV=0, imm=0).to_bytes(16, "little")
        # 4. sum
        _cfg(prog, 0, 1, xw); _cfg(prog, 2, 1, _w(SUMV))
        prog += I.encode_inst("VREDUCE_SUM", srcA=I.DT_BF16, srcB=I.DT_BF16,
                              acc=I.ACC_FP32, ARa=0, ARb=0, ARd=2, len=CACHE_LEN,
                              CV=8, imm=0).to_bytes(16, "little")
        # 5. recip
        _cfg(prog, 0, 1, _w(SUMV)); _cfg(prog, 2, 1, _w(RINV))
        prog += I.encode_inst("VRECIP", srcA=I.DT_BF16, srcB=I.DT_BF16,
                              acc=I.ACC_FP32, ARa=0, ARb=0, ARd=2, len=1,
                              CV=0, imm=0).to_bytes(16, "little")
        # 6. mul (broadcast rinv) -> probs (overwrite scores at SC_BASE)
        _cfg(prog, 0, 1, xw); _cfg(prog, 1, 1, _w(RINV)); _cfg(prog, 2, 1, hw)
        prog += I.encode_inst("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16,
                              acc=I.ACC_FP32, ARa=0, ARb=1, ARd=2, len=CACHE_LEN,
                              CV=1, imm=0).to_bytes(16, "little")

    def qsim_reader(e):
        out = np.empty((QH, CACHE_LEN), dtype=np.float32)
        for h in range(QH):
            raw = e.read_bytes("sram", SC_BASE + h * HEAD_STRIDE_S, CACHE_LEN * 2)
            out[h] = np.frombuffer(raw, dtype=BF16).astype(np.float32)
        return out.reshape(-1)

    def rtl_reader(dump):
        out = np.empty((QH, CACHE_LEN), dtype=np.float32)
        for h in range(QH):
            raw = dump[h * HEAD_STRIDE_S:h * HEAD_STRIDE_S + CACHE_LEN * 2]
            out[h] = np.frombuffer(raw, dtype=BF16).astype(np.float32)
        return out.reshape(-1)

    return exe, bytes(prog), [(0, SC_BASE, QH * HEAD_STRIDE_S)], \
        qsim_reader, rtl_reader, ref.reshape(-1)


def _build_attn_ctx(layer: int):
    """ctx[16,128] = probs[16,1025] @ v_rep[8->16,1025,128] (K=1025, one BMM/group)."""
    inp, out = _load_golden("attn_ctx", layer)
    probs = inp["probs"].astype(np.float32)     # [16,1,1025]
    v = inp["v"].astype(np.float32)             # [8,1025,128]
    ref = out["ctx"].astype(np.float32)         # [16,1,128]

    exe = Executor()
    _write_padded_heads(exe, P_BASE, probs.reshape(QH, CACHE_LEN), HEAD_STRIDE_S)
    exe.write_bytes("sram", V_BASE, _bf16b(v.reshape(-1)))
    prog = bytearray()
    _cfg(prog, 0, 0, _stride(HEAD_STRIDE_S, HEAD_STRIDE_S))  # A row/batch=2064
    _cfg(prog, 1, 0, _stride(HD * 2, 0))                     # B row=256 batch=0
    _cfg(prog, 2, 0, _stride(HD * 2, HD * 2))                # C row/batch=256
    for g in range(KVH):
        _cfg(prog, 0, 1, _w(P_BASE + 2 * g * HEAD_STRIDE_S))
        _cfg(prog, 1, 1, _w(V_BASE + g * K_HEAD_STRIDE))
        _cfg(prog, 2, 1, _w(C_BASE + 2 * g * HD * 2))
        prog += I.encode_inst(
            "BMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
            ARa=0, ARb=1, ARc=2, M=1, N=HD, K=CACHE_LEN, batch=GQA,
            CA=0, CB=1, CC=2, CD=0, acc_init=1, bsrc=0, dequant=0,
            transpose_A=0, transpose_B=0).to_bytes(16, "little")

    def reader(e_or_dump):
        if isinstance(e_or_dump, Executor):
            raw = e_or_dump.read_bytes("sram", C_BASE, QH * HD * 2)
        else:
            raw = e_or_dump[:QH * HD * 2]
        return np.frombuffer(raw, dtype=BF16).astype(np.float32)

    return exe, bytes(prog), [(0, C_BASE, QH * HD * 2)], \
        reader, reader, ref.reshape(-1)


# --------------------------------------------------------------------------- #
# op instance table (15 classes; multi-output ops aggregated over sub-outputs)
# --------------------------------------------------------------------------- #
def _build_all_ops(layer: int):
    """Return a list of (op_class, builder) for the 15 golden instances."""
    return [
        ("rmsnorm_in",   lambda: _build_rmsnorm("rmsnorm_in",
                                                "model.layers.{}.input_layernorm.weight", layer)),
        ("rmsnorm_mlp",  lambda: _build_rmsnorm("rmsnorm_mlp",
                                                "model.layers.{}.post_attention_layernorm.weight", layer)),
        ("attn_qknorm",  lambda: _build_qknorm(layer)),
        ("attn_qkv",     lambda: _build_qkv(layer)),
        ("attn_rope",    lambda: _build_rope(layer)),
        ("attn_score",   lambda: _build_attn_score(layer)),
        ("attn_softmax", lambda: _build_attn_softmax(layer)),
        ("attn_ctx",     lambda: _build_attn_ctx(layer)),
        ("attn_o",       lambda: _build_linear("attn_o", "ctx", "o",
                                                "model.layers.{}.self_attn.o_proj.weight",
                                                (1, QH * HD), (1, H), layer)),
        ("mlp_gate",     lambda: _build_linear("mlp_gate", "x", "gate",
                                                "model.layers.{}.mlp.gate_proj.weight",
                                                (1, H), (1, 3072), layer)),
        ("mlp_up",       lambda: _build_linear("mlp_up", "x", "up",
                                                "model.layers.{}.mlp.up_proj.weight",
                                                (1, H), (1, 3072), layer)),
        ("mlp_silu",     lambda: _build_silu(layer)),
        ("mlp_down",     lambda: _build_linear("mlp_down", "x", "down",
                                                "model.layers.{}.mlp.down_proj.weight",
                                                (1, 3072), (1, H), layer)),
        ("residual_attn", lambda: _build_residual("residual_attn", layer)),
        ("residual_mlp", lambda: _build_residual("residual_mlp", layer)),
    ]


def _build_qkv(layer: int):
    """attn_qkv -> q/k/v three sub-outputs concatenated."""
    parts = []
    for ok, wk, rs in (("q", "q_proj.weight", 2048), ("k", "k_proj.weight", 1024),
                       ("v", "v_proj.weight", 1024)):
        exe, prog, regions, qr, rr, ref = _build_linear(
            "attn_qkv", "x", ok, f"model.layers.{{}}.self_attn.{wk}",
            (1, 1024), (1, rs), layer)
        parts.append((exe, prog, regions, qr, rr, ref))
    # run each sub-output through _threeway separately and aggregate in main
    return parts


def main():
    results = []

    # --- 15 op instances at L00 (attn_qkv = 3 sub-outputs) ---
    for op_class, builder in _build_all_ops(LAYER):
        built = builder()
        if op_class == "attn_qkv":
            for i, (exe, prog, regions, qr, rr, ref) in enumerate(built):
                tag = ["q", "k", "v"][i]
                r = _threeway(exe, prog, regions, qr, rr, ref,
                              f"{op_class}_{tag}")
                r["op_class"] = op_class
                results.append(r)
        else:
            r = _threeway(*built, op_class)
            r["op_class"] = op_class
            results.append(r)

    # --- per-layer hidden (residual_mlp) three-way for L00/L13/L27 ---
    for layer in (0, 13, 27):
        exe, prog, regions, qr, rr, ref = _build_residual("residual_mlp", layer)
        r = _threeway(exe, prog, regions, qr, rr, ref,
                      f"hidden_L{layer:02d}")
        r["op_class"] = "hidden"
        r["layer"] = layer
        results.append(r)

    # console table (per op class)
    print(f"{'op':16s} {'qsim-PT':>8s} {'RTL-qsim':>9s} {'RTL-PT':>8s} "
          f"{'trace':>5s} {'cyc':>7s}")
    for r in results:
        qp, rq, rp = r["qsim_pytorch"], r["rtl_qsim"], r["rtl_pytorch"]
        print(f"{r['name']:16s} "
              f"{qp['max_ulp_normal']:>6.1f}u/{qp['n_tiny']:>4d} "
              f"{rq['max_ulp_normal']:>6.1f}u/{rq['n_tiny']:>4d} "
              f"{rp['max_ulp_normal']:>6.1f}u/{rp['n_tiny']:>4d} "
              f"{str(rq['trace_ok']):>5s} {r['total_cycles']:>7d}")

    out_path = os.path.join(REPO, "docs", "p8", "golden3-results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"layer": LAYER, "cache": CACHE, "results": results}, f,
                  indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
