"""qrun weight/gamma loading.

INT8: the .qbin already holds the 141 quantized W8A8 tensors (weights + BF16
per-128-group scales). The qbin was compiled with the placeholder activation
scale sx=1.0 (qforge.config.ACTIVATION_SCALE_DEFAULT — "runtime calibration
lands in P5/qrun"), so qrun **calibrates** the activation scale at load time:
each projection's per-tensor activation scale sx = max|activation|/127 is
measured from a reference trace, the stored per-group weight scales are
rescaled by sx, and sx is written to a per-projection SRAM slot the program
CONFIGs into C_ACT before each QUANT. This keeps
    y = round(a / sx) @ W_int8 * (sw * sx) ~= a @ W
with the full INT8 dynamic range instead of collapsing the normalised
activations (~+-1) to integer rounding.

BF16: the 141 projections are loaded from the BF16 qbin tensors table by
default (container mode, `qforge compile --dtype bf16`); `--weights-from-hf`
selects the legacy safetensors path. Either way the weights are packed
[N, K] row-major (transpose_b=1 convention) and cast to BF16.

Gamma (RMSNorm weights, 113 small BF16 vectors) is not part of the 141
projections; it is written into the qrun gamma SRAM region (see
qrun/program.py gamma_layout()) at load time, tiling q_norm/k_norm per-head
vectors across heads.
"""
from __future__ import annotations

import numpy as np

import qforge.quant as Q
import qforge.graph as G
from compiler.isa.qbin import FLAG_DTYPE_MASK, DTYPE_FLAG
from qrun import bf16 as B
from qrun import program as P

WEIGHT_BASE = 0x0010_0000   # 1 MiB (mirrors qforge.build)


def _align(x: int, a: int = 64) -> int:
    return (x + a - 1) // a * a


def _bf16_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=B.BF16_NP).tobytes()


def _rescale_scales_group(raw: bytes, sx: np.ndarray, G: int) -> bytes:
    """Per-column rescale of [N, G] per-128-group weight scales by sx[g]."""
    arr = np.frombuffer(raw, dtype=B.BF16_NP).astype(np.float32).reshape(-1, G)
    arr = arr * sx[None, :].astype(np.float32)
    return np.asarray(arr.reshape(-1), dtype=B.BF16_NP).tobytes()


def _group_scales(a: np.ndarray, group: int = 128) -> np.ndarray:
    """[seq, K] fp32 -> [K//128] per-128-group symmetric scales (max|.|/127),
    pooled over the sequence dim (04 §1.5 group=128)."""
    a = np.asarray(a, dtype=np.float32)
    seq, K = a.shape
    G = K // group
    a = a.reshape(seq, G, group)
    sx = np.abs(a).max(axis=(0, 2)) / 127.0
    return np.maximum(sx, 1e-6).astype(np.float32)

def calibrate_act_scales(ref, tokenizer, prompt_text: str) -> list[np.ndarray]:
    """Per-projection per-128-group activation scales (graph order) from a
    reference trace (golden projection inputs).  Each entry is a [K//128] fp32
    array of symmetric per-group scales (max|.|/127 pooled over the seq dim)."""
    import torch
    ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].to(ref.device)
    trace = []
    ref.forward(ids, trace=trace)
    sx = {}
    for L in range(28):
        ltr = trace[L + 1][1]
        a = ltr["rmsnorm_in"]["outputs"]["y"].detach().float().cpu().numpy()
        sx[L, "qkv"] = _group_scales(a)                              # [seq,1024]
        a = ltr["attn_ctx"]["outputs"]["ctx"].detach().float().cpu().numpy()
        # ctx is [16 heads, seq, 128] -> [seq, 2048] head-major (o_proj K)
        seq = a.shape[1]
        a = a.transpose(1, 0, 2).reshape(seq, -1)
        sx[L, "o"] = _group_scales(a)
        a = ltr["rmsnorm_mlp"]["outputs"]["y"].detach().float().cpu().numpy()
        sx[L, "gate"] = sx[L, "up"] = _group_scales(a)               # [seq,1024]
        a = ltr["mlp_silu"]["outputs"]["y"].detach().float().cpu().numpy()
        sx[L, "down"] = _group_scales(a)                             # [seq,3072]
    a = trace[29][1]["outputs"]["y"].detach().float().cpu().numpy()
    sx[None, "lm_head"] = _group_scales(a)                           # [seq,1024]
    out = []
    for L in range(28):
        for kind in ("qkv", "o", "gate", "up", "down"):
            out.append(sx[L, kind])
    out.append(sx[None, "lm_head"])
    return out

def calibrate_act_inputs(ref, tokenizer, prompt_text: str) -> list[np.ndarray]:
    """Per-projection calibration activation VALUES (graph order), for AWQ
    weight-scale search. Each entry is the [seq, K] fp32 projection input
    captured from the reference trace (golden projection inputs)."""
    import torch
    ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].to(ref.device)
    trace = []
    ref.forward(ids, trace=trace)
    x = {}
    for L in range(28):
        ltr = trace[L + 1][1]
        x[L, "qkv"] = ltr["rmsnorm_in"]["outputs"]["y"].detach().float().cpu().numpy()
        a = ltr["attn_ctx"]["outputs"]["ctx"].detach().float().cpu().numpy()
        seq = a.shape[1]
        x[L, "o"] = a.transpose(1, 0, 2).reshape(seq, -1)             # [seq,2048]
        x[L, "gate"] = x[L, "up"] = \
            ltr["rmsnorm_mlp"]["outputs"]["y"].detach().float().cpu().numpy()
        x[L, "down"] = ltr["mlp_silu"]["outputs"]["y"].detach().float().cpu().numpy()
    x[None, "lm_head"] = trace[29][1]["outputs"]["y"].detach().float().cpu().numpy()
    out = []
    for L in range(28):
        for kind in ("qkv", "o", "gate", "up", "down"):
            out.append(x[L, kind])
    out.append(x[None, "lm_head"])
    return out


def int8_layouts(qbin, qmetal, act_scales=None) -> list[P.Layout]:
    """Load INT8 tensors into HBM, rescaling the per-128-group weight scales
    per-column by the calibrated per-group activation scales (cd = sx[g]·sw[n,g]);
    return graph-order layouts."""
    tmap = {t.name: t for t in qbin.tensors}
    projs = G.build_graph()
    gaddrs = P.act_scale_layout()
    layouts = []
    for i, p in enumerate(projs):
        t = tmap[p.name]
        qmetal.load_tensor_hbm(t.hbm_off, t.data)
        Gk = p.K // 128
        if act_scales is None:
            sx = np.full(Gk, 1.0, dtype=np.float32)
        else:
            sx = np.asarray(act_scales[i], dtype=np.float32)
            assert sx.shape == (Gk,), (p.name, sx.shape, Gk)
        if t.scales is not None:
            qmetal.load_tensor_hbm(t.scales_hbm_off,
                                   _rescale_scales_group(t.scales, sx, Gk))
            qmetal.write_sram(gaddrs[p.name], _bf16_bytes(sx))
        layouts.append(P.Layout(proj=p, wq_hbm=t.hbm_off,
                                scale_hbm=t.scales_hbm_off,
                                act_sram=gaddrs[p.name] // 16))
    return layouts


def int8_weights_end(qbin) -> int:
    end = 0
    for t in qbin.tensors:
        end = max(end, t.hbm_off + len(t.data))
        if t.scales is not None:
            end = max(end, t.scales_hbm_off + len(t.scales))
    return end


def check_bf16_container(qbin) -> None:
    """Validate the container's default-dtype flag is BF16 (bit[1:0]==0).

    Refuses to load a non-BF16 container (e.g. an INT8 W8A8 qbin) when a BF16
    load is requested — no silent fallback to safetensors or a wrong dtype."""
    code = qbin.flags & FLAG_DTYPE_MASK
    if code != DTYPE_FLAG["BF16"]:
        names = {"BF16": 0, "FP16": 1, "INT8": 2, "INT4": 3}
        label = next((k for k, v in names.items() if v == code), code)
        raise ValueError(
            f"container flags dtype={label!r} (bit[1:0]={code}) != BF16(0); "
            f"refusing bf16 load from a non-BF16 container — recompile with "
            f"`qforge compile --dtype bf16` or pass --weights-from-hf")


def bf16_layouts_from_qbin(qbin, qmetal) -> tuple[list[P.Layout], int]:
    """Load BF16 projection weights from the qbin tensors table (container
    mode); return (layouts, weights_end). Each of the 141 tensors must be BF16
    with scales omitted (the BF16 container writes no scale section)."""
    tmap = {t.name: t for t in qbin.tensors}
    projs = G.build_graph()
    layouts = []
    end = 0
    for p in projs:
        t = tmap[p.name]
        assert t.dtype == "BF16", (p.name, t.dtype)
        assert t.scales is None, (p.name, "BF16 container must omit scales")
        assert tuple(t.shape) == (p.N, p.K), (p.name, t.shape, p.N, p.K)
        qmetal.load_tensor_hbm(t.hbm_off, t.data)
        layouts.append(P.Layout(proj=p, wq_hbm=t.hbm_off, scale_hbm=None))
        end = max(end, t.hbm_off + len(t.data))
    return layouts, end

def bf16_layouts(st, qmetal, base: int = WEIGHT_BASE) -> tuple[list[P.Layout], int]:
    """Load BF16 projection weights; return (layouts, weights_end)."""
    projs = G.build_graph()
    layouts = []
    cursor = base
    for p in projs:
        parts = [st.get_float32(k) for k in p.weight_keys]
        w = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        assert tuple(w.shape) == (p.N, p.K), (p.name, w.shape, p.N, p.K)
        wbf = np.asarray(w, dtype=B.BF16_NP)
        nbytes = wbf.nbytes
        wq_hbm = _align(cursor)
        qmetal.load_tensor_hbm(wq_hbm, wbf.tobytes())
        layouts.append(P.Layout(proj=p, wq_hbm=wq_hbm, scale_hbm=None))
        cursor = wq_hbm + nbytes
    return layouts, cursor

def int4_layouts(st, qmetal, base: int = WEIGHT_BASE,
                 act_inputs: list[np.ndarray] | None = None
                 ) -> tuple[list[P.Layout], int]:
    """Load INT4 projection weights (W4A16); return (layouts, weights_end).

    Weights are quantized symmetric per-128-K-group (qforge.quant) and packed
    2-per-byte (even -> low nibble, qforge.quant.pack_int4), loaded row-major
    [N, K//2] bytes at freshly assigned HBM offsets (transpose_b=1 convention).
    Per-128-group scales sw are stored as [N, G] BF16 (the executor's
    `_gemm_dequant` [N, G] layout) and DMA-loaded per output tile.

    If `act_inputs` (graph order, [seq, K] calibration activations) is given,
    scales are searched AWQ-style (`qforge.quant.quantize_weight_int4_awq`)
    instead of plain symmetric max|w|/7.
    """
    projs = G.build_graph()
    layouts = []
    cursor = base
    for i, p in enumerate(projs):
        parts = [st.get_float32(k) for k in p.weight_keys]
        w = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        assert tuple(w.shape) == (p.N, p.K), (p.name, w.shape, p.N, p.K)
        if act_inputs is not None:
            wqi, sw = Q.quantize_weight_int4_awq(w, act_inputs[i])
        else:
            wqi, sw = Q.quantize_weight_int4(w)      # [N,K] int8, [N,G] fp32
        packed = Q.pack_int4(wqi)                     # [N, K//2] uint8
        wq_hbm = _align(cursor)
        qmetal.load_tensor_hbm(wq_hbm, packed.tobytes())
        scale_bf = np.asarray(sw, dtype=B.BF16_NP)    # [N, G] bf16
        scale_hbm = _align(wq_hbm + packed.nbytes)
        qmetal.load_tensor_hbm(scale_hbm, scale_bf.tobytes())
        layouts.append(P.Layout(proj=p, wq_hbm=wq_hbm, scale_hbm=scale_hbm))
        cursor = scale_hbm + scale_bf.nbytes
    return layouts, cursor


def inject_gamma(st, qmetal) -> None:
    """Write the 113 RMSNorm gamma vectors into the qrun gamma SRAM region."""
    gaddrs = P.gamma_layout()
    for L in range(28):
        inp = st.get_float32(f"model.layers.{L}.input_layernorm.weight")       # [1024]
        q = st.get_float32(f"model.layers.{L}.self_attn.q_norm.weight")        # [128]
        k = st.get_float32(f"model.layers.{L}.self_attn.k_norm.weight")        # [128]
        post = st.get_float32(f"model.layers.{L}.post_attention_layernorm.weight")
        qmetal.write_sram(gaddrs[("input", L)], _bf16_bytes(inp))
        qmetal.write_sram(gaddrs[("q_norm", L)], _bf16_bytes(np.tile(q, 16)))
        qmetal.write_sram(gaddrs[("k_norm", L)], _bf16_bytes(np.tile(k, 8)))
        qmetal.write_sram(gaddrs[("post", L)], _bf16_bytes(post))
    final = st.get_float32("model.norm.weight")
    qmetal.write_sram(gaddrs[("final", None)], _bf16_bytes(final))


def inject_act_scale(qmetal, layout_M1, layout_M128) -> None:
    """Write the calibrated per-tensor activation scale (BF16) for both layouts
    (fallback for BF16 mode, where no QUANT happens but the slot is still
    initialised to 1.0)."""
    one = _bf16_bytes(np.array([1.0], dtype=np.float32))
    qmetal.write_sram(layout_M1["act_scale"], one)
    qmetal.write_sram(layout_M128["act_scale"], one)
