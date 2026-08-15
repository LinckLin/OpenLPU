from __future__ import annotations

"""qbin assembly: memory layout + full-model / single-projection qbin writer.

Full model (W8A8 / W4A16): 141 weight tensors (28 layers x {qkv,o,gate,up,down}
+ lm_head) + matching per-128-group BF16 scale tensors, plus one PF program
(GEMM) and one DC program (GEMV) covering every projection with DMA move
segments. BF16 mode: the same 141 tensors stored as raw BF16 weights (64B
aligned) with scales omitted; PF/DC programs are left empty (qrun regenerates
the BF16 programs at load time). The activation input / output regions are
shared HBM scratch (not part of the file body); weights + scales are laid out
sequentially from WEIGHT_BASE and written at hbm_off == file offset
(00-container §2).
"""
from dataclasses import dataclass, field

import numpy as np

try:
    import ml_dtypes
    _BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    _BF16 = np.float16

from compiler.isa.qbin import (Tensor, write_qbin, read_qbin,
                               FLAG_DTYPE_MASK, FLAG_DUAL_MODE, DTYPE_FLAG)
from . import config as C
from . import graph, quant, program
from . import lowering as L

WEIGHT_BASE = 0x0010_0000   # 1 MiB — first weight tensor base (after header)

# shared activation scratch (runtime, not written into the file body)
INPUT_SCRATCH = 0x0010_0000    # 1 MiB, sized for the largest PF input (128x3072)
OUTPUT_SCRATCH = 0x1000_0000   # 256 MiB, sized for lm_head PF logits (77.8 MB)


def _flags(quant_j: dict) -> int:
    """Container flags: bits[1:0] default dtype code, bit2 dual-mode.

    compiler/isa/qbin.py's default derives the code from quant['mode'], which
    only understands dtype strings ("BF16"/"INT8"/...), not the W8A8/W4A16
    quant-mode namespace — so we resolve the dtype code here and pass flags
    explicitly.
    """
    mode = quant_j.get("mode", "BF16")
    code = {"W8A8": DTYPE_FLAG["INT8"], "W4A16": DTYPE_FLAG["INT4"]}.get(
        mode, DTYPE_FLAG.get(mode, 0))
    return (code & FLAG_DTYPE_MASK) | FLAG_DUAL_MODE

def _align(x: int, a: int = 64) -> int:
    return (x + a - 1) // a * a


@dataclass
class Layout:
    """Assigned HBM offsets for one quantized projection."""
    proj: graph.Projection
    wq_hbm: int
    scale_hbm: int | None
    weight_bytes: bytes
    scale_bytes: bytes
    sw: np.ndarray = field(repr=False)


def _load_weight(proj: graph.Projection, st) -> np.ndarray:
    """Load a projection's fp32 weight; fuse q/k/v for qkv."""
    parts = [st.get_float32(k) for k in proj.weight_keys]
    if len(parts) == 1:
        w = parts[0]
    else:
        w = np.concatenate(parts, axis=0)   # qkv: stack q/k/v rows -> [4096, K]
    assert tuple(w.shape) == (proj.N, proj.K), \
        f"{proj.name}: weight {w.shape} != {(proj.N, proj.K)}"
    return w


def quantize_projections(projs: list[graph.Projection], st,
                         activation_scale: float,
                         wmode: str = "W8A8") -> list[Layout]:
    """Quantize every projection (W8A8 or W4A16) or pack raw BF16 (BF16 mode)
    and assign sequential HBM offsets. W4A16: INT4 weights (packed 2-per-byte),
    BF16 activations (activation_scale is 1.0 — no activation quantization).
    BF16: fp32 weights cast to BF16 (row-major, 64B aligned); scales omitted
    (scale_hbm=None, scale_bytes empty)."""
    assert wmode in ("W8A8", "W4A16", "BF16")
    layouts = []
    w_cursor = WEIGHT_BASE
    # first pass: quantize (or cast) + measure
    for p in projs:
        w = _load_weight(p, st)
        if wmode == "BF16":
            wbytes = np.asarray(w, dtype=_BF16).tobytes()
            layouts.append(Layout(proj=p, wq_hbm=0, scale_hbm=None,
                                  weight_bytes=wbytes, scale_bytes=b"",
                                  sw=None))
        elif wmode == "W4A16":
            wqi, sw = quant.quantize_weight_int4(w)
            wbytes = quant.pack_int4(wqi).tobytes()
            cd = quant.combine_scale(sw, activation_scale)
            layouts.append(Layout(proj=p, wq_hbm=0, scale_hbm=0,
                                  weight_bytes=wbytes,
                                  scale_bytes=cd.tobytes(), sw=sw))
        else:
            wqi, sw = quant.quantize_weight(w)
            wbytes = wqi.tobytes()
            cd = quant.combine_scale(sw, activation_scale)
            layouts.append(Layout(proj=p, wq_hbm=0, scale_hbm=0,
                                  weight_bytes=wbytes,
                                  scale_bytes=cd.tobytes(), sw=sw))
    # assign weight offsets (ascending, non-overlapping)
    for lay in layouts:
        lay.wq_hbm = _align(w_cursor)
        w_cursor = lay.wq_hbm + len(lay.weight_bytes)
    if wmode == "BF16":
        return layouts
    # scale offsets filled after weights (quantized modes only)
    s_cursor = _align(w_cursor)
    for lay in layouts:
        lay.scale_hbm = _align(s_cursor)
        s_cursor = lay.scale_hbm + len(lay.scale_bytes)
    return layouts


def _tensors(layouts: list[Layout], wmode: str = "W8A8") -> list[Tensor]:
    dtype = {"W4A16": "INT4", "W8A8": "INT8", "BF16": "BF16"}[wmode]
    out = []
    for lay in layouts:
        p = lay.proj
        if wmode == "BF16":
            out.append(Tensor(name=p.name, shape=[p.N, p.K], dtype=dtype,
                              hbm_off=lay.wq_hbm, data=lay.weight_bytes))
        else:
            out.append(Tensor(name=p.name, shape=[p.N, p.K], dtype=dtype,
                              hbm_off=lay.wq_hbm, data=lay.weight_bytes,
                              scales_hbm_off=lay.scale_hbm,
                              scales=lay.scale_bytes, scale_dtype="BF16"))
    return out


def build_model_qbin(path: str, st, quant_j: dict, activation_scale: float):
    """Full 0.6B transformer qbin (PF + DC full programs). Returns read_qbin(path).

    BF16 mode: weights stored as raw BF16 with scales omitted and empty PF/DC
    programs (qrun regenerates the BF16 programs at load time; the container is
    weight-self-contained)."""
    wmode = quant_j.get("mode", "W8A8")
    projs = graph.build_graph()
    layouts = quantize_projections(projs, st, activation_scale, wmode)
    tensors = _tensors(layouts, wmode)
    if wmode == "BF16":
        pf = b""
        dc = b""
    else:
        pf = program.lower_transformer("PF", layouts, wmode)
        dc = program.lower_transformer("DC", layouts, wmode)
    cfg = dict(C.MODEL_CFG)
    write_qbin(path, cfg["model"], cfg, quant_j, tensors, pf, dc,
               flags=_flags(quant_j))
    return read_qbin(path)


def build_projection_qbin(path: str, kind: str, w: np.ndarray, x: np.ndarray,
                          mode: str):
    """Single-projection INT8 qbin for M3 per-class verification.

    Quantizes w (weight side, per-128-K-group) and x (activation side,
    per-tensor, folding sx into the stored CD scale), builds a PF/DC qbin, and
    returns (qbin, xq, cd, wqi, input_hbm, output_hbm) so the caller can run the
    executor and build the M2a dequant reference with the exact stored scales.
    HBM addresses are auto-laid-out (weight -> scale -> input -> output) to stay
    collision-free for any tensor size (lm_head weight is 155 MB).

    w: [N,K] fp32. x: [M,K] fp32. mode: 'PF' | 'DC'.
    """
    N, K = w.shape
    M = x.shape[0]
    G = K // 128
    wq_hbm = L.WQ_HBM
    scale_hbm = _align(wq_hbm + N * K)
    input_hbm = _align(scale_hbm + N * G * 2)
    output_hbm = _align(input_hbm + M * K)

    wqi, sw = quant.quantize_weight(w)
    xq, sx = quant.quantize_activation(x)
    cd = quant.combine_scale(sw, sx)
    t = Tensor(name=f"{kind}_proj.weight", shape=[N, K], dtype="INT8",
               hbm_off=wq_hbm, data=wqi.tobytes(),
               scales_hbm_off=scale_hbm, scales=cd.tobytes(), scale_dtype="BF16")
    plan = L.lower_projection(x.shape, (N, K), mode, wq_hbm=wq_hbm,
                              scale_hbm=scale_hbm, input_hbm=input_hbm,
                              output_hbm=output_hbm)
    prog = L.encode_program(plan)
    cfg = dict(C.MODEL_CFG)
    write_qbin(path, cfg["model"], cfg, C.QUANT_INT8, [t],
               prog if mode == "PF" else b"",
               prog if mode == "DC" else b"", flags=_flags(C.QUANT_INT8))
    return read_qbin(path), xq, cd, wqi, input_hbm, output_hbm


def build_projection_qbin_int4(path: str, kind: str, w: np.ndarray,
                               x: np.ndarray, mode: str):
    """Single-projection W4A16 qbin (INT4 weight, BF16 activation) for M4 INT4
    per-class verification. Returns (qbin, cd, wqi, sw, input_hbm, output_hbm).

    cd: [N,G] BF16 per-group scale (activation scale = 1.0, no folding).
    wqi: [N,K] int8 (INT4 values -7..7). sw: [N,G] fp32 scales.
    Weight is stored packed (2-per-byte) in the qbin; activation stays BF16.
    """
    N, K = w.shape
    M = x.shape[0]
    G = K // 128
    wbytes = N * (K // 2)           # packed INT4
    wq_hbm = L.WQ_HBM
    scale_hbm = _align(wq_hbm + wbytes)
    input_hbm = _align(scale_hbm + N * G * 2)
    output_hbm = _align(input_hbm + M * K * 2)

    wqi, sw = quant.quantize_weight_int4(w)
    cd = quant.combine_scale(sw, 1.0)          # BF16 scale (sx = 1)
    packed = quant.pack_int4(wqi)
    t = Tensor(name=f"{kind}_proj.weight", shape=[N, K], dtype="INT4",
               hbm_off=wq_hbm, data=packed.tobytes(),
               scales_hbm_off=scale_hbm, scales=cd.tobytes(), scale_dtype="BF16")
    plan = L.lower_projection(x.shape, (N, K), mode, wq_hbm=wq_hbm,
                              scale_hbm=scale_hbm, input_hbm=input_hbm,
                              output_hbm=output_hbm, wmode="W4A16")
    prog = L.encode_program(plan)
    cfg = dict(C.MODEL_CFG)
    write_qbin(path, cfg["model"], cfg, C.QUANT_INT4, [t],
               prog if mode == "PF" else b"",
               prog if mode == "DC" else b"", flags=_flags(C.QUANT_INT4))
    return read_qbin(path), cd, wqi, sw, input_hbm, output_hbm
