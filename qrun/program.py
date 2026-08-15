"""qrun program generator — corrected full-model PF/DC lowering.

Reuses the frozen model card / SRAM layout / Em emitter / online-softmax pieces
of `qforge/program.py`, but fixes the wave-1 issues that block M4 end-to-end
numerical correctness:

  PF (M=128) layout fixes (the wave-1 PF was structurally checked, not numeric):
    * linear output is written **directly** by the GEMM/GEMV with the full
      token-major row stride (`C_CC = N*2`, `arc = dst + t*128*2`), not the
      buggy `dst + t*256` contiguous VMOV that overlapped tiles for M>1;
    * QKV is emitted as **three separate linears** (Q->b[q], K->b[k], V->b[v],
      each token-major with its own stride), not one fused [M,4096] blob whose
      K/V fell outside the k/v buffers;
    * KV is written with **per-token KV.APPEND** (C_KV_POS=t) instead of
      KV.STORE_BLOCK, because STORE_BLOCK needs head-major K/V while the linear
      produces token-major;
    * the attention AV GEMM writes ctx **directly** with the ctx token stride
      (C_CC = QH*HD*2) instead of a contiguous VMOV that produced head-major
      ctx while the O projection reads token-major;
    * per-token ROPE (pos 0..M-1) and per-token normal-mode RMSNorm (one rms
      per token), not one flattened op over M*1024;
    * PF attention reads Q with the correct token stride (C_CA = QH*HD*2).

  DC fixes:
    * decode window = [0, pos] (pos+1 incl. the current token); 4 KV.LOAD tiles
      cover [0,8192) and the current token (position 8192, unreachable by the
      13-bit KV.LOAD pos_start) is VMOV'd from the post-RoPE SRAM K/V buffers;
    * a runtime-written per-subtile BF16 tail mask (0 in-window, -inf
      out-of-window) is VADD'd onto scores before the online softmax.

  HBM addresses are parameters (no v0 placeholders); RMSNorm gamma points at
  the per-norm gamma region injected by qrun. dtype in {"int8","int4","bf16"}.
"""
from __future__ import annotations

from dataclasses import dataclass

from compiler.isa import isa as I
from qforge.program import (  # noqa: F401  (structural reuse)
    ATTN_SCALE, BLOCK, C_ACT, C_ATTN, C_BROADCAST, C_CA, C_CB, C_CC,
    C_CD, C_DMA, C_EPS, C_KV_POS, C_MASK, C_SLAB_SHIFT, C_THETA, GQA, H, HD,
    INT, KVH, KV_TILE, KSTAGE_SRAM, LAYERS, N_TILE, QH, RMS_EPS, ROPE_THETA,
    VSTAGE_SRAM, VOCAB, AR_ACT8, AR_ACT_SCALE, AR_ALPHA, AR_CTX, AR_CTX_ACC,
    AR_DOWN, AR_DST, AR_ES, AR_EXP, AR_GATE, AR_H, AR_IN_HBM, AR_K, AR_KG,
    AR_KSTAGE, AR_KVBASE, AR_LRUN, AR_MASK, AR_MNEW, AR_MRUN, AR_O, AR_ONES,
    AR_ONESROW, AR_OUTT, AR_OUT_HBM, AR_Q, AR_QG, AR_RINV, AR_SC0, AR_SC1,
    AR_SC2, AR_SCALE, AR_SCALE_HBM, AR_SCORES, AR_STILE, AR_UP, AR_V, AR_VG,
    AR_VSTAGE, AR_WB, AR_X, AR_XN, Em, _addr, _cd, _f32bits, _layout, _sram,
    _stride, _emit_residual, _emit_softmax_pf, _softmax_head, MAX_VLEN,
)
GAMMA_BASE = 0x700000
GAMMA_TOTAL = LAYERS * (H + QH * HD + KVH * HD + H) * 2 + H * 2  # 288768 B
N_SUBTILES = 65            # 64 (4 KV.LOAD tiles x 16) + 1 (current token)
N_PROJ = LAYERS * 5 + 1    # 28 layers x 5 projections + lm_head = 141

MASK_BASE = 0x748000       # per-subtile BF16 tail mask [65, 2, 128] = 65 x 512 B
MASK_TOTAL = N_SUBTILES * (2 * N_TILE) * 2   # 0x8200 B
ACT_BASE = 0x768000        # per-projection per-128-group activation-scale region
# per-projection [K//128] BF16 scales: 28*(16+32+16+16+48) + 16 = 3600 B
ACT_TOTAL = 3600

# F1: the DC tail mask region and the per-projection activation-scale region
# must be disjoint and both must fit inside 8 MiB SRAM (0x800000). The old
# MASK_BASE=0x760000 put the 65th subtile (0x768000..0x768200) on top of the
# first 32 activation scales, corrupting them to -inf on every DC step.
assert MASK_BASE % 16 == 0 and ACT_BASE % 16 == 0
assert MASK_BASE + MASK_TOTAL <= ACT_BASE or ACT_BASE + ACT_TOTAL <= MASK_BASE, \
    "MASK and ACT SRAM regions overlap"
assert MASK_BASE + MASK_TOTAL <= 0x800000 and ACT_BASE + ACT_TOTAL <= 0x800000, \
    "MASK/ACT SRAM regions exceed 8 MiB"


def gamma_layout() -> dict:
    """Map (kind, layer) -> SRAM byte address of the per-norm gamma buffer."""
    addrs: dict = {}
    off = GAMMA_BASE

    def alloc(key, nbytes):
        nonlocal off
        assert nbytes % 16 == 0
        addrs[key] = off
        off += nbytes

    for L in range(LAYERS):
        alloc(("input", L), H * 2)
        alloc(("q_norm", L), QH * HD * 2)
        alloc(("k_norm", L), KVH * HD * 2)
        alloc(("post", L), H * 2)
    alloc(("final", None), H * 2)
    assert off - GAMMA_BASE == GAMMA_TOTAL
    return addrs


def act_scale_layout() -> dict:
    """Per-projection per-128-group activation-scale SRAM byte addresses.

    Each projection gets `K//128` BF16 scales (16B-aligned).  Region follows
    ACT_BASE in graph order (qkv/o/gate/up/down x 28 + lm_head); total =
    28*(16+32+16+16+48) + 16 = 3600 B = ACT_TOTAL.
    """
    import qforge.graph as GG
    addrs: dict = {}
    off = ACT_BASE
    for p in GG.build_graph():
        nbytes = (p.K // 128) * 2
        assert nbytes % 16 == 0, (p.name, nbytes)
        addrs[p.name] = off
        off += nbytes
    assert off - ACT_BASE == ACT_TOTAL, (off - ACT_BASE, ACT_TOTAL)
    return addrs


@dataclass
class Layout:
    proj: object            # graph.Projection (.layer / .kind / .N / .K)
    wq_hbm: int
    scale_hbm: int | None = None   # INT8 only
    act_sram: int = 0              # per-128-group activation-scale SRAM word addr


def _emit_rmsnorm_qr(em: Em, b: dict, M: int, L: int, src_ar: int, dst_ar: int,
                     mode: int, gamma_addr: int):
    """RMSNorm with per-token emission (PF) / single (DC). gamma = per-norm."""
    for t in range(M):
        em.cfg_ar(AR_SC0, _addr(b, src_ar) + t * L * 2)
        em.cfg_ar(AR_SC1, gamma_addr)
        em.cfg_ar(AR_SC2, _addr(b, dst_ar) + t * L * 2)
        em.i("RMSNORM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_SC1, ard=AR_SC2, len=L, cv=C_EPS,
             imm=mode << 31)


def _emit_rope_qr(em: Em, b: dict, M: int, L: int, ar: int, first_pos: int):
    """ROPE with per-token position (PF) / single (DC, runtime patches pos)."""
    for t in range(M):
        em.cfg_ar(AR_SC0, _addr(b, ar) + t * L * 2)
        em.cfg_ar(AR_SC1, _addr(b, AR_ONES))
        em.cfg_ar(AR_SC2, _addr(b, ar) + t * L * 2)
        em.i("ROPE", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_SC1, ard=AR_SC2, len=L, cv=C_THETA,
             imm=(first_pos + t) & 0xFFFF)


def _emit_prologue_qr(em: Em, M: int, b: dict, hbm, slab_shift: int):
    em.i("MODE", mode=1 if M == 1 else 0)
    em.cfg_ar(AR_KVBASE, hbm.kv_base, hbm=True)
    em.cfg_c(C_KV_POS, 0)
    em.cfg_c(C_SLAB_SHIFT, slab_shift)
    for name, ar in _AR_NAME_MAP.items():
        em.cfg_ar(ar, b[name])
    em.cfg_ar(AR_KSTAGE, KSTAGE_SRAM)
    em.cfg_ar(AR_VSTAGE, VSTAGE_SRAM)
    em.cfg_ar(AR_IN_HBM, hbm.input_hbm, hbm=True)
    em.cfg_ar(AR_OUT_HBM, hbm.logits_hbm, hbm=True)
    em.cfg_c(C_EPS, _f32bits(RMS_EPS))
    em.cfg_c(C_THETA, _f32bits(ROPE_THETA))
    em.cfg_c(C_ACT, (0 << 20) | (0 << 19) | _sram(b["act_scale"]))
    em.cfg_c(C_ATTN, _f32bits(ATTN_SCALE))
    if M > 1:
        em.i("VSUB", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_ONESROW, arb=AR_ONESROW, ard=AR_ONESROW, len=BLOCK, cv=0,
             imm=0)
        em.i("VEXP", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_ONESROW, arb=AR_ONES, ard=AR_ONESROW, len=BLOCK, cv=0,
             imm=0)


def _emit_quant_group(em: Em, b: dict, M: int, K: int, src_ar: int):
    """QUANT BF16 -> INT8, per-128-group activation scale (04 §1.5 / 02 §6).

    One QUANT per token row so each row reads the same [K//128] per-group scale
    array from SRAM (n=K is a multiple of 128, gcnt=K//128 aligns to the group
    boundaries).  CV = C_ACT carries the per-128-group scale descriptor
    (mode=1) set by the caller before this emission.
    """
    for t in range(M):
        em.cfg_ar(AR_SC0, _addr(b, src_ar) + t * K * 2)
        em.cfg_ar(AR_SC2, _addr(b, AR_ACT8) + t * K)
        em.i("QUANT", srcA=I.DT_BF16, acc=0,
             ara=AR_SC0, arb=AR_ONES, ard=AR_SC2, len=K, cv=C_ACT, imm=0)
        em.barrier()


def _emit_linear_int8(em: Em, M: int, N: int, K: int, src_ar: int, dst_ar: int,
                      wq_hbm: int, scale_hbm: int, b: dict, act_sram: int = 0):
    mn = "GEMV" if M == 1 else "GEMM"
    G = K // 128
    ntiles = N // N_TILE
    scale_tile_bytes = N_TILE * G * 2
    em.cfg_c(C_CA, _stride(K))
    em.cfg_c(C_CB, _stride(K))
    em.cfg_c(C_CC, _stride(N * 2))
    em.cfg_c(C_CD, _cd(_sram(b["scale"])))
    if act_sram:
        em.cfg_c(C_ACT, (1 << 20) | (0 << 19) | act_sram)
    _emit_quant_group(em, b, M, K, src_ar)
    for t in range(ntiles):
        em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * K, hbm=True)
        em.cfg_ar(AR_SCALE_HBM, scale_hbm + t * scale_tile_bytes, hbm=True)
        em.i("DMA.LOAD", srcA=I.DT_BF16,
             SrcAR=AR_SCALE_HBM, DstAR=AR_SCALE, RowBytes=scale_tile_bytes,
             NumRows=1, StrideC=0, mode=0)
        em.cfg_ar(AR_OUTT, _addr(b, dst_ar) + t * N_TILE * 2)
        em.i(mn, srcA=I.DT_INT8, srcB=I.DT_INT8, acc=I.ACC_INT32,
             ara=AR_ACT8, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=K, batch=1,
             ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD,
             acc_init=1, bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
        em.barrier()


def _emit_linear_int4(em: Em, M: int, N: int, K: int, src_ar: int, dst_ar: int,
                      wq_hbm: int, scale_hbm: int, b: dict):
    """W4A16 linear: srcA=BF16 (no activation quant), srcB=INT4, acc=FP32,
    dequant=1 per-128-group scale (02 §6 / 04 §1.2). No runtime QUANT/DEQUANT."""
    mn = "GEMV" if M == 1 else "GEMM"
    G = K // 128
    ntiles = N // N_TILE
    scale_tile_bytes = N_TILE * G * 2
    em.cfg_c(C_CA, _stride(K * 2))       # BF16 activation row
    em.cfg_c(C_CB, _stride(K // 2))      # INT4 weight row (2/byte)
    em.cfg_c(C_CC, _stride(N * 2))       # BF16 output row
    em.cfg_c(C_CD, _cd(_sram(b["scale"])))
    for t in range(ntiles):
        em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * (K // 2), hbm=True)
        em.cfg_ar(AR_SCALE_HBM, scale_hbm + t * scale_tile_bytes, hbm=True)
        em.i("DMA.LOAD", srcA=I.DT_BF16,
             SrcAR=AR_SCALE_HBM, DstAR=AR_SCALE, RowBytes=scale_tile_bytes,
             NumRows=1, StrideC=0, mode=0)
        em.cfg_ar(AR_OUTT, _addr(b, dst_ar) + t * N_TILE * 2)
        em.i(mn, srcA=I.DT_BF16, srcB=I.DT_INT4, acc=I.ACC_FP32,
             ara=src_ar, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=K, batch=1,
             ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD,
             acc_init=1, bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
        em.barrier()

def _emit_linear_bf16(em: Em, M: int, N: int, K: int, src_ar: int, dst_ar: int,
                      wq_hbm: int, b: dict):
    mn = "GEMV" if M == 1 else "GEMM"
    ntiles = N // N_TILE
    em.cfg_c(C_CA, _stride(K * 2))
    em.cfg_c(C_CB, _stride(K * 2))
    em.cfg_c(C_CC, _stride(N * 2))
    for t in range(ntiles):
        em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * K * 2, hbm=True)
        em.cfg_ar(AR_OUTT, _addr(b, dst_ar) + t * N_TILE * 2)
        em.i(mn, srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=src_ar, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=K, batch=1,
             ca=C_CA, cb=C_CB, cc=C_CC, cd=0,
             acc_init=1, bsrc=1, dequant=0, transpose_a=0, transpose_b=1)
        em.barrier()
def _emit_linear(em: Em, M: int, N: int, K: int, src_ar: int, dst_ar: int,
                 wq_hbm: int, scale_hbm: int | None, b: dict, dtype: str,
                 act_sram: int = 0):
    if dtype == "int8":
        _emit_linear_int8(em, M, N, K, src_ar, dst_ar, wq_hbm, scale_hbm, b,
                          act_sram)
    elif dtype == "int4":
        _emit_linear_int4(em, M, N, K, src_ar, dst_ar, wq_hbm, scale_hbm, b)
    else:
        _emit_linear_bf16(em, M, N, K, src_ar, dst_ar, wq_hbm, b)


def _emit_qkv(em: Em, M: int, src_ar: int, qkv: Layout, b: dict, dtype: str):
    esz = 1 if dtype == "int8" else (0.5 if dtype == "int4" else 2)
    H_bytes = int(H * esz)
    _emit_linear(em, M, QH * HD, H, src_ar, AR_Q, qkv.wq_hbm,
                 qkv.scale_hbm, b, dtype, qkv.act_sram)
    _emit_linear(em, M, KVH * HD, H, src_ar, AR_K,
                 qkv.wq_hbm + QH * HD * H_bytes,
                 None if qkv.scale_hbm is None
                 else qkv.scale_hbm + QH * HD * (H // 128) * 2, b, dtype,
                 qkv.act_sram)
    _emit_linear(em, M, KVH * HD, H, src_ar, AR_V,
                 qkv.wq_hbm + (QH * HD + KVH * HD) * H_bytes,
                 None if qkv.scale_hbm is None
                 else qkv.scale_hbm + (QH * HD + KVH * HD) * (H // 128) * 2,
                 b, dtype, qkv.act_sram)


def _emit_subtile(em: Em, b: dict, first: bool, mask_addr: int):
    em.i("BMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_QG, arb=AR_KSTAGE, arc=AR_SCORES, m=1, n=N_TILE,
         k=HD, batch=GQA, ca=C_CA, cb=C_CB, cc=C_CC, cd=0,
         acc_init=1, bsrc=0, dequant=0, transpose_a=0, transpose_b=1)
    em.barrier()
    em.i("VSCALE", srcA=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_SCORES, arb=AR_ONES, ard=AR_SCORES, len=2 * N_TILE,
         cv=C_ATTN, imm=0)
    em.cfg_ar(AR_SC1, mask_addr)
    em.i("VADD", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_SCORES, arb=AR_SC1, ard=AR_SCORES, len=2 * N_TILE, cv=0, imm=0)
    _softmax_head(em, 0, N_TILE, first, b)
    _softmax_head(em, 1, N_TILE, first, b)
    em.i("BMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_ES, arb=AR_VSTAGE, arc=AR_CTX_ACC, m=1, n=HD,
         k=N_TILE, batch=GQA, ca=C_CA, cb=C_CB, cc=C_CC, cd=0,
         acc_init=1 if first else 0, bsrc=0, dequant=0,
         transpose_a=0, transpose_b=0)
    em.barrier()


def _emit_attention_dc_qr(em: Em, layer: int, b: dict, mask_base: int):
    q_stride = HD * 2
    ctx_stride = HD * 2
    score_stride = N_TILE * 2
    em.cfg_c(C_CA, _stride(q_stride, q_stride))
    em.cfg_c(C_CB, _stride(HD * 2, 0))
    em.cfg_c(C_CC, _stride(score_stride, score_stride))

    for g in range(KVH):
        em.cfg_ar(AR_QG, b["q"] + 2 * g * q_stride)
        first = True
        sub_idx = 0
        for kt in range(4):
            pos_start = kt * KV_TILE
            em.cfg_ar(AR_KSTAGE, KSTAGE_SRAM)
            em.cfg_ar(AR_VSTAGE, VSTAGE_SRAM)
            em.i("KV.LOAD", dstK=AR_KSTAGE, dstV=AR_VSTAGE, layer=layer,
                 head=g, sel=2, pos_start=pos_start, count=KV_TILE)
            em.wait(8)
            for _st in range(KV_TILE // N_TILE):
                em.cfg_ar(AR_KSTAGE, KSTAGE_SRAM + _st * N_TILE * HD * 2)
                em.cfg_ar(AR_VSTAGE, VSTAGE_SRAM + _st * N_TILE * HD * 2)
                _emit_subtile(em, b, first, mask_base + sub_idx * (2 * N_TILE) * 2)
                first = False
                sub_idx += 1
        em.cfg_ar(AR_KSTAGE, KSTAGE_SRAM)
        em.cfg_ar(AR_VSTAGE, VSTAGE_SRAM)
        em.cfg_ar(AR_KG, b["k"] + g * HD * 2)
        em.cfg_ar(AR_VG, b["v"] + g * HD * 2)
        em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_KG, arb=AR_ONES, ard=AR_KSTAGE, len=HD, cv=0, imm=0)
        em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_VG, arb=AR_ONES, ard=AR_VSTAGE, len=HD, cv=0, imm=0)
        _emit_subtile(em, b, first, mask_base + sub_idx * (2 * N_TILE) * 2)
        sub_idx += 1
        for hh in range(GQA):
            ch = b["ctx_acc"] + hh * HD * 2
            lh = b["lrun"] + hh * 16
            rh = b["rinv"] + hh * 16
            em.cfg_ar(AR_LRUN, lh)
            em.cfg_ar(AR_RINV, rh)
            em.cfg_ar(AR_SC0, ch)
            em.cfg_ar(AR_SC1, b["ctx"] + (2 * g + hh) * ctx_stride)
            em.i("VRECIP", srcA=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_LRUN, arb=AR_ONES, ard=AR_RINV, len=1, cv=0, imm=0)
            em.i("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_SC0, arb=AR_RINV, ard=AR_SC0, len=HD, cv=C_BROADCAST,
                 imm=0)
            em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_SC0, arb=AR_ONES, ard=AR_SC1, len=HD, cv=0, imm=0)
        em.barrier()


def _emit_attention_pf_qr(em: Em, layer: int, b: dict):
    """Prefill attention (single 128-token block, window=128), token-major Q
    and ctx. Q token stride = QH*HD*2; ctx token stride = QH*HD*2."""
    em.i("VMASK", srcA=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_ONES, arb=AR_ONES, ard=AR_MASK, len=0, cv=C_MASK,
         imm=(BLOCK << 16) | BLOCK)
    for g in range(KVH):
        em.i("KV.LOAD", dstK=AR_KSTAGE, dstV=AR_VSTAGE, layer=layer,
             head=g, sel=2, pos_start=0, count=BLOCK)
        em.wait(8)
        for q in range(GQA):
            qhead = g * GQA + q
            em.cfg_ar(AR_QG, b["q"] + qhead * HD * 2)
            em.cfg_c(C_CA, _stride(QH * HD * 2, 0))
            em.cfg_c(C_CB, _stride(HD * 2, 0))
            em.cfg_c(C_CC, _stride(BLOCK * 2, 0))
            em.i("GEMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_QG, arb=AR_KSTAGE, arc=AR_SCORES, m=BLOCK, n=BLOCK,
                 k=HD, batch=1, ca=C_CA, cb=C_CB, cc=C_CC, cd=0,
                 acc_init=1, bsrc=0, dequant=0, transpose_a=0, transpose_b=1)
            em.barrier()
            em.i("VSCALE", srcA=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_SCORES, arb=AR_ONES, ard=AR_SCORES, len=BLOCK * BLOCK,
                 cv=C_ATTN, imm=0)
            em.i("VADD", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_SCORES, arb=AR_MASK, ard=AR_SCORES, len=BLOCK * BLOCK,
                 cv=0, imm=0)
            _emit_softmax_pf(em, b)
            em.cfg_c(C_CA, _stride(BLOCK * 2, 0))
            em.cfg_c(C_CB, _stride(HD * 2, 0))
            em.cfg_c(C_CC, _stride(QH * HD * 2, 0))   # ctx token stride
            em.cfg_ar(AR_CTX_ACC, b["ctx"] + qhead * HD * 2)
            em.i("GEMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_ES, arb=AR_VSTAGE, arc=AR_CTX_ACC, m=BLOCK, n=HD,
                 k=BLOCK, batch=1, ca=C_CA, cb=C_CB, cc=C_CC, cd=0,
                 acc_init=1, bsrc=0, dequant=0, transpose_a=0, transpose_b=0)
            em.barrier()


def _emit_lm_head_qr(em: Em, M: int, b: dict, wq_hbm: int, scale_hbm: int | None,
                     dtype: str, hbm, gamma_final: int, act_sram: int = 0):
    _emit_rmsnorm_qr(em, b, M, H, AR_X, AR_XN, 0, gamma_final)
    ntiles = VOCAB // N_TILE
    mn = "GEMV" if M == 1 else "GEMM"
    if dtype == "int8":
        G = H // 128
        scale_tile_bytes = N_TILE * G * 2
        em.cfg_c(C_CA, _stride(H))
        em.cfg_c(C_CB, _stride(H))
        em.cfg_c(C_CC, _stride(N_TILE * 2))
        em.cfg_c(C_CD, _cd(_sram(b["scale"])))
        if act_sram:
            em.cfg_c(C_ACT, (1 << 20) | (0 << 19) | act_sram)
        _emit_quant_group(em, b, M, H, AR_XN)
        for t in range(ntiles):
            em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * H, hbm=True)
            em.cfg_ar(AR_SCALE_HBM, scale_hbm + t * scale_tile_bytes, hbm=True)
            em.i("DMA.LOAD", srcA=I.DT_BF16, SrcAR=AR_SCALE_HBM,
                 DstAR=AR_SCALE, RowBytes=scale_tile_bytes, NumRows=1,
                 StrideC=0, mode=0)
            em.i(mn, srcA=I.DT_INT8, srcB=I.DT_INT8, acc=I.ACC_INT32,
                 ara=AR_ACT8, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=H,
                 batch=1, ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD, acc_init=1,
                 bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
            em.barrier()
            em.cfg_ar(AR_DST, b["out_tile"])
            em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32, ara=AR_OUTT,
                 arb=AR_ONES, ard=AR_DST, len=M * N_TILE, cv=0, imm=0)
            em.barrier()
            em.cfg_ar(AR_OUT_HBM, hbm.logits_hbm + t * M * N_TILE * 2, hbm=True)
            em.i("DMA.STORE", srcA=I.DT_BF16, SrcAR=AR_DST, DstAR=AR_OUT_HBM,
                 RowBytes=M * N_TILE * 2, NumRows=1, StrideC=0, mode=0)
    elif dtype == "int4":
        G = H // 128
        scale_tile_bytes = N_TILE * G * 2
        em.cfg_c(C_CA, _stride(H * 2))
        em.cfg_c(C_CB, _stride(H // 2))
        em.cfg_c(C_CC, _stride(N_TILE * 2))
        em.cfg_c(C_CD, _cd(_sram(b["scale"])))
        for t in range(ntiles):
            em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * (H // 2), hbm=True)
            em.cfg_ar(AR_SCALE_HBM, scale_hbm + t * scale_tile_bytes, hbm=True)
            em.i("DMA.LOAD", srcA=I.DT_BF16, SrcAR=AR_SCALE_HBM,
                 DstAR=AR_SCALE, RowBytes=scale_tile_bytes, NumRows=1,
                 StrideC=0, mode=0)
            em.i(mn, srcA=I.DT_BF16, srcB=I.DT_INT4, acc=I.ACC_FP32,
                 ara=AR_XN, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=H,
                 batch=1, ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD, acc_init=1,
                 bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
            em.barrier()
            em.cfg_ar(AR_DST, b["out_tile"])
            em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32, ara=AR_OUTT,
                 arb=AR_ONES, ard=AR_DST, len=M * N_TILE, cv=0, imm=0)
            em.barrier()
            em.cfg_ar(AR_OUT_HBM, hbm.logits_hbm + t * M * N_TILE * 2, hbm=True)
            em.i("DMA.STORE", srcA=I.DT_BF16, SrcAR=AR_DST, DstAR=AR_OUT_HBM,
                 RowBytes=M * N_TILE * 2, NumRows=1, StrideC=0, mode=0)
    else:
        em.cfg_c(C_CA, _stride(H * 2))
        em.cfg_c(C_CB, _stride(H * 2))
        em.cfg_c(C_CC, _stride(N_TILE * 2))
        for t in range(ntiles):
            em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * H * 2, hbm=True)
            em.i(mn, srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_XN, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=H,
                 batch=1, ca=C_CA, cb=C_CB, cc=C_CC, cd=0, acc_init=1,
                 bsrc=1, dequant=0, transpose_a=0, transpose_b=1)
            em.barrier()
            em.cfg_ar(AR_DST, b["out_tile"])
            em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32, ara=AR_OUTT,
                 arb=AR_ONES, ard=AR_DST, len=M * N_TILE, cv=0, imm=0)
            em.barrier()
            em.cfg_ar(AR_OUT_HBM, hbm.logits_hbm + t * M * N_TILE * 2, hbm=True)
            em.i("DMA.STORE", srcA=I.DT_BF16, SrcAR=AR_DST, DstAR=AR_OUT_HBM,
                 RowBytes=M * N_TILE * 2, NumRows=1, StrideC=0, mode=0)
    em.barrier()


def _tvec_qr(em: Em, b: dict, mn: str, a_ar: int, b_ar: int, d_ar: int,
             total: int):
    off = 0
    while off < total:
        n = min(MAX_VLEN, total - off)
        em.cfg_ar(AR_SC0, _addr(b, a_ar) + off * 2)
        em.cfg_ar(AR_SC1, _addr(b, b_ar) + off * 2)
        em.cfg_ar(AR_SC2, _addr(b, d_ar) + off * 2)
        em.i(mn, srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_SC1, ard=AR_SC2, len=n, cv=0, imm=0)
        off += n


def lower_transformer(mode: str, layouts: list[Layout], dtype: str, hbm,
                      slab_shift: int = 22, n_layers: int = 28) -> bytes:
    assert mode in ("PF", "DC") and dtype in ("int8", "int4", "bf16")
    assert 1 <= n_layers <= 28
    M = BLOCK if mode == "PF" else 1
    b = _layout(M)
    gaddrs = gamma_layout()
    em = Em()
    _emit_prologue_qr(em, M, b, hbm, slab_shift)

    by_layer: dict = {}
    lm = None
    for lay in layouts:
        p = lay.proj
        if p.layer is None:
            lm = lay
        else:
            by_layer.setdefault(p.layer, {})[p.kind] = lay

    em.cfg_c(C_DMA, H * 2)
    em.i("DMA.LOAD", srcA=I.DT_BF16, SrcAR=AR_IN_HBM, DstAR=AR_X,
         RowBytes=H * 2, NumRows=M, StrideC=C_DMA, mode=1)
    em.barrier()

    for L in range(n_layers):
        lp = by_layer[L]
        _emit_rmsnorm_qr(em, b, M, H, AR_X, AR_XN, 0, gaddrs[("input", L)])
        _emit_qkv(em, M, AR_XN, lp["qkv"], b, dtype)
        _emit_rmsnorm_qr(em, b, M, QH * HD, AR_Q, AR_Q, 1,
                         gaddrs[("q_norm", L)])
        _emit_rmsnorm_qr(em, b, M, KVH * HD, AR_K, AR_K, 1,
                         gaddrs[("k_norm", L)])
        _emit_rope_qr(em, b, M, QH * HD, AR_Q, 0)
        _emit_rope_qr(em, b, M, KVH * HD, AR_K, 0)
        if M == 1:
            for h in range(KVH):
                em.cfg_ar(AR_KG, b["k"] + h * HD * 2)
                em.cfg_ar(AR_VG, b["v"] + h * HD * 2)
                em.i("KV.APPEND", srcK=AR_KG, srcV=AR_VG, layer=L, head=h)
        else:
            for t in range(M):
                em.cfg_c(C_KV_POS, t)
                for h in range(KVH):
                    em.cfg_ar(AR_KG, b["k"] + t * KVH * HD * 2 + h * HD * 2)
                    em.cfg_ar(AR_VG, b["v"] + t * KVH * HD * 2 + h * HD * 2)
                    em.i("KV.APPEND", srcK=AR_KG, srcV=AR_VG, layer=L, head=h)
        em.barrier()
        if M == 1:
            _emit_attention_dc_qr(em, L, b, MASK_BASE)
        else:
            _emit_attention_pf_qr(em, L, b)
        _emit_linear(em, M, H, QH * HD, AR_CTX, AR_O,
                     lp["o"].wq_hbm, lp["o"].scale_hbm, b, dtype,
                     lp["o"].act_sram)
        _emit_residual(em, b, M, AR_O, AR_X)
        _emit_rmsnorm_qr(em, b, M, H, AR_X, AR_XN, 0, gaddrs[("post", L)])
        _emit_linear(em, M, INT, H, AR_XN, AR_GATE,
                     lp["gate"].wq_hbm, lp["gate"].scale_hbm, b, dtype,
                     lp["gate"].act_sram)
        _emit_linear(em, M, INT, H, AR_XN, AR_UP,
                     lp["up"].wq_hbm, lp["up"].scale_hbm, b, dtype,
                     lp["up"].act_sram)
        _tvec_qr(em, b, "VSILU", AR_GATE, AR_ONES, AR_H, M * INT)
        _tvec_qr(em, b, "VMUL", AR_H, AR_UP, AR_H, M * INT)
        em.barrier()
        _emit_linear(em, M, H, INT, AR_H, AR_DOWN,
                     lp["down"].wq_hbm, lp["down"].scale_hbm, b, dtype,
                     lp["down"].act_sram)
        _emit_residual(em, b, M, AR_DOWN, AR_X)

    _emit_lm_head_qr(em, M, b, lm.wq_hbm, lm.scale_hbm, dtype, hbm,
                     gaddrs[("final", None)], lm.act_sram)
    return em.encode()


_AR_NAME_MAP = {
    "x": AR_X, "xn": AR_XN, "q": AR_Q, "k": AR_K, "v": AR_V, "ctx": AR_CTX,
    "o": AR_O, "gate": AR_GATE, "up": AR_UP, "h": AR_H, "down": AR_DOWN,
    "act8": AR_ACT8, "scale": AR_SCALE, "out_tile": AR_OUTT,
    "scores": AR_SCORES, "es": AR_ES, "ctx_acc": AR_CTX_ACC,
    "mrun": AR_MRUN, "lrun": AR_LRUN, "mnew": AR_MNEW, "stile": AR_STILE,
    "alpha": AR_ALPHA, "rinv": AR_RINV, "ones": AR_ONES,
    "act_scale": AR_ACT_SCALE,
    "mask": AR_MASK, "expanded": AR_EXP, "ones_row": AR_ONESROW,
}
