"""qforge full-model transformer lowering (P5 / FullProgGen).

Generates the *complete* PF (prefill) and DC (decode) Q-ISA programs for
Qwen3-0.6B (28 layers), replacing the P4 linear-only skeleton.  This is the
"wave 1" structural deliverable: the programs encode the frozen 33-instruction
ISA faithfully and decode cleanly; the exact numerical semantics of the VECTOR
/ KV engines (scalar-broadcast VMUL/VSUB stride, QUANT scale descriptor layout,
VREDUCE grouping) are ExecVecKv's wave-1 sibling work, and full numeric
validation is deferred to wave 2 (EndToEndRuntime / M4).

One decode layer (02-isa §12 + 05-kv-cache §5.2):

    RMSNorm(x) -> QKV GEMV -> QK-norm(per-head q,k) -> ROPE(q,k)
    -> KV.APPEND (8 heads)
    -> per KV head g in [0,8):                    # GQA 2:1 -> 2 Q heads
         KV.LOAD(sel=both, tile<=2048)           # 4 tiles for the 8K window
         -> per score tile (N<=128, 64 tiles @8K):
              BMM QK^T (batch=2, batch_stride_B=0) -> scores
              -> N-tiling + online softmax (VREDUCE_MAX/SUM running, per-head)
              -> BMM AV (batch=2, batch_stride_B=0, acc_init=1 first tile)
    -> O GEMV -> residual(x += o)
    -> RMSNorm(x) -> gate/up GEMV -> VSILU -> VMUL -> down GEMV -> residual
    last layer: RMSNorm(x) -> lm_head GEMV -> DMA.STORE logits

PF is isomorphic (GEMM M=128, KV.STORE_BLOCK, window=128 for one block).

Dtype convention (documented, wave-2 numeric refinement):
  * weights: INT8 W8A8, per-128-group BF16 scales (141 tensors unchanged).
  * activations (x/q/k/v/ctx/gate/up/down/...): BF16 in SRAM.
  * linear: QUANT (BF16->INT8) -> GEMM/GEMV (INT8xINT8->INT32, dequant=1,
    CD = per-128-group sw x activation sx) -> VMOV (BF16 same-dtype copy).
  * attention BMM: srcA=srcB=BF16, acc=FP32, output BF16 (a BF16 matmul writes
    BF16 out per the unified "输出按 srcA dtype 落盘" rule).
  * vector ops: srcA=BF16, acc=FP32 (internal fp32, out = srcA dtype).

Needs-review items (P5) surfaced by this lowering:
  1. RMSNorm gamma (input_layernorm / post_attention_layernorm / q_norm /
     k_norm) weights (~85 small BF16 vectors) are not in the 141 projection
     tensors;
     the program points ARb at an all-ones vector.  Wave 2 must add these
     tensors to the qbin or inject them via qrun.
  2. VMOV dtype conversion (fp32 matrix-dequant output -> BF16): RESOLVED — the
     MATRIX engine writes BF16 for dequant=1 (04 §1.5 post-process), and VMOV is
     a same-dtype copy in v0 (no dtype conversion).
  3. QUANT per-tensor activation scale descriptor layout mirrors CD
     (02-isa §7.2 note); per-tensor scale sx is ACTIVATION_SCALE_DEFAULT (1.0),
     stored as one BF16 in SRAM.
  4. ROPE pos is an immediate (imm[15:0]); decode needs it to track C_KV_POS,
     so the runtime must patch ROPE imm (not only the C_KV_POS CONFIG) per
     token, or ROPE should read pos from C30 (wave-2 decision).
  5. VSUB/VMUL/VDIV/VMAX scalar broadcast: RESOLVED — cv field dispatches the
     binary-op ARb semantics (cv=0 -> ARb contiguous len elements; cv!=0 ->
     ARb[0] scalar broadcast, per 02-isa §7.2).  The lowering uses
     C_BROADCAST for every scalar-broadcast binary op; continuous ops stay cv=0.
  6. HBM addresses (INPUT_HBM / LOGITS_HBM / AR_KV_BASE) are v0 placeholders;
     qrun re-plans the final allocation (weights end ~579 MiB, so the logits
     and KV regions are placed above it).
"""

from __future__ import annotations

import numpy as np

from compiler.isa import isa as I

# -- frozen model card (01b) -------------------------------------------------
H = 1024
LAYERS = 28
QH = 16
KVH = 8
HD = 128
INT = 3072
VOCAB = 151936
GQA = QH // KVH                      # 2:1
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6
ACT_SCALE = 1.0                      # compile-time activation scale (sx)
ATTN_SCALE = 1.0 / np.sqrt(HD)      # attention score scale (128^-0.5)

# -- tiling ---------------------------------------------------------------
N_TILE = 128                         # scores / linear N tiling
KV_TILE = 2048                       # KV.LOAD tile (05 §1.5 single-copy max)
WINDOW = 8192                        # compile-time max decode window (8K)
BLOCK = 128                          # prefill block size

# -- HBM placeholders (qrun re-plans; see docstring item 6) ----------------
INPUT_HBM = 0x0100_0000              # hidden input (runtime writes)
LOGITS_HBM = 0x1000_0000             # logits output (256 MiB; runtime reads)
AR_KV_BASE = 0x4000_0000             # KV region base (1 GiB, 2 MiB aligned)

# -- DC decode tail-mask SRAM region (runtime-written; qrun mirrors this) ----
MASK_BASE = 0x760000                  # per-subtile BF16 tail mask [65, 2, 128]
N_SUBTILES = 65                       # 64 (4 KV.LOAD tiles x 16) + 1 (current token)

# -- AR register allocation (SRAM buffers, bit63=0) --------------------------
AR_X = 0
AR_XN = 1
AR_Q = 2
AR_K = 3
AR_V = 4
AR_CTX = 5
AR_O = 6
AR_GATE = 7
AR_UP = 8
AR_H = 9
AR_DOWN = 10
AR_ACT8 = 11
AR_SCALE = 12
AR_OUTT = 13                        # linear output tile (fp32)
AR_SCORES = 14                      # QK^T scores (BF16, 2 heads x N_TILE)
AR_ES = 15                          # exp(scores) (BF16)
AR_CTX_ACC = 16                     # per-group ctx accumulator (BF16, 2 x HD)
AR_MRUN = 17
AR_LRUN = 18
AR_MNEW = 19
AR_STILE = 20
AR_ALPHA = 21
AR_RINV = 22
AR_ONES = 23                        # all-ones gamma placeholder
AR_ACT_SCALE = 24                   # one BF16 activation scale (sx)
AR_KSTAGE = 25                      # KV staging K (SRAM 0x300000)
AR_VSTAGE = 26                      # KV staging V (SRAM 0x380000)
AR_KG = 27                          # per-head K source (re-CONFIG'd)
AR_VG = 28                          # per-head V source (re-CONFIG'd)
AR_QG = 29                          # per-group Q base (re-CONFIG'd)
AR_CTXG = 30                        # per-group ctx destination (re-CONFIG'd)
AR_WB = 31                          # per-tile weight HBM base (re-CONFIG'd)
AR_SCALE_HBM = 32                   # per-tile scale HBM base (re-CONFIG'd)
AR_DST = 33                         # per-tile BF16 destination (re-CONFIG'd)
AR_IN_HBM = 34                      # input HBM base
AR_OUT_HBM = 35                     # logits HBM base
AR_SC0 = 36                         # softmax scratch: head scores row
AR_SC1 = 37                         # softmax scratch: head exp row
AR_SC2 = 38                         # softmax scratch: head ctx row
AR_MASK = 39                        # PF causal mask tile (BLOCK x BLOCK)
AR_EXP = 40                         # PF softmax per-row scalar expansion (BLOCK x BLOCK)
AR_ONESROW = 41                     # PF softmax all-ones row (BLOCK elements)
AR_KVBASE = 63                      # frozen AR_KV_BASE

# -- C register allocation ---------------------------------------------------
C_CA = 0
C_CB = 1
C_CC = 2
C_CD = 3
C_ACT = 4                            # QUANT per-tensor scale descriptor
C_EPS = 5                            # rms eps (fp32 bits)
C_THETA = 6                          # rope theta (fp32 bits)
C_MASK = 8                           # VMASK row/col base
C_DMA = 9                            # DMA row stride
C_ATTN = 10                          # attention scale (fp32 bits, 128^-0.5)
C_BROADCAST = 11                     # cv!=0 -> ARb[0] scalar broadcast (binary ops)
C_KV_POS = 30                        # frozen C_KV_POS
C_SLAB_SHIFT = 31                    # frozen C_SLAB_SHIFT

KSTAGE_SRAM = 0x300000
VSTAGE_SRAM = 0x380000

# buffer name -> AR index (for byte-address lookups)
_AR_NAME = {
    "x": AR_X, "xn": AR_XN, "q": AR_Q, "k": AR_K, "v": AR_V, "ctx": AR_CTX,
    "o": AR_O, "gate": AR_GATE, "up": AR_UP, "h": AR_H, "down": AR_DOWN,
    "act8": AR_ACT8, "scale": AR_SCALE, "out_tile": AR_OUTT,
    "scores": AR_SCORES, "es": AR_ES, "ctx_acc": AR_CTX_ACC,
    "mrun": AR_MRUN, "lrun": AR_LRUN, "mnew": AR_MNEW, "stile": AR_STILE,
    "alpha": AR_ALPHA, "rinv": AR_RINV, "ones": AR_ONES,
    "act_scale": AR_ACT_SCALE,
    "mask": AR_MASK, "expanded": AR_EXP, "ones_row": AR_ONESROW,
}
_AR_BY_IDX = {v: k for k, v in _AR_NAME.items()}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sram(byte_addr: int) -> int:
    assert byte_addr % 16 == 0 and 0 <= byte_addr < (1 << 19) * 16
    return byte_addr // 16


def _hbm(byte_addr: int) -> int:
    assert 0 <= byte_addr < (1 << 40)
    return (1 << 63) | byte_addr


def _f32bits(x: float) -> int:
    return int.from_bytes(np.float32(x).tobytes(), "little")


def _stride(row: int, batch: int = 0) -> int:
    assert 0 <= row <= 0xFFFF and 0 <= batch <= 0xFFFF
    return (row << 16) | batch


def _cd(scale_sram_word: int) -> int:
    """CD dequant descriptor: [20]=1 per-128-group, [19]=0 BF16, [18:0] word addr."""
    return (1 << 20) | (0 << 19) | scale_sram_word


def _act_desc(scale_sram_word: int) -> int:
    """QUANT per-tensor scale descriptor: [20]=0 per-tensor, [19]=0 BF16, [18:0] word addr."""
    return (0 << 20) | (0 << 19) | scale_sram_word


_OPERAND_ALIAS = {
    "ara": "ARa", "arb": "ARb", "arc": "ARc", "ard": "ARd",
    "m": "M", "n": "N", "k": "K",
    "ca": "CA", "cb": "CB", "cc": "CC", "cd": "CD", "cv": "CV",
    "transpose_a": "transpose_A", "transpose_b": "transpose_B",
}


class Em:
    """Instruction emitter: builds Inst objects, encodes to a byte stream."""

    __slots__ = ("insts",)

    def __init__(self):
        self.insts: list = []

    def i(self, mn, srcA=0, srcB=0, acc=0, **op):
        norm = {_OPERAND_ALIAS.get(k, k): v for k, v in op.items()}
        self.insts.append(I.Inst(mn, srcA, srcB, acc, norm))
    def cfg_ar(self, reg, byte_addr, hbm=False):
        a = _hbm(byte_addr) if hbm else _sram(byte_addr)
        self.i("CONFIG", REG=reg, reg_class=1, IMM64=a)

    def cfg_c(self, reg, val):
        self.i("CONFIG", REG=reg, reg_class=0, IMM64=val)

    def barrier(self):
        self.i("BARRIER")

    def wait(self, mask):
        self.i("WAIT", eng_mask=mask)

    def encode(self) -> bytes:
        return b"".join(inst.to_bytes() for inst in self.insts)


# ---------------------------------------------------------------------------
# SRAM buffer layout (monotonic; skips the fixed KV staging region)
# ---------------------------------------------------------------------------
def _layout(M: int) -> dict:
    e = 2                              # BF16 bytes

    def al(n):                         # 16B align
        return (n + 15) // 16 * 16

    b: dict = {}
    off = 0x000000

    def alloc(name, nbytes):
        nonlocal off
        if 0x300000 <= off < 0x400000:
            off = 0x400000
        b[name] = off
        off += nbytes

    alloc("x", al(M * H * e))
    alloc("xn", al(M * H * e))
    alloc("q", al(M * QH * HD * e))     # q|k|v fused region (contiguous)
    alloc("k", al(M * KVH * HD * e))
    alloc("v", al(M * KVH * HD * e))
    alloc("ctx", al(M * QH * HD * e))
    alloc("o", al(M * H * e))
    alloc("gate", al(M * INT * e))
    alloc("up", al(M * INT * e))
    alloc("h", al(M * INT * e))
    alloc("down", al(M * H * e))
    alloc("act8", al(M * INT))          # INT8 (1B)
    alloc("scale", al(N_TILE * 24 * e))  # per-128-group scale tile (max G=24)
    alloc("out_tile", al(M * N_TILE * 4))  # fp32 linear output tile
    n_score = BLOCK * BLOCK if M > 1 else 2 * N_TILE   # PF 128x128 / DC 2 heads x N_TILE
    n_ctx = BLOCK * HD if M > 1 else 2 * HD            # PF 128xHD / DC 2 heads x HD
    alloc("scores", al(n_score * e))       # BF16 QK^T output
    alloc("es", al(n_score * e))           # BF16 exp(scores)
    alloc("ctx_acc", al(n_ctx * e))        # BF16 ctx accumulator
    # softmax scalar buffers: DC keeps 2 per-head scalars, PF keeps BLOCK
    # per-row scalars — each at a 16B stride so the binary-op broadcast
    # (cv!=0 -> ARb[0]) can address any single scalar on a 16B boundary.
    n_scal = BLOCK if M > 1 else 2
    alloc("mrun", n_scal * 16)
    alloc("lrun", n_scal * 16)
    alloc("mnew", 2 * 16)                 # DC-only online running max (2 heads)
    alloc("stile", 2 * 16)                # DC-only tile reduction scratch
    alloc("alpha", 2 * 16)                # DC-only online rescale factor
    alloc("rinv", n_scal * 16)
    alloc("ones_row", al(BLOCK * e))           # PF softmax all-ones row (VEXP(0))
    alloc("expanded", al(BLOCK * BLOCK * e))   # PF per-row scalar expansion
    alloc("ones", al(M * QH * HD * e))     # all-ones gamma (per-token max len QH*HD)
    alloc("mask", al(BLOCK * BLOCK * e))   # PF causal mask tile (VMASK output)
    alloc("act_scale", al(1 * e))          # one BF16 activation scale
    assert off < (1 << 19) * 16, f"SRAM layout overflow: 0x{off:X}"
    return b


def _addr(b: dict, ar: int) -> int:
    return b[_AR_BY_IDX[ar]]


# ---------------------------------------------------------------------------
# op emitters
# ---------------------------------------------------------------------------
def _emit_prologue(em: Em, M: int, b: dict, slab_shift: int = 22):
    """CONFIG segment: AR_KV_BASE / C_KV_POS / C_SLAB_SHIFT / scale pointers."""
    em.i("MODE", mode=1 if M == 1 else 0)
    em.cfg_ar(AR_KVBASE, AR_KV_BASE, hbm=True)
    em.cfg_c(C_KV_POS, 0)
    em.cfg_c(C_SLAB_SHIFT, slab_shift)
    for name, ar in _AR_NAME.items():
        em.cfg_ar(ar, b[name])
    em.cfg_ar(AR_KSTAGE, KSTAGE_SRAM)
    em.cfg_ar(AR_VSTAGE, VSTAGE_SRAM)
    em.cfg_ar(AR_IN_HBM, INPUT_HBM, hbm=True)
    em.cfg_ar(AR_OUT_HBM, LOGITS_HBM, hbm=True)
    em.cfg_c(C_EPS, _f32bits(RMS_EPS))
    em.cfg_c(C_THETA, _f32bits(ROPE_THETA))
    em.cfg_c(C_ACT, _act_desc(_sram(b["act_scale"])))
    em.cfg_c(C_ATTN, _f32bits(ATTN_SCALE))
    if M > 1:                              # PF softmax all-ones row (self-contained)
        em.i("VSUB", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_ONESROW, arb=AR_ONESROW, ard=AR_ONESROW, len=BLOCK, cv=0,
             imm=0)                        # ones_row = ones_row - ones_row = 0
        em.i("VEXP", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_ONESROW, arb=AR_ONES, ard=AR_ONESROW, len=BLOCK, cv=0,
             imm=0)                        # ones_row = exp(0) = 1


MAX_VLEN = 65520                      # largest 16B-aligned len (16-bit field)


def _tvec(em: Em, b: dict, mn: str, a_ar: int, b_ar: int, d_ar: int, total: int,
          srcA: int, srcB: int, acc: int, cv: int = 0, imm: int = 0):
    """Emit a BF16 vector op tiled to len<=65535 (all operands BF16, esz=2)."""
    off = 0
    while off < total:
        n = min(MAX_VLEN, total - off)
        em.cfg_ar(AR_SC0, _addr(b, a_ar) + off * 2)
        em.cfg_ar(AR_SC1, _addr(b, b_ar) + off * 2)
        em.cfg_ar(AR_SC2, _addr(b, d_ar) + off * 2)
        em.i(mn, srcA=srcA, srcB=srcB, acc=acc,
             ara=AR_SC0, arb=AR_SC1, ard=AR_SC2, len=n, cv=cv, imm=imm)
        off += n


def _emit_quant(em: Em, b: dict, M: int, K: int, src_ar: int):
    """QUANT BF16 activation -> INT8 (per-tensor scale sx), tiled."""
    total = M * K
    off = 0
    while off < total:
        n = min(MAX_VLEN, total - off)
        em.cfg_ar(AR_SC0, _addr(b, src_ar) + off * 2)
        em.cfg_ar(AR_SC2, _addr(b, AR_ACT8) + off)
        em.i("QUANT", srcA=I.DT_BF16, acc=0,
             ara=AR_SC0, arb=AR_ONES, ard=AR_SC2, len=n, cv=C_ACT, imm=0)
        em.barrier()
        off += n


def _emit_linear(em: Em, M: int, N: int, K: int, src_ar: int, dst_ar: int,
                 wq_hbm: int, scale_hbm: int, b: dict, wmode: str = "W8A8"):
    """Linear: W8A8 (QUANT -> INT8 GEMM dequant) or W4A16 (BF16 x INT4 GEMM
    dequant, activation stays BF16 — no QUANT), direct token-major write.

    The GEMM dequant path writes BF16 (04 §1.5) directly into the destination
    buffer with the full token-major row stride (C_CC = N*2 bytes), so tile t
    lands at dst + t*N_TILE*2 *within each token row* — no VMOV staging (the
    wave-1 VMOV wrote a contiguous M*N_TILE blob at dst + t*N_TILE*2, which
    overlapped tiles for M>1). src_ar: activation AR (M x K). dst_ar: BF16
    output AR (M x N). wmode: 'W8A8' | 'W4A16'.
    """
    assert wmode in ("W8A8", "W4A16")
    w4 = wmode == "W4A16"
    mn = "GEMV" if M == 1 else "GEMM"
    G = K // 128
    ntiles = N // N_TILE
    scale_tile_bytes = N_TILE * G * 2
    w_row_bytes = K // 2 if w4 else K        # packed INT4 / INT8 weight row
    a_row_bytes = K * 2 if w4 else K         # BF16 / INT8 activation row
    em.cfg_c(C_CA, _stride(a_row_bytes))
    em.cfg_c(C_CB, _stride(w_row_bytes))
    em.cfg_c(C_CC, _stride(N * 2))
    em.cfg_c(C_CD, _cd(_sram(b["scale"])))
    if w4:
        a_ar = src_ar                          # BF16 activation read directly
    else:
        _emit_quant(em, b, M, K, src_ar)
        a_ar = AR_ACT8
    for t in range(ntiles):
        em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * w_row_bytes, hbm=True)
        em.cfg_ar(AR_SCALE_HBM, scale_hbm + t * scale_tile_bytes, hbm=True)
        em.i("DMA.LOAD", srcA=I.DT_BF16,
             SrcAR=AR_SCALE_HBM, DstAR=AR_SCALE, RowBytes=scale_tile_bytes,
             NumRows=1, StrideC=0, mode=0)
        em.cfg_ar(AR_OUTT, _addr(b, dst_ar) + t * N_TILE * 2)
        if w4:
            em.i(mn, srcA=I.DT_BF16, srcB=I.DT_INT4, acc=I.ACC_FP32,
                 ara=a_ar, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=K, batch=1,
                 ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD,
                 acc_init=1, bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
        else:
            em.i(mn, srcA=I.DT_INT8, srcB=I.DT_INT8, acc=I.ACC_INT32,
                 ara=a_ar, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=K, batch=1,
                 ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD,
                 acc_init=1, bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
        em.barrier()

def _emit_qkv(em: Em, M: int, src_ar: int, qkv, b: dict, wmode: str = "W8A8"):
    """QKV as three separate linears (Q->q, K->k, V->v), each token-major.

    The wave-1 fused [M, 4096] blob put K/V outside the k/v buffers; three
    linears keep each projection's rows contiguous in its own AR buffer.
    """
    H_bytes = H // 2 if wmode == "W4A16" else H   # packed INT4 / INT8 bytes/elem
    _emit_linear(em, M, QH * HD, H, src_ar, AR_Q, qkv.wq_hbm, qkv.scale_hbm,
                 b, wmode)
    _emit_linear(em, M, KVH * HD, H, src_ar, AR_K,
                 qkv.wq_hbm + QH * HD * H_bytes,
                 qkv.scale_hbm + QH * HD * (H // 128) * 2, b, wmode)
    _emit_linear(em, M, KVH * HD, H, src_ar, AR_V,
                 qkv.wq_hbm + (QH * HD + KVH * HD) * H_bytes,
                 qkv.scale_hbm + (QH * HD + KVH * HD) * (H // 128) * 2, b,
                 wmode)


def _emit_rmsnorm(em: Em, b: dict, M: int, L: int, src_ar: int, dst_ar: int,
                  mode: int):
    """RMSNorm with per-token emission (PF) / single (DC). gamma = AR_ONES."""
    for t in range(M):
        em.cfg_ar(AR_SC0, _addr(b, src_ar) + t * L * 2)
        em.cfg_ar(AR_SC1, _addr(b, AR_ONES))
        em.cfg_ar(AR_SC2, _addr(b, dst_ar) + t * L * 2)
        em.i("RMSNORM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_SC1, ard=AR_SC2, len=L, cv=C_EPS,
             imm=mode << 31)


def _emit_rope(em: Em, b: dict, M: int, L: int, ar: int, first_pos: int = 0):
    """ROPE with per-token position (PF) / single (DC, runtime patches pos)."""
    for t in range(M):
        em.cfg_ar(AR_SC0, _addr(b, ar) + t * L * 2)
        em.cfg_ar(AR_SC1, _addr(b, AR_ONES))
        em.cfg_ar(AR_SC2, _addr(b, ar) + t * L * 2)
        em.i("ROPE", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_SC1, ard=AR_SC2, len=L, cv=C_THETA,
             imm=(first_pos + t) & 0xFFFF)


def _emit_residual(em: Em, b: dict, M: int, acc_ar: int, dst_ar: int):
    _tvec(em, b, "VADD", dst_ar, acc_ar, dst_ar, M * H,
          I.DT_BF16, I.DT_BF16, I.ACC_FP32)


# ---------------------------------------------------------------------------
# online softmax (DC, per KV head group, 2 Q heads, running max/sum)
# ---------------------------------------------------------------------------
def _softmax_head(em: Em, b: int, n: int, first: bool, bs: dict):
    """Online-softmax per Q head b (running max/sum across score tiles).

    Per-head scalar buffers sit at (mrun/lrun/stile/... + b*16); the binary ops
    that broadcast a per-head scalar across `len` lanes carry cv=C_BROADCAST
    (cv!=0 -> ARb[0] broadcast), the rest are cv=0 (contiguous).
    """
    sh = bs["scores"] + b * n * 2
    eh = bs["es"] + b * n * 2
    ch = bs["ctx_acc"] + b * HD * 2
    em.cfg_ar(AR_MRUN, bs["mrun"] + b * 16)
    em.cfg_ar(AR_LRUN, bs["lrun"] + b * 16)
    em.cfg_ar(AR_MNEW, bs["mnew"] + b * 16)
    em.cfg_ar(AR_STILE, bs["stile"] + b * 16)
    em.cfg_ar(AR_ALPHA, bs["alpha"] + b * 16)
    em.cfg_ar(AR_SC0, sh)
    em.cfg_ar(AR_SC1, eh)
    em.cfg_ar(AR_SC2, ch)
    # tile max -> stile; running max mnew = max(mrun, stile)
    em.i("VREDUCE_MAX", srcA=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_SC0, arb=AR_ONES, ard=AR_STILE, len=n, cv=C_MASK, imm=0)
    if first:
        em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_STILE, arb=AR_ONES, ard=AR_MRUN, len=1, cv=0, imm=0)
    else:
        em.i("VMAX", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_MRUN, arb=AR_STILE, ard=AR_MNEW, len=1, cv=0, imm=0)
        em.i("VSUB", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_MRUN, arb=AR_MNEW, ard=AR_ALPHA, len=1, cv=0, imm=0)
        em.i("VEXP", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_ALPHA, arb=AR_ONES, ard=AR_ALPHA, len=1, cv=0, imm=0)
        # ctx *= alpha (scalar broadcast over HD); l *= alpha
        em.i("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC2, arb=AR_ALPHA, ard=AR_SC2, len=HD, cv=C_BROADCAST, imm=0)
        em.i("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_LRUN, arb=AR_ALPHA, ard=AR_LRUN, len=1, cv=0, imm=0)
        em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_MNEW, arb=AR_ONES, ard=AR_MRUN, len=1, cv=0, imm=0)
    # es = exp(scores - mrun)
    em.i("VSUB", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_SC0, arb=AR_MRUN, ard=AR_SC1, len=n, cv=C_BROADCAST, imm=0)
    em.i("VEXP", srcA=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_SC1, arb=AR_ONES, ard=AR_SC1, len=n, cv=0, imm=0)
    # stile = sum(es); l += stile (first: l = stile)
    em.i("VREDUCE_SUM", srcA=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_SC1, arb=AR_ONES, ard=AR_STILE, len=n, cv=C_MASK, imm=0)
    if first:
        em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_STILE, arb=AR_ONES, ard=AR_LRUN, len=1, cv=0, imm=0)
    else:
        em.i("VADD", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_LRUN, arb=AR_STILE, ard=AR_LRUN, len=1, cv=0, imm=0)


def _emit_subtile(em: Em, b: dict, first: bool, mask_addr: int):
    """One DC score subtile: BMM QK^T -> VSCALE -> VADD tail mask -> online
    softmax (2 Q heads) -> BMM AV accumulate into ctx_acc."""
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


def _emit_attention_dc(em: Em, layer: int, b: dict, mask_base: int):
    """Decode attention: 4 KV.LOAD tiles cover [0,8192); the current token
    (pos 8192, unreachable by 13-bit KV.LOAD pos_start) is VMOV'd from the
    post-RoPE SRAM K/V buffers; a runtime-written per-subtile tail mask is
    VADD'd onto scores before the online softmax (65 subtiles per head)."""
    q_stride = HD * 2                   # 256 B between Q head rows (BF16)
    ctx_stride = HD * 2
    score_stride = N_TILE * 2           # BF16 row
    em.cfg_c(C_CA, _stride(q_stride, q_stride))
    em.cfg_c(C_CB, _stride(HD * 2, 0))  # batch_stride_B = 0 (shared K/V)
    em.cfg_c(C_CC, _stride(score_stride, score_stride))

    kv_tiles = WINDOW // KV_TILE        # 4 tiles of 2048
    for g in range(KVH):
        em.cfg_ar(AR_QG, b["q"] + 2 * g * q_stride)
        first = True
        sub_idx = 0
        for kt in range(kv_tiles):
            pos_start = kt * KV_TILE
            # reset staging base before each KV.LOAD (prior tile advanced it)
            em.cfg_ar(AR_KSTAGE, KSTAGE_SRAM)
            em.cfg_ar(AR_VSTAGE, VSTAGE_SRAM)
            em.i("KV.LOAD", dstK=AR_KSTAGE, dstV=AR_VSTAGE, layer=layer,
                 head=g, sel=2, pos_start=pos_start, count=KV_TILE)
            em.wait(8)
            for _st in range(KV_TILE // N_TILE):
                # advance K/V staging pointers to sub-tile _st (128 tokens)
                em.cfg_ar(AR_KSTAGE, KSTAGE_SRAM + _st * N_TILE * HD * 2)
                em.cfg_ar(AR_VSTAGE, VSTAGE_SRAM + _st * N_TILE * HD * 2)
                _emit_subtile(em, b, first,
                              mask_base + sub_idx * (2 * N_TILE) * 2)
                first = False
                sub_idx += 1
        # current-token tail subtile: VMOV post-RoPE K/V into staging
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
        # normalize ctx_acc /= l_run (per head) and write to global ctx
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


def _emit_attention_pf(em: Em, layer: int, b: dict):
    """Prefill attention (single block, window=128): GEMM QK^T/AV per Q head.
    Token-major Q (QH*HD*2 token stride) and direct token-major ctx write
    (C_CC = QH*HD*2), not the wave-1 head-major ctx_acc + contiguous VMOV."""
    # causal mask tile (single 128-token block, row/col base 0): d[i,j]=0
    # if j<=i else -inf; shared across all Q heads / KV groups.
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
            em.cfg_c(C_CA, _stride(QH * HD * 2, 0))   # Q token stride
            em.cfg_c(C_CB, _stride(HD * 2, 0))
            em.cfg_c(C_CC, _stride(BLOCK * 2, 0))
            em.i("GEMM", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
                 ara=AR_QG, arb=AR_KSTAGE, arc=AR_SCORES, m=BLOCK, n=BLOCK,
                 k=HD, batch=1, ca=C_CA, cb=C_CB, cc=C_CC, cd=0,
                 acc_init=1, bsrc=0, dequant=0, transpose_a=0, transpose_b=1)
            em.barrier()
            # scores x= 128^-0.5, then add causal mask (-inf / 0)
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


def _expand_rows(em: Em, b: dict, n: int, scalar_name: str):
    """Expand n per-row scalars (16B-strided) into b['expanded'] (n x n):
    expanded[i, :] = scalar[i] via VMUL broadcast (cv!=0) against ones_row."""
    for i in range(n):
        em.cfg_ar(AR_SC1, b[scalar_name] + i * 16)
        em.cfg_ar(AR_SC2, b["expanded"] + i * n * 2)
        em.i("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_ONESROW, arb=AR_SC1, ard=AR_SC2, len=n, cv=C_BROADCAST,
             imm=0)


def _emit_softmax_pf(em: Em, b: dict):
    """Prefill softmax over seq per token row (128 rows x 128 seq).

    VREDUCE emits one per-row scalar per call (ngroups=1 via C[C_MASK]=0),
    each landing on a 16B boundary; the scalars are then expanded into a
    16384-element buffer (each row's scalar broadcast across its 128 columns)
    so the final VSUB/VMUL read the buffer contiguously with cv=0 — per-row
    values, never a b[0] broadcast substitution.
    """
    n = BLOCK
    # per-row max -> mrun[i] (16B-strided)
    for i in range(n):
        em.cfg_ar(AR_SC0, b["scores"] + i * n * 2)
        em.cfg_ar(AR_SC2, b["mrun"] + i * 16)
        em.i("VREDUCE_MAX", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_ONES, ard=AR_SC2, len=n, cv=C_MASK, imm=0)
    _expand_rows(em, b, n, "mrun")
    em.i("VSUB", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_SCORES, arb=AR_EXP, ard=AR_ES, len=n * n, cv=0, imm=0)
    em.i("VEXP", srcA=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_ES, arb=AR_ONES, ard=AR_ES, len=n * n, cv=0, imm=0)
    # per-row sum -> lrun[i]; reciprocal -> rinv[i]
    for i in range(n):
        em.cfg_ar(AR_SC0, b["es"] + i * n * 2)
        em.cfg_ar(AR_SC2, b["lrun"] + i * 16)
        em.i("VREDUCE_SUM", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_ONES, ard=AR_SC2, len=n, cv=C_MASK, imm=0)
        em.cfg_ar(AR_SC0, b["lrun"] + i * 16)
        em.cfg_ar(AR_SC2, b["rinv"] + i * 16)
        em.i("VRECIP", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_SC0, arb=AR_ONES, ard=AR_SC2, len=1, cv=0, imm=0)
    _expand_rows(em, b, n, "rinv")
    em.i("VMUL", srcA=I.DT_BF16, srcB=I.DT_BF16, acc=I.ACC_FP32,
         ara=AR_ES, arb=AR_EXP, ard=AR_ES, len=n * n, cv=0, imm=0)


# ---------------------------------------------------------------------------
# lm_head -> logits (tiled to HBM)
# ---------------------------------------------------------------------------
def _emit_lm_head(em: Em, M: int, b: dict, wq_hbm: int, scale_hbm: int,
                  wmode: str = "W8A8"):
    _emit_rmsnorm(em, b, M, H, AR_X, AR_XN, 0)
    G = H // 128
    ntiles = VOCAB // N_TILE
    w4 = wmode == "W4A16"
    row_a = H * 2 if w4 else H              # BF16 / INT8 activation row bytes
    row_b = H // 2 if w4 else H             # packed INT4 / INT8 weight row bytes
    row_c = N_TILE * 2
    scale_tile_bytes = N_TILE * G * 2
    em.cfg_c(C_CA, _stride(row_a))
    em.cfg_c(C_CB, _stride(row_b))
    em.cfg_c(C_CC, _stride(row_c))
    em.cfg_c(C_CD, _cd(_sram(b["scale"])))
    if w4:
        a_ar = AR_XN                          # BF16 activation read directly
    else:
        _emit_quant(em, b, M, H, AR_XN)
        a_ar = AR_ACT8
    mn = "GEMV" if M == 1 else "GEMM"
    for t in range(ntiles):
        em.cfg_ar(AR_WB, wq_hbm + t * N_TILE * row_b, hbm=True)
        em.cfg_ar(AR_SCALE_HBM, scale_hbm + t * scale_tile_bytes, hbm=True)
        em.i("DMA.LOAD", srcA=I.DT_BF16,
             SrcAR=AR_SCALE_HBM, DstAR=AR_SCALE, RowBytes=scale_tile_bytes,
             NumRows=1, StrideC=0, mode=0)
        if w4:
            em.i(mn, srcA=I.DT_BF16, srcB=I.DT_INT4, acc=I.ACC_FP32,
                 ara=a_ar, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=H, batch=1,
                 ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD,
                 acc_init=1, bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
        else:
            em.i(mn, srcA=I.DT_INT8, srcB=I.DT_INT8, acc=I.ACC_INT32,
                 ara=a_ar, arb=AR_WB, arc=AR_OUTT, m=M, n=N_TILE, k=H, batch=1,
                 ca=C_CA, cb=C_CB, cc=C_CC, cd=C_CD,
                 acc_init=1, bsrc=1, dequant=1, transpose_a=0, transpose_b=1)
        em.barrier()
        em.cfg_ar(AR_DST, b["out_tile"])
        em.i("VMOV", srcA=I.DT_BF16, acc=I.ACC_FP32,
             ara=AR_OUTT, arb=AR_ONES, ard=AR_DST, len=M * N_TILE, cv=0, imm=0)
        em.barrier()
        em.cfg_ar(AR_OUT_HBM, LOGITS_HBM + t * M * N_TILE * 2, hbm=True)
        em.i("DMA.STORE", srcA=I.DT_BF16,
             SrcAR=AR_DST, DstAR=AR_OUT_HBM, RowBytes=M * N_TILE * 2,
             NumRows=1, StrideC=0, mode=0)
    em.barrier()


# ---------------------------------------------------------------------------
# main lowering
# ---------------------------------------------------------------------------
def lower_transformer(mode: str, layouts, wmode: str = "W8A8") -> bytes:
    """Lower the full 0.6B transformer to one PF or DC program (bytes).

    layouts: list of build.Layout with proj / wq_hbm / scale_hbm (the 141
    projections, graph order: per layer [qkv,o,gate,up,down], then lm_head).
    mode: 'PF' (M=128, GEMM, per-token KV.APPEND) | 'DC' (M=1, GEMV, KV.APPEND).
    wmode: 'W8A8' (INT8 weights + INT8 activations) | 'W4A16' (INT4 weights +
    BF16 activations). gamma = AR_ONES placeholder (qrun injects real gamma at
    runtime); HBM addresses are v0 placeholders; slab_shift=22 (4 MiB slabs,
    16K token capacity; pos-8192 tail via VMOV).
    """
    assert mode in ("PF", "DC")
    assert wmode in ("W8A8", "W4A16")
    M = BLOCK if mode == "PF" else 1
    b = _layout(M)
    em = Em()
    _emit_prologue(em, M, b)

    by_layer: dict = {}
    lm = None
    for lay in layouts:
        p = lay.proj
        if p.layer is None:
            lm = lay
        else:
            by_layer.setdefault(p.layer, {})[p.kind] = lay

    # DMA.LOAD input hidden (host writes INPUT_HBM; 2D: M rows of H*2 bytes)
    em.cfg_c(C_DMA, H * 2)
    em.i("DMA.LOAD", srcA=I.DT_BF16,
         SrcAR=AR_IN_HBM, DstAR=AR_X, RowBytes=H * 2,
         NumRows=M, StrideC=C_DMA, mode=1)
    em.barrier()

    for L in range(LAYERS):
        lp = by_layer[L]
        # -- attention
        _emit_rmsnorm(em, b, M, H, AR_X, AR_XN, 0)
        _emit_qkv(em, M, AR_XN, lp["qkv"], b, wmode)
        _emit_rmsnorm(em, b, M, QH * HD, AR_Q, AR_Q, 1)
        _emit_rmsnorm(em, b, M, KVH * HD, AR_K, AR_K, 1)
        _emit_rope(em, b, M, QH * HD, AR_Q, 0)
        _emit_rope(em, b, M, KVH * HD, AR_K, 0)
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
            _emit_attention_dc(em, L, b, MASK_BASE)
        else:
            _emit_attention_pf(em, L, b)
        # -- O projection + residual
        _emit_linear(em, M, H, QH * HD, AR_CTX, AR_O,
                     lp["o"].wq_hbm, lp["o"].scale_hbm, b, wmode)
        _emit_residual(em, b, M, AR_O, AR_X)
        # -- MLP
        _emit_rmsnorm(em, b, M, H, AR_X, AR_XN, 0)
        _emit_linear(em, M, INT, H, AR_XN, AR_GATE,
                     lp["gate"].wq_hbm, lp["gate"].scale_hbm, b, wmode)
        _emit_linear(em, M, INT, H, AR_XN, AR_UP,
                     lp["up"].wq_hbm, lp["up"].scale_hbm, b, wmode)
        _tvec(em, b, "VSILU", AR_GATE, AR_ONES, AR_H, M * INT,
              I.DT_BF16, I.DT_BF16, I.ACC_FP32)
        _tvec(em, b, "VMUL", AR_H, AR_UP, AR_H, M * INT,
              I.DT_BF16, I.DT_BF16, I.ACC_FP32)
        em.barrier()
        _emit_linear(em, M, H, INT, AR_H, AR_DOWN,
                     lp["down"].wq_hbm, lp["down"].scale_hbm, b, wmode)
        _emit_residual(em, b, M, AR_DOWN, AR_X)

    _emit_lm_head(em, M, b, lm.wq_hbm, lm.scale_hbm, wmode)
    return em.encode()
