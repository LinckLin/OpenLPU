"""B' KV RTL co-sim — streaming B-feed (dequant + on-the-fly RoPE) acceptance.

Replaces the retired staged LOAD dequant cases with B-feed assertions: quantize
on write (KV.APPEND/STORE_BLOCK), then a BMM whose CD[31] (KV_QUANT) reads the
quantized KV slab inline and dequantizes (+ rotates K by absolute position) in
the matrix B operand feed.  Three cases:
  * sink   — DC decode, S=4 sink rows at absolute pos 0..3 (K rotate + V dequant)
  * window — DC decode, rolling-window segment at pos_base=8 (large angles)
  * pf     — PF mode B' attention, pos_base=0 full sequence (GEMM QK^T + AV)

Judgement: trace + total cycles identical, BF16 output <= 1 ULP.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE = os.path.join(REPO, "rtl", "ref", "qsim_baseline")

sys.path.insert(0, BASELINE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    BF16 = np.float16

from qsim.executor import Executor, C_KVNORM_BASE, AR_KV_SCALE_BASE  # noqa: E402
from compiler.isa import isa as I  # noqa: E402
import cosim  # noqa: E402
from cosim import dump_preload, run_rtl, expected_trace, bf16_ulp_dist  # noqa: E402

HEAD_DIM = 128
K_NORM_SRAM_WORD = 0x3000          # static k_norm table (SRAM word addr)
KV_SCALE_HBM = 0x10000000          # per-token scale slab region (HBM byte)
KV_BASE_HBM = 0x0                  # KV data slab base (HBM byte)
SLAB_SHIFT = 21
V_SLAB_OFF = 1 << SLAB_SHIFT       # slab_index(V)=1 -> +2 MiB

CD_KV_QUANT = 1 << 31
CD_ROTATE_K = 1 << 30

# C-register indices (program convention)
C_CA, C_CB, C_CC, C_CD = 5, 6, 7, 8
# AR indices
AR_KSTAGE, AR_VSTAGE = 0, 1      # BF16 quantize source staging
AR_Q, AR_S = 2, 3                 # query / scores
AR_SM, AR_CTX = 4, 5              # softmax / ctx
AR_KSLAB, AR_VSLAB = 6, 7         # K / V data slab bases (HBM)


def _bf16b(x) -> bytes:
    return np.asarray(x, dtype=np.float32).astype(BF16).tobytes()


def _cfg(prog: bytearray, reg: int, cls: int, val: int):
    prog += I.encode_inst("CONFIG", REG=reg, reg_class=cls,
                          IMM64=int(val)).to_bytes(16, "little")


def _stride(row: int, batch: int) -> int:
    return ((row & 0xFFFF) << 16) | (batch & 0xFFFF)


def _unit_k(rng, n):
    k = rng.standard_normal((n, HEAD_DIM)).astype(np.float32)
    return k / np.sqrt(np.mean(k * k, axis=1, keepdims=True) + 1e-6)


def _setup(exe: Executor, rng, n_tokens: int):
    exe.C[exe.C_SLAB_SHIFT] = SLAB_SHIFT
    exe.AR[exe.AR_KV_BASE] = (1 << 63) | KV_BASE_HBM
    exe.C[C_KVNORM_BASE] = K_NORM_SRAM_WORD
    exe.AR[AR_KV_SCALE_BASE] = (1 << 63) | KV_SCALE_HBM
    exe.AR[AR_KSTAGE] = 0x2000
    exe.AR[AR_VSTAGE] = 0x2010
    exe.AR[AR_Q] = 0x2020
    exe.AR[AR_S] = 0x2030
    exe.AR[AR_SM] = 0x2040
    exe.AR[AR_CTX] = 0x2050
    exe.AR[AR_KSLAB] = (1 << 63) | KV_BASE_HBM
    exe.AR[AR_VSLAB] = (1 << 63) | (KV_BASE_HBM + V_SLAB_OFF)
    # static signed k_norm table (with negative channels, ~3.4% folded-scale set)
    kn = rng.standard_normal((1, HEAD_DIM)).astype(np.float32) * 2.0
    kn[0, :5] = -np.abs(kn[0, :5])
    exe.write_bytes("sram", K_NORM_SRAM_WORD * 16, _bf16b(kn[0]))
    return kn[0]


def _run_and_compare(exe: Executor, prog: bytes, regions, name: str):
    preload = dump_preload(exe)
    etr, etot = expected_trace(prog)
    trace, total, dump = run_rtl(prog, preload, regions)
    exe.run(prog)
    ok_trace = (trace == etr) and (total == etot)
    off = 0
    max_ulp = 0.0
    for sel, addr, nb in regions:
        ref = exe.read_bytes("hbm" if sel else "sram", addr, nb)
        got = dump[off:off + nb]
        off += nb
        if nb % 2 == 0 and len(ref) == len(got):
            a = np.frombuffer(got, dtype=BF16).astype(np.float32)
            b = np.frombuffer(ref, dtype=BF16).astype(np.float32)
            if a.size:
                max_ulp = max(max_ulp, float(bf16_ulp_dist(a, b).max()))
        elif ref != got:
            ok_trace = False
    return {"name": name, "trace_ok": ok_trace, "max_ulp": max_ulp,
            "cycles": total, "expected_cycles": etot,
            "pass": ok_trace and max_ulp <= 1}


def _bmm_inst(prog: bytearray, *, rotate_k: bool, kv_idx: int, pos_base: int,
              ara: int, arb: int, arc: int, m: int, n: int, k: int,
              srcB: int, transpose_b: int, batch: int = 1, acc_init: int = 1):
    """Emit a B-feed BMM (CD[31]=1, pos_base in reserved [20:5])."""
    cd = CD_KV_QUANT | ((CD_ROTATE_K if rotate_k else 0)) | ((kv_idx & 0x1FF) << 21)
    _cfg(prog, C_CD, 0, cd)
    word = I.encode_inst("BMM", srcA=I.DT_BF16, srcB=srcB, acc=I.ACC_FP32,
                         ARa=ara, ARb=arb, ARc=arc, M=m, N=n, K=k, batch=batch,
                         CA=C_CA, CB=C_CB, CC=C_CC, CD=C_CD,
                         acc_init=acc_init, bsrc=1, dequant=0,
                         transpose_A=0, transpose_B=transpose_b)
    word |= (pos_base & 0xFFFF) << 5
    prog += word.to_bytes(16, "little")


def _common_prologue(prog: bytearray, n_tokens: int):
    _cfg(prog, 31, 0, SLAB_SHIFT)
    _cfg(prog, 63, 1, (1 << 63) | KV_BASE_HBM)
    _cfg(prog, C_KVNORM_BASE, 0, K_NORM_SRAM_WORD)
    for ar, w in ((AR_KSTAGE, 0x2000), (AR_VSTAGE, 0x2010), (AR_Q, 0x2020),
                  (AR_S, 0x2030), (AR_SM, 0x2040), (AR_CTX, 0x2050)):
        _cfg(prog, ar, 1, w)
    _cfg(prog, AR_KV_SCALE_BASE, 1, (1 << 63) | KV_SCALE_HBM)
    _cfg(prog, AR_KSLAB, 1, (1 << 63) | KV_BASE_HBM)
    _cfg(prog, AR_VSLAB, 1, (1 << 63) | (KV_BASE_HBM + V_SLAB_OFF))
    _cfg(prog, 30, 0, 0)  # C_KV_POS = 0


def case_bfeed_sink():
    rng = np.random.default_rng(0)
    exe = Executor()
    S = 4
    kn = _setup(exe, rng, S)
    k_unit = _unit_k(rng, S)
    v = rng.standard_normal((S, HEAD_DIM)).astype(np.float32)
    q = rng.standard_normal((1, HEAD_DIM)).astype(np.float32)
    softmax = np.abs(rng.standard_normal((1, S))).astype(np.float32)

    exe.write_bytes("sram", 0x2000 * 16, b"".join(_bf16b(k_unit[t]) for t in range(S)))
    exe.write_bytes("sram", 0x2010 * 16, b"".join(_bf16b(v[t]) for t in range(S)))
    exe.write_bytes("sram", 0x2020 * 16, _bf16b(q[0]))
    exe.write_bytes("sram", 0x2040 * 16, _bf16b(softmax[0]))

    prog = bytearray()
    _common_prologue(prog, S)
    prog += I.encode_inst("MODE", mode=1).to_bytes(16, "little")  # DC
    # quantize-on-write: 4 tokens of K (INT8 fold) + V (INT4)
    prog += I.encode_inst("KV.STORE_BLOCK", srcA=I.DT_INT8, srcB=I.DT_INT4,
                          srcK=AR_KSTAGE, srcV=AR_VSTAGE, layer=0, head=0,
                          pos_start=0, count=S).to_bytes(16, "little")
    # QK^T: B = K slab (INT8), rotate by absolute pos (sink 0..3)
    _cfg(prog, C_CA, 0, _stride(HEAD_DIM * 2, 0))
    _cfg(prog, C_CB, 0, 0)
    _cfg(prog, C_CC, 0, _stride(S * 2, 0))
    _bmm_inst(prog, rotate_k=True, kv_idx=0, pos_base=0,
              ara=AR_Q, arb=AR_KSLAB, arc=AR_S, m=1, n=S, k=HEAD_DIM,
              srcB=I.DT_INT8, transpose_b=1)
    # AV: B = V slab (INT4), no rotate
    _cfg(prog, C_CA, 0, _stride(S * 2, 0))
    _cfg(prog, C_CC, 0, _stride(HEAD_DIM * 2, 0))
    _bmm_inst(prog, rotate_k=False, kv_idx=0, pos_base=0,
              ara=AR_SM, arb=AR_VSLAB, arc=AR_CTX, m=1, n=HEAD_DIM, k=S,
              srcB=I.DT_INT4, transpose_b=0)
    regions = [(0, 0x2030 * 16, S * 2), (0, 0x2050 * 16, HEAD_DIM * 2)]
    return _run_and_compare(exe, bytes(prog), regions, "BFEED_SINK_K_ROTATE_V")


def case_bfeed_window():
    rng = np.random.default_rng(1)
    exe = Executor()
    W = 8
    POS_BASE = 8                    # rolling-window segment (larger angles)
    kn = _setup(exe, rng, W)
    k_unit = _unit_k(rng, W)
    v = rng.standard_normal((W, HEAD_DIM)).astype(np.float32)
    q = rng.standard_normal((1, HEAD_DIM)).astype(np.float32)
    softmax = np.abs(rng.standard_normal((1, W))).astype(np.float32)

    exe.write_bytes("sram", 0x2000 * 16, b"".join(_bf16b(k_unit[t]) for t in range(W)))
    exe.write_bytes("sram", 0x2010 * 16, b"".join(_bf16b(v[t]) for t in range(W)))
    exe.write_bytes("sram", 0x2020 * 16, _bf16b(q[0]))
    exe.write_bytes("sram", 0x2040 * 16, _bf16b(softmax[0]))

    prog = bytearray()
    _common_prologue(prog, W)
    prog += I.encode_inst("MODE", mode=1).to_bytes(16, "little")
    prog += I.encode_inst("KV.STORE_BLOCK", srcA=I.DT_INT8, srcB=I.DT_INT4,
                          srcK=AR_KSTAGE, srcV=AR_VSTAGE, layer=0, head=0,
                          pos_start=POS_BASE, count=W).to_bytes(16, "little")
    _cfg(prog, C_CA, 0, _stride(HEAD_DIM * 2, 0))
    _cfg(prog, C_CB, 0, 0)
    _cfg(prog, C_CC, 0, _stride(W * 2, 0))
    _bmm_inst(prog, rotate_k=True, kv_idx=0, pos_base=POS_BASE,
              ara=AR_Q, arb=AR_KSLAB, arc=AR_S, m=1, n=W, k=HEAD_DIM,
              srcB=I.DT_INT8, transpose_b=1)
    _cfg(prog, C_CA, 0, _stride(W * 2, 0))
    _cfg(prog, C_CC, 0, _stride(HEAD_DIM * 2, 0))
    _bmm_inst(prog, rotate_k=False, kv_idx=0, pos_base=POS_BASE,
              ara=AR_SM, arb=AR_VSLAB, arc=AR_CTX, m=1, n=HEAD_DIM, k=W,
              srcB=I.DT_INT4, transpose_b=0)
    regions = [(0, 0x2030 * 16, W * 2), (0, 0x2050 * 16, HEAD_DIM * 2)]
    return _run_and_compare(exe, bytes(prog), regions, "BFEED_WINDOW_POS8")


def case_bfeed_pf():
    """PF-mode B' attention: GEMM QK^T (M=seq) + AV, pos_base=0."""
    rng = np.random.default_rng(2)
    exe = Executor()
    seq = 4
    kn = _setup(exe, rng, seq)
    k_unit = _unit_k(rng, seq)
    v = rng.standard_normal((seq, HEAD_DIM)).astype(np.float32)
    q = rng.standard_normal((seq, HEAD_DIM)).astype(np.float32)
    softmax = np.abs(rng.standard_normal((seq, seq))).astype(np.float32)

    exe.write_bytes("sram", 0x2000 * 16, b"".join(_bf16b(k_unit[t]) for t in range(seq)))
    exe.write_bytes("sram", 0x2010 * 16, b"".join(_bf16b(v[t]) for t in range(seq)))
    exe.write_bytes("sram", 0x2020 * 16, b"".join(_bf16b(q[t]) for t in range(seq)))
    exe.write_bytes("sram", 0x2040 * 16, b"".join(_bf16b(softmax[t]) for t in range(seq)))

    prog = bytearray()
    _common_prologue(prog, seq)
    prog += I.encode_inst("KV.STORE_BLOCK", srcA=I.DT_INT8, srcB=I.DT_INT4,
                          srcK=AR_KSTAGE, srcV=AR_VSTAGE, layer=0, head=0,
                          pos_start=0, count=seq).to_bytes(16, "little")
    # PF QK^T: GEMM, A=[seq,128], B=K^T [128,seq]
    _cfg(prog, C_CA, 0, _stride(HEAD_DIM * 2, 0))
    _cfg(prog, C_CB, 0, 0)
    _cfg(prog, C_CC, 0, _stride(seq * 2, 0))
    _bmm_inst(prog, rotate_k=True, kv_idx=0, pos_base=0,
              ara=AR_Q, arb=AR_KSLAB, arc=AR_S, m=seq, n=seq, k=HEAD_DIM,
              srcB=I.DT_INT8, transpose_b=1, batch=1)
    # PF AV: GEMM, A=[seq,seq] softmax, B=V [seq,128]
    _cfg(prog, C_CA, 0, _stride(seq * 2, 0))
    _cfg(prog, C_CC, 0, _stride(HEAD_DIM * 2, 0))
    _bmm_inst(prog, rotate_k=False, kv_idx=0, pos_base=0,
              ara=AR_SM, arb=AR_VSLAB, arc=AR_CTX, m=seq, n=HEAD_DIM, k=seq,
              srcB=I.DT_INT4, transpose_b=0, batch=1)
    regions = [(0, 0x2030 * 16, seq * seq * 2), (0, 0x2050 * 16, seq * HEAD_DIM * 2)]
    return _run_and_compare(exe, bytes(prog), regions, "BFEED_PF_QKT_AV")


def main():
    results = [case_bfeed_sink(), case_bfeed_window(), case_bfeed_pf()]
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[{mark}] {r['name']:30s} trace={r['trace_ok']} "
              f"max_ulp={r['max_ulp']:.1f} "
              f"cycles={r['cycles']} (expect {r['expected_cycles']})")
    all_pass = all(r["pass"] for r in results)
    print(f"\nB' B-feed co-sim result: {'ALL PASS' if all_pass else 'FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
