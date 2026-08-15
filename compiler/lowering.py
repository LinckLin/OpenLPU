"""QCore lowering: qnn.matmul (linear layer) -> qisa (GEMM/GEMV) -> Q-ISA asm.

Delivers M2a: a single linear layer W^T @ x (x: [M, K], W: [N, K]) is lowered to
- PF (prefill): a MODE PF program of N/128 GEMM tiles (M <= 128, K streamed),
- DC (decode):  a MODE DC program of N/128 GEMV tiles (M = 1, K streamed).

Tiling honours the frozen constraints (02-isa §6): N <= 128 per instruction
(tiled), K streamed in one instruction (K <= 65535), M = seq <= 128.
Weights are stored PyTorch-style [out, in] = [N, K], read with transpose_B=1
(02-isa §12 step 1); A is [M, K] row-major with transpose_A=0.

Memory plan (documented, 64B-aligned HBM offsets; SRAM in 16B words):
  HBM:  INPUT_HBM  x | OUTPUT_HBM  result | WQ_HBM  weights | SCALES_HBM scales
  SRAM: X_SRAM x | OUT_SRAM result tiles | SCALE_SRAM CD scale array (INT8)

INT8 (W8A8): activation is per-tensor symmetric INT8 (scale_x), weight is
per-128-K-group symmetric INT8 (scale_w[n, g]). The GEMM CD descriptor carries a
single per-group dequant scale; the compiler folds scale_x into it
(cd_scale[n, g] = scale_x * scale_w[n, g]) since scale_x is constant across all
groups. This is a compiler-side fold — no ISA change (02 §6 dequant=1+CD).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from compiler.isa import isa as I

# -- memory plan constants (all 64B aligned, non-overlapping) ---------------
WQ_HBM = 0x0010_0000        # 1 MiB  — weight tensor base (max 4 MiB BF16 fits)
SCALES_HBM = 0x0080_0000    # 8 MiB  — per-128-group scale base (INT8)
INPUT_HBM = 0x0100_0000     # 16 MiB — activation input (harness writes x here)
OUTPUT_HBM = 0x0200_0000    # 32 MiB — output (harness reads here)

# -- AR / C register allocation ---------------------------------------------
AR_X_SRAM = 0
AR_X_HBM = 1
AR_OUT_SRAM = 2
AR_OUT_HBM = 3
AR_WQ_HBM = 4
AR_SCALE_SRAM = 5
AR_SCALE_HBM = 6

C_CA = 0     # A stride descriptor
C_CB = 1     # B stride descriptor
C_CC = 2     # C stride descriptor
C_DMA_X = 4  # DMA source row stride for activation (K * esz_a)
C_DMA_OUT = 5  # DMA source row stride for output (N * esz_out)
C_CD_BASE = 6  # per-tile dequant descriptor registers C6 .. C6+ntiles-1
GROUP = 128    # per-128-group (frozen)


def _sram_addr(byte_addr: int) -> int:
    """SRAM address: bit63=0, 19-bit word address (16B words)."""
    assert byte_addr % 16 == 0
    return byte_addr // 16


def _hbm_addr(byte_addr: int) -> int:
    """HBM address: bit63=1, 40-bit byte address."""
    assert byte_addr < (1 << 40)
    return (1 << 63) | byte_addr


def _stride_val(row_stride: int, batch_stride: int = 0) -> int:
    return (row_stride << 16) | batch_stride

AR_TILE_BASE = 10    # per-tile C base registers: AR10 .. AR10+ntiles-1
AR_TILE_B_BASE = 34  # per-tile B (weight) base registers: AR34 .. AR34+ntiles-1（N=3072 → 24 tiles，与 C 的 AR10..33 不重叠；AR63=KV_BASE）


def _dtype_size(code: int) -> int:
    return {I.DT_BF16: 2, I.DT_FP16: 2, I.DT_INT8: 1, I.DT_INT4: 0.5,
            I.DT_INT32: 4, I.DT_INT16: 2, I.DT_FP8: 1}[code]


@dataclass
class Plan:
    """A lowered linear layer: asm for PF/DC + memory plan."""
    mode: str
    quant: str                       # "BF16" | "INT8"
    M: int
    N: int
    K: int
    x_shape: tuple[int, int]
    w_shape: tuple[int, int]
    src_dtype: int                   # A/B dtype code
    acc: int                         # acc code
    out_dtype: int                   # element dtype written to SRAM/HBM
    dequant: int
    esz_a: int
    esz_b: int
    esz_out: int
    ntiles: int                      # N // 128
    asm: str
    io: dict = field(default_factory=dict)


def lower_linear(x_shape: tuple[int, int], w_shape: tuple[int, int],
                 mode: str, quant: str = "BF16") -> Plan:
    """Lower a single linear layer x @ W^T.

    x_shape: (M, K) activation (M = seq <= 128).  w_shape: (N, K) weight
    (PyTorch [out, in]).  mode: 'PF' | 'DC'.  quant: 'BF16' | 'INT8'.
    """
    M, K = x_shape
    N, K2 = w_shape
    assert K == K2, "x and W reduction dims must match"
    assert 1 <= M <= 128, "M = seq must be <= 128"
    assert 1 <= N <= 65535 and N % 128 == 0, "N must be a multiple of 128"
    assert K % 128 == 0 and K <= 65535, "K streamed in one instruction, mult of 128"
    assert mode in ("PF", "DC")
    if mode == "DC":
        assert M == 1, "DC linear layer has seq=1"

    if quant == "BF16":
        src_dtype = I.DT_BF16
        acc = I.ACC_FP32
        out_dtype = I.DT_BF16     # executor writes srcA dtype (BF16, 2B)
        dequant = 0
    elif quant == "INT8":
        src_dtype = I.DT_INT8
        acc = I.ACC_INT32
        out_dtype = I.DT_BF16     # dequant -> BF16 (04 §1.5 post-process)
        dequant = 1
    else:
        raise ValueError(f"unknown quant {quant!r}")

    esz_a = _dtype_size(src_dtype)
    esz_b = _dtype_size(src_dtype)
    esz_out = _dtype_size(out_dtype)
    G = K // GROUP
    ntiles = N // 128

    x_bytes = M * K * esz_a
    out_bytes = M * N * esz_out
    scale_bytes = N * G * 2 if quant == "INT8" else 0

    # SRAM layout in byte addresses (16B-aligned; _sram_addr converts to word
    # address at encode time).
    x_sram = 0
    out_sram = (x_bytes + 15) // 16 * 16
    scale_sram = out_sram + (out_bytes + 15) // 16 * 16

    row_stride_a = K * esz_a          # A [M,K] row-major: stride between M rows
    row_stride_b = K * esz_b          # B stored [N,K] (transpose_B=1): stride between N rows
    row_stride_c = N * esz_out      # C tile rows are N*esz_out apart (full-width output)


    L = []
    def emit(s):
        L.append(s)

    mn = "GEMM" if mode == "PF" else "GEMV"
    dts = f"srcA={I.DTYPE_NAMES[src_dtype]} srcB={I.DTYPE_NAMES[src_dtype]} " \
          f"acc={I.ACC_NAMES[acc]}"

    emit(f"MODE {mode}")
    # AR registers
    emit(f"CONFIG AR{AR_X_SRAM} = 0x{_sram_addr(x_sram):X}")
    emit(f"CONFIG AR{AR_X_HBM} = 0x{_hbm_addr(INPUT_HBM):X}")
    emit(f"CONFIG AR{AR_OUT_SRAM} = 0x{_sram_addr(out_sram):X}")
    emit(f"CONFIG AR{AR_OUT_HBM} = 0x{_hbm_addr(OUTPUT_HBM):X}")
    emit(f"CONFIG AR{AR_WQ_HBM} = 0x{_hbm_addr(WQ_HBM):X}")
    if dequant:
        emit(f"CONFIG AR{AR_SCALE_SRAM} = 0x{_sram_addr(scale_sram):X}")
        emit(f"CONFIG AR{AR_SCALE_HBM} = 0x{_hbm_addr(SCALES_HBM):X}")
    # per-tile base registers: C (column tile) + B (weight row tile)
    for t in range(ntiles):
        emit(f"CONFIG AR{AR_TILE_BASE + t} = 0x{_sram_addr(out_sram + t * 128 * esz_out):X}")
        emit(f"CONFIG AR{AR_TILE_B_BASE + t} = 0x{_hbm_addr(WQ_HBM + t * 128 * K * esz_b):X}")
    # C stride descriptors
    emit(f"CONFIG C{C_CA} = 0x{_stride_val(row_stride_a):X}")
    emit(f"CONFIG C{C_CB} = 0x{_stride_val(row_stride_b):X}")
    emit(f"CONFIG C{C_CC} = 0x{_stride_val(row_stride_c):X}")
    if dequant:
        for t in range(ntiles):
            # per-tile CD: scale block for columns [t*128, (t+1)*128) is
            # cd[t*128:(t+1)*128, :] = 128*G bf16 values, contiguous at
            # scale_sram + t*128*G*2 bytes.
            cd_t = (1 << 20) | (0 << 19) | _sram_addr(scale_sram + t * 128 * G * 2)
            emit(f"CONFIG C{C_CD_BASE + t} = 0x{cd_t:X}")
    emit(f"CONFIG C{C_DMA_X} = 0x{K * esz_a:X}")
    emit(f"CONFIG C{C_DMA_OUT} = 0x{N * esz_out:X}")
    # DMA.LOAD activation (2D: M dense rows of K*esz_a bytes, stride = row bytes)
    emit(f"DMA.LOAD SrcAR={AR_X_HBM} DstAR={AR_X_SRAM} RowBytes={K * esz_a} "
         f"NumRows={M} StrideC={C_DMA_X} mode=1 srcA={I.DTYPE_NAMES[src_dtype]}")
    if dequant:
        emit(f"DMA.LOAD SrcAR={AR_SCALE_HBM} DstAR={AR_SCALE_SRAM} "
             f"RowBytes={scale_bytes} NumRows=1 StrideC=0 mode=0 srcA=BF16")
    # GEMM/GEMV tiles (N tiled to 128, K streamed in one instruction)
    for t in range(ntiles):
        emit(f"{mn} ARa={AR_X_SRAM} ARb={AR_TILE_B_BASE + t} ARc={AR_TILE_BASE + t} "
             f"M={M} N=128 K={K} batch=1 "
             f"CA={C_CA} CB={C_CB} CC={C_CC} CD={C_CD_BASE + t} "
             f"acc_init=1 bsrc=1 dequant={dequant} transpose_A=0 transpose_B=1 "
             f"{dts}")
    emit("BARRIER")
    # DMA.STORE result (2D: M dense rows of N*esz_out bytes). Output is BF16
    # (2B/element) — the executor writes the dequant / fp32-accumulator result
    # back in BF16; DMA does not convert, dtype only fixes byte interpretation.
    emit(f"DMA.STORE SrcAR={AR_OUT_SRAM} DstAR={AR_OUT_HBM} "
         f"RowBytes={N * esz_out} NumRows={M} StrideC={C_DMA_OUT} mode=1 "
         f"srcA={I.DTYPE_NAMES[out_dtype]}")

    plan = Plan(
        mode=mode, quant=quant, M=M, N=N, K=K, x_shape=x_shape, w_shape=w_shape,
        src_dtype=src_dtype, acc=acc, out_dtype=out_dtype, dequant=dequant,
        esz_a=esz_a, esz_b=esz_b, esz_out=esz_out, ntiles=ntiles,
        asm="\n".join(L),
        io={
            "input_hbm": INPUT_HBM, "output_hbm": OUTPUT_HBM,
            "wq_hbm": WQ_HBM, "scales_hbm": SCALES_HBM,
            "x_sram": x_sram, "out_sram": out_sram, "scale_sram": scale_sram,
            "out_bytes": out_bytes, "scale_bytes": scale_bytes,
            "row_stride_c": row_stride_c,
        },
    )
    return plan


def encode_program(plan: Plan) -> bytes:
    """Encode a plan's asm to a raw 128-bit instruction byte stream.

    Per-tile output column offsets are already encoded via distinct per-tile
    ARc registers (AR10 .. AR10+ntiles-1) emitted by lower_linear.
    """
    out = bytearray()
    for line in plan.asm.splitlines():
        s = line.strip()
        if not s:
            continue
        insts = I.assemble(s)
        assert len(insts) == 1, f"unexpected multi-inst line: {s!r}"
        inst = insts[0]
        out += inst.to_bytes()
    return bytes(out)


def build_linear_qbin(path: str, model: str, cfg: dict, quant: str,
                      w_shape: tuple[int, int], m_pf: int, m_dc: int,
                      weight_bytes: bytes, scale_bytes: bytes | None = None,
                      weight_name: str = "model.layers.0.self_attn.q_proj.weight",
                      ) -> "object":
    """Build a minimal legal qbin for a single linear layer: PF + DC programs
    sharing one weight tensor in HBM (00-container §2, J2)."""
    from compiler.isa.qbin import Tensor, write_qbin, read_qbin

    N, K = w_shape
    pf = lower_linear((m_pf, K), (N, K), "PF", quant)
    dc = lower_linear((m_dc, K), (N, K), "DC", quant)
    pf_prog = encode_program(pf)
    dc_prog = encode_program(dc)

    if quant == "INT8":
        assert scale_bytes is not None
        t = Tensor(name=weight_name, shape=[N, K], dtype="INT8", hbm_off=WQ_HBM,
                   data=weight_bytes, scales_hbm_off=SCALES_HBM,
                   scales=scale_bytes, scale_dtype="BF16")
        quant_j = {"mode": "W8A8", "group": 128, "sym": True}
    else:
        t = Tensor(name=weight_name, shape=[N, K], dtype="BF16", hbm_off=WQ_HBM,
                   data=weight_bytes)
        quant_j = {"mode": "BF16", "group": 128, "sym": True}

    write_qbin(path, model, cfg, quant_j, [t], pf_prog, dc_prog)
    return read_qbin(path)
