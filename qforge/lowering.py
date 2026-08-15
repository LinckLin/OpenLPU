"""qforge linear lowering: qnn.matmul (linear layer) -> Q-ISA asm.

Streaming-tile lowering, one projection per call:
  MODE -> CONFIG AR/C -> DMA.LOAD x -> per-tile (CONFIG weight/scale/out base
  -> DMA.LOAD scale tile -> GEMM/GEMV N<=128 -> BARRIER -> DMA.STORE out tile).

N is tiled to <=128 per GEMM/GEMV (executor hard limit); K is streamed in a
single instruction (K <= 65535). The per-128-group dequant scale is folded into
a CD descriptor pointing at a per-tile scale block in SRAM. Memory-plan
constants and CD descriptor encoding mirror compiler/lowering.py (read-only).

Reuses compiler/isa/isa.py for instruction encoding (encode -> 128-bit bytes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compiler.isa import isa as I

GROUP = 128

# -- fixed harness HBM regions (mirror compiler/lowering.py) -----------------
INPUT_HBM = 0x0100_0000      # 16 MiB — activation input
OUTPUT_HBM = 0x0200_0000     # 32 MiB — output (logits region)
WQ_HBM = 0x0010_0000         # 1 MiB  — single-tensor weight base (verification)
SCALES_HBM = 0x0080_0000     # 8 MiB  — single-tensor scale base (verification)

# -- AR / C register allocation (mirror compiler/lowering.py) ----------------
AR_X_SRAM = 0
AR_X_HBM = 1
AR_OUT_SRAM = 2
AR_OUT_HBM = 3
AR_WQ_HBM = 4
AR_SCALE_SRAM = 5
AR_SCALE_HBM = 6
AR_TILE_B = 26              # per-tile weight row base (re-CONFIG'd each tile)
AR_TILE_C = 10              # per-tile output column base (fixed reuse buffer)

C_CA = 0
C_CB = 1
C_CC = 2
C_CD = 6
C_DMA_X = 4
C_DMA_OUT = 5


def _sram_addr(byte_addr: int) -> int:
    """SRAM address: bit63=0, 19-bit word address (16B words)."""
    assert byte_addr % 16 == 0 and 0 <= byte_addr < (1 << 19) * 16
    return byte_addr // 16


def _hbm_addr(byte_addr: int) -> int:
    """HBM address: bit63=1, 40-bit byte address."""
    assert 0 <= byte_addr < (1 << 40)
    return (1 << 63) | byte_addr


def _stride_val(row_stride: int, batch_stride: int = 0) -> int:
    assert 0 <= row_stride <= 0xFFFF
    return (row_stride << 16) | batch_stride


def _cd(scale_sram_word: int) -> int:
    """CD dequant descriptor: [20]=1 (per-128-group), [19]=0 (BF16), [18:0] word addr."""
    return (1 << 20) | (0 << 19) | scale_sram_word


@dataclass
class Plan:
    """A lowered linear projection: PF or DC program asm + memory plan."""
    mode: str
    M: int
    N: int
    K: int
    ntiles: int
    asm: str
    io: dict = field(default_factory=dict)


def lower_projection(x_shape: tuple[int, int], w_shape: tuple[int, int],
                     mode: str, *, wq_hbm: int, scale_hbm: int,
                     input_hbm: int, output_hbm: int,
                     wmode: str = "W8A8") -> Plan:
    """Lower one linear projection x @ W^T (dequant) for PF or DC.

    x_shape: (M, K). w_shape: (N, K). mode: 'PF' | 'DC'. wmode: 'W8A8' |
    'W4A16' (srcB=INT4, srcA=BF16, acc=FP32; activation stays BF16, no QUANT).
    Weight at wq_hbm (INT8 [N,K] or packed INT4 [N,K//2]); per-128-group BF16
    scales [N,G] at scale_hbm.
    """
    M, K = x_shape
    N, K2 = w_shape
    assert K == K2, "x and W reduction dims must match"
    assert 1 <= M <= 128, "M = seq <= 128"
    assert N % 128 == 0 and N <= 65535 * 128, "N multiple of 128"
    assert K % GROUP == 0 and K <= 65535, "K streamed in one instr, mult of 128"
    assert mode in ("PF", "DC")
    assert wmode in ("W8A8", "W4A16")
    if mode == "DC":
        assert M == 1, "DC linear layer has seq=1"

    if wmode == "W4A16":
        src_dtype_a = I.DT_BF16     # activation stays BF16 (W4A16)
        src_dtype_b = I.DT_INT4
        acc = I.ACC_FP32
        esz_a = 2                   # BF16 activation element bytes
        esz_b = 0.5                 # packed INT4 element bytes (2 per byte)
    else:
        src_dtype_a = I.DT_INT8
        src_dtype_b = I.DT_INT8
        acc = I.ACC_INT32
        esz_a = 1
        esz_b = 1
    out_dtype = I.DT_BF16  # dequant -> BF16 (04 §1.5 post-process)
    esz_out = 2
    G = K // GROUP
    ntiles = N // 128

    x_bytes = M * K * esz_a
    out_tile_bytes = M * 128 * esz_out
    scale_tile_bytes = 128 * G * 2

    # SRAM layout (16B-aligned, word-addr at encode time)
    x_sram = 0
    out_tile_sram = (x_bytes + 15) // 16 * 16
    scale_sram = out_tile_sram + (out_tile_bytes + 15) // 16 * 16

    row_stride_a = K * esz_a                 # A [M,K] row-major
    row_stride_b = K // 2 if wmode == "W4A16" else K * esz_b  # B [N,K//2] packed
    row_stride_c = 128 * esz_out             # C tile rows within the reused buffer
    dma_out_stride = 128 * esz_out           # output tile store row stride
    weight_row_bytes = row_stride_b          # bytes per B row (tile offset)

    mn = "GEMM" if mode == "PF" else "GEMV"
    dts = f"srcA={I.DTYPE_NAMES[src_dtype_a]} srcB={I.DTYPE_NAMES[src_dtype_b]} " \
          f"acc={I.ACC_NAMES[acc]}"

    L = []
    emit = L.append

    emit(f"MODE {mode}")
    # AR registers
    emit(f"CONFIG AR{AR_X_SRAM} = 0x{_sram_addr(x_sram):X}")
    emit(f"CONFIG AR{AR_X_HBM} = 0x{_hbm_addr(input_hbm):X}")
    emit(f"CONFIG AR{AR_OUT_SRAM} = 0x{_sram_addr(out_tile_sram):X}")
    emit(f"CONFIG AR{AR_OUT_HBM} = 0x{_hbm_addr(output_hbm):X}")
    emit(f"CONFIG AR{AR_WQ_HBM} = 0x{_hbm_addr(wq_hbm):X}")
    emit(f"CONFIG AR{AR_SCALE_SRAM} = 0x{_sram_addr(scale_sram):X}")
    emit(f"CONFIG AR{AR_SCALE_HBM} = 0x{_hbm_addr(scale_hbm):X}")
    emit(f"CONFIG AR{AR_TILE_C} = 0x{_sram_addr(out_tile_sram):X}")
    # C stride descriptors
    emit(f"CONFIG C{C_CA} = 0x{_stride_val(row_stride_a):X}")
    emit(f"CONFIG C{C_CB} = 0x{_stride_val(row_stride_b):X}")
    emit(f"CONFIG C{C_CC} = 0x{_stride_val(row_stride_c):X}")
    emit(f"CONFIG C{C_CD} = 0x{_cd(_sram_addr(scale_sram)):X}")
    emit(f"CONFIG C{C_DMA_X} = 0x{K * esz_a:X}")
    emit(f"CONFIG C{C_DMA_OUT} = 0x{dma_out_stride:X}")
    # DMA.LOAD activation (2D: M dense rows of K bytes)
    emit(f"DMA.LOAD SrcAR={AR_X_HBM} DstAR={AR_X_SRAM} RowBytes={K * esz_a} "
         f"NumRows={M} StrideC={C_DMA_X} mode=1 srcA={I.DTYPE_NAMES[src_dtype_a]}")
    # streaming N-tiles (N<=128), K streamed in one instruction
    for t in range(ntiles):
        emit(f"CONFIG AR{AR_TILE_B} = 0x{_hbm_addr(wq_hbm + t * 128 * weight_row_bytes):X}")
        emit(f"CONFIG AR{AR_SCALE_HBM} = 0x{_hbm_addr(scale_hbm + t * scale_tile_bytes):X}")
        emit(f"CONFIG AR{AR_OUT_HBM} = 0x{_hbm_addr(output_hbm + t * out_tile_bytes):X}")
        emit(f"DMA.LOAD SrcAR={AR_SCALE_HBM} DstAR={AR_SCALE_SRAM} "
             f"RowBytes={scale_tile_bytes} NumRows=1 StrideC=0 mode=0 srcA=BF16")
        emit(f"{mn} ARa={AR_X_SRAM} ARb={AR_TILE_B} ARc={AR_TILE_C} "
             f"M={M} N=128 K={K} batch=1 "
             f"CA={C_CA} CB={C_CB} CC={C_CC} CD={C_CD} "
             f"acc_init=1 bsrc=1 dequant=1 transpose_A=0 transpose_B=1 {dts}")
        emit("BARRIER")
        emit(f"DMA.STORE SrcAR={AR_OUT_SRAM} DstAR={AR_OUT_HBM} "
             f"RowBytes={128 * esz_out} NumRows={M} StrideC={C_DMA_OUT} mode=1 "
             f"srcA={I.DTYPE_NAMES[out_dtype]}")
    emit("BARRIER")

    return Plan(mode=mode, M=M, N=N, K=K, ntiles=ntiles, asm="\n".join(L), io={
        "input_hbm": input_hbm, "output_hbm": output_hbm,
        "wq_hbm": wq_hbm, "scale_hbm": scale_hbm,
        "x_sram": x_sram, "out_tile_sram": out_tile_sram,
        "scale_sram": scale_sram,
        "out_tile_bytes": out_tile_bytes,
        "scale_tile_bytes": scale_tile_bytes,
        "out_bytes": M * N * esz_out,
        "scale_bytes": N * G * 2,
        "row_stride_c": row_stride_c,
    })


def encode_program(plan: Plan) -> bytes:
    """Encode a plan's asm to a raw 128-bit instruction byte stream."""
    out = bytearray()
    for line in plan.asm.splitlines():
        s = line.strip()
        if not s:
            continue
        insts = I.assemble(s)
        assert len(insts) == 1, f"unexpected multi-inst line: {s!r}"
        out += insts[0].to_bytes()
    return bytes(out)


# ---------------------------------------------------------------------------
# P5 FullProgGen: full-model transformer lowering (0.6B 28-layer PF + DC).
# Kept in qforge/program.py to preserve this linear-only P4 skeleton
# (`lower_projection` is still consumed by verify_m3.py per-class checks).
# Re-exported here so the P5 program generator is discoverable via the same
# module namespace the plan names.
# ---------------------------------------------------------------------------
from . import program as _program  # noqa: E402

lower_transformer = _program.lower_transformer
