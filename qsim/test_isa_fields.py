"""Field-level assertion tests: Q-ISA encoder/decoder vs 02-isa (frozen).

Runs standalone (`python3 qsim/test_isa_fields.py`) and under pytest.
Every assertion is grounded in docs/spec-src/02-isa.md §2/§3/§4–§8.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from compiler.isa import isa as I


# --- §3 opcode table (33 instructions, frozen) ------------------------------
EXPECTED_OPCODES = {
    # SYS
    "MODE": 0x00, "CONFIG": 0x01, "BARRIER": 0x02, "WAIT": 0x03, "NOP": 0x04,
    # DMA
    "DMA.LOAD": 0x20, "DMA.STORE": 0x21, "DMA.PREFETCH": 0x22,
    # MATRIX
    "GEMM": 0x40, "GEMV": 0x41, "BMM": 0x42,
    # VECTOR
    "VADD": 0x80, "VSUB": 0x81, "VMUL": 0x82, "VDIV": 0x83, "VRECIP": 0x84,
    "VEXP": 0x85, "VRSQRT": 0x86, "VSILU": 0x87, "VMAX": 0x88, "VMOV": 0x89,
    "VSCALE": 0x8A, "VMASK": 0x8B, "VREDUCE_SUM": 0x8C, "VREDUCE_MAX": 0x8D,
    "ROPE": 0x8E, "RMSNORM": 0x8F, "QUANT": 0x90, "DEQUANT": 0x91,
    # KV
    "KV.APPEND": 0xC0, "KV.STORE_BLOCK": 0xC1, "KV.LOAD": 0xC2, "KV.GATHER": 0xC3,
}

EXPECTED_ENGINE = {
    "MODE": 0x00, "CONFIG": 0x00, "BARRIER": 0x00, "WAIT": 0x00, "NOP": 0x00,
    "DMA.LOAD": 0x01, "DMA.STORE": 0x01, "DMA.PREFETCH": 0x01,
    "GEMM": 0x02, "GEMV": 0x02, "BMM": 0x02,
    "KV.APPEND": 0x04, "KV.STORE_BLOCK": 0x04, "KV.LOAD": 0x04, "KV.GATHER": 0x04,
}


def test_33_instructions_frozen():
    assert len(I.OPSPEC) == 33, f"expected 33, got {len(I.OPSPEC)}"
    assert set(I.OPSPEC) == set(EXPECTED_OPCODES)
    for mn, op in EXPECTED_OPCODES.items():
        assert I.OPCODE_BY_NAME[mn] == op, (mn, I.OPCODE_BY_NAME[mn], op)
    # counts per engine
    from collections import Counter
    cnt = Counter(I._engine_for_opcode(op) for op in I.NAME_BY_OPCODE)
    assert cnt == {0x00: 5, 0x01: 3, 0x02: 3, 0x03: 18, 0x04: 4}, dict(cnt)


def test_engine_tag_ranges():
    # §2.2: engine tag must equal opcode-range engine; §3 ranges
    for mn, op in EXPECTED_OPCODES.items():
        eng = I._engine_for_opcode(op)
        assert eng == EXPECTED_ENGINE.get(mn, {"VADD": 0x03, "VSUB": 0x03,
            "VMUL": 0x03, "VDIV": 0x03, "VRECIP": 0x03, "VEXP": 0x03,
            "VRSQRT": 0x03, "VSILU": 0x03, "VMAX": 0x03, "VMOV": 0x03,
            "VSCALE": 0x03, "VMASK": 0x03, "VREDUCE_SUM": 0x03,
            "VREDUCE_MAX": 0x03, "ROPE": 0x03, "RMSNORM": 0x03,
            "QUANT": 0x03, "DEQUANT": 0x03}.get(mn))


def test_header_bit_layout():
    # §2.2: [127:120] engine, [119:112] opcode, [111:109] srcA, [108:106] srcB,
    #       [105:104] acc
    w = I.encode_inst("GEMM", srcA=5, srcB=6, acc=2)
    assert ((w >> 120) & 0xFF) == 0x02          # engine MATRIX
    assert ((w >> 112) & 0xFF) == 0x40          # opcode GEMM
    assert ((w >> 109) & 0x7) == 5              # srcA
    assert ((w >> 106) & 0x7) == 6              # srcB
    assert ((w >> 104) & 0x3) == 2              # acc
    # total width check: 8+8+3+3+2+104 = 128
    assert w < (1 << 128)


def test_dtype_acc_codes():
    # §2.2 tables
    assert I.DTYPE_NAMES[0] == "BF16" and I.DTYPE_NAMES[1] == "FP16"
    assert I.DTYPE_NAMES[2] == "INT8" and I.DTYPE_NAMES[3] == "INT4"
    assert I.DTYPE_NAMES[4] == "INT32" and I.DTYPE_NAMES[5] == "INT16"
    assert I.DTYPE_NAMES[6] == "FP8E4M3"
    assert I.ACC_NAMES[0] == "INT32" and I.ACC_NAMES[1] == "FP32"
    assert I.ACC_NAMES[2] == "FP16"


def test_matrix_field_layout():
    # §6.1 operand layout (bit positions)
    d = I.decode_inst(I.encode_inst(
        "GEMM", ARa=3, ARb=4, ARc=5, M=128, N=64, K=4096, batch=2,
        CA=7, CB=8, CC=9, CD=10, acc_init=1, bsrc=1, dequant=1,
        transpose_A=1, transpose_B=1))
    assert (d["ARa"], d["ARb"], d["ARc"]) == (3, 4, 5)
    assert (d["M"], d["N"], d["K"]) == (128, 64, 4096)
    assert d["batch"] == 2
    assert (d["CA"], d["CB"], d["CC"], d["CD"]) == (7, 8, 9, 10)
    assert (d["acc_init"], d["bsrc"], d["dequant"]) == (1, 1, 1)
    assert (d["transpose_A"], d["transpose_B"]) == (1, 1)
    # CD descriptor bit layout (§6.1): [20] mode, [19] scale_dtype, [18:0] scale_base
    cd_val = (1 << 20) | (1 << 19) | 0x12345
    mode = (cd_val >> 20) & 1
    sdtype = (cd_val >> 19) & 1
    sbase = cd_val & ((1 << 19) - 1)
    assert (mode, sdtype, sbase) == (1, 1, 0x12345)


def test_dma_field_layout():
    # §5.1: [103:98] SrcAR, [97:92] DstAR, [91:76] RowBytes, [75:60] NumRows,
    #       [59:55] StrideC, [54] mode
    d = I.decode_inst(I.encode_inst(
        "DMA.LOAD", SrcAR=1, DstAR=2, RowBytes=4096, NumRows=128,
        StrideC=5, mode=1))
    assert (d["SrcAR"], d["DstAR"], d["RowBytes"], d["NumRows"]) == (1, 2, 4096, 128)
    assert (d["StrideC"], d["mode"]) == (5, 1)


def test_kv_field_layout():
    # §8.4 KV.GATHER: sel(1b @82), broadcast(1b @81), pos_start(13b @68),
    #                 count(14b @54), Cstride(5b @49)
    d = I.decode_inst(I.encode_inst(
        "KV.GATHER", dst=3, layer=5, head=2, sel=1, broadcast=1,
        pos_start=100, count=512, Cstride=7))
    assert (d["dst"], d["layer"], d["head"]) == (3, 5, 2)
    assert (d["sel"], d["broadcast"]) == (1, 1)
    assert (d["pos_start"], d["count"], d["Cstride"]) == (100, 512, 7)


def test_roundtrip_all_instructions():
    for mn in I.OPSPEC:
        w = I.encode_inst(mn, srcA=1, srcB=2, acc=1)
        d = I.decode_inst(w)
        assert d["mnemonic"] == mn
        assert d["engine_tag_valid"]
        assert (d["srcA"], d["srcB"], d["acc"]) == (1, 2, 1)


def test_engine_tag_mismatch_flagged():
    # §10: engine tag must agree with opcode range; decode flags disagreement
    w = I.encode_inst("GEMM")   # engine tag 0x02
    w = (w & ~(0xFF << 120)) | (0x01 << 120)   # corrupt engine tag to DMA
    d = I.decode_inst(w)
    assert not d["engine_tag_valid"]


def test_reserved_opcode_rejected():
    try:
        I.decode_inst(0x55 << 112)
        assert False, "expected reserved-opcode error"
    except ValueError:
        pass


def test_assembler_disassembler_roundtrip():
    asm = """MODE PF
CONFIG AR0 = 0x0000000000100000
CONFIG C3 = 0x00020000
GEMM ARa=0 ARb=1 ARc=2 M=128 N=128 K=1024 batch=1 CA=0 CB=1 CC=2 CD=3 acc_init=1 bsrc=1 dequant=1 transpose_B=1 srcA=INT8 srcB=INT8 acc=INT32
BARRIER
"""
    insts = I.assemble(asm)
    assert [i.mnemonic for i in insts] == ["MODE", "CONFIG", "CONFIG", "GEMM", "BARRIER"]
    # disassemble GEMM and re-assemble -> same encoding
    s = I.disassemble(insts[3].encode())
    assert I.assemble(s)[0].encode() == insts[3].encode()


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} field-level assertion groups passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
