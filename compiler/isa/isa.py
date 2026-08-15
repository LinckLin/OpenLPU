"""QCore Tensor Command ISA v0 — encoder/decoder (128-bit fixed-width).

Authoritative reference: docs/spec-src/02-isa.md (frozen P0).
Every field layout below is transcribed from 02-isa §2/§4–§8; the module is
data-driven so field-level assertions can be checked mechanically against the
spec in the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Engine tags  (§2.2 [127:120])
# ---------------------------------------------------------------------------
ENGINE_SYS = 0x00
ENGINE_DMA = 0x01
ENGINE_MATRIX = 0x02
ENGINE_VECTOR = 0x03
ENGINE_KV = 0x04

ENGINE_NAMES = {
    ENGINE_SYS: "SYS",
    ENGINE_DMA: "DMA",
    ENGINE_MATRIX: "MATRIX",
    ENGINE_VECTOR: "VECTOR",
    ENGINE_KV: "KV",
}

# ---------------------------------------------------------------------------
# dtype codes  (§2.2 3-bit srcA/srcB)
# ---------------------------------------------------------------------------
DT_BF16 = 0
DT_FP16 = 1
DT_INT8 = 2
DT_INT4 = 3
DT_INT32 = 4
DT_INT16 = 5
DT_FP8 = 6

DTYPE_NAMES = {
    DT_BF16: "BF16", DT_FP16: "FP16", DT_INT8: "INT8", DT_INT4: "INT4",
    DT_INT32: "INT32", DT_INT16: "INT16", DT_FP8: "FP8E4M3",
}
DTYPE_CODE = {v: k for k, v in DTYPE_NAMES.items()}

# acc 2-bit codes  (§2.2)
ACC_INT32 = 0
ACC_FP32 = 1
ACC_FP16 = 2
ACC_NAMES = {ACC_INT32: "INT32", ACC_FP32: "FP32", ACC_FP16: "FP16"}
ACC_CODE = {v: k for k, v in ACC_NAMES.items()}

# ---------------------------------------------------------------------------
# opcode table  (§3)
# ---------------------------------------------------------------------------
OP_MODE = 0x00
OP_CONFIG = 0x01
OP_BARRIER = 0x02
OP_WAIT = 0x03
OP_NOP = 0x04

OP_DMA_LOAD = 0x20
OP_DMA_STORE = 0x21
OP_DMA_PREFETCH = 0x22

OP_GEMM = 0x40
OP_GEMV = 0x41
OP_BMM = 0x42

OP_VADD = 0x80
OP_VSUB = 0x81
OP_VMUL = 0x82
OP_VDIV = 0x83
OP_VRECIP = 0x84
OP_VEXP = 0x85
OP_VRSQRT = 0x86
OP_VSILU = 0x87
OP_VMAX = 0x88
OP_VMOV = 0x89
OP_VSCALE = 0x8A
OP_VMASK = 0x8B
OP_VREDUCE_SUM = 0x8C
OP_VREDUCE_MAX = 0x8D
OP_ROPE = 0x8E
OP_RMSNORM = 0x8F
OP_QUANT = 0x90
OP_DEQUANT = 0x91

OP_KV_APPEND = 0xC0
OP_KV_STORE_BLOCK = 0xC1
OP_KV_LOAD = 0xC2
OP_KV_GATHER = 0xC3


def _engine_for_opcode(op: int) -> int:
    if 0x00 <= op <= 0x1F:
        return ENGINE_SYS
    if 0x20 <= op <= 0x3F:
        return ENGINE_DMA
    if 0x40 <= op <= 0x7F:
        return ENGINE_MATRIX
    if 0x80 <= op <= 0xBF:
        return ENGINE_VECTOR
    if 0xC0 <= op <= 0xDF:
        return ENGINE_KV
    raise ValueError(f"opcode 0x{op:02X} falls in reserved range")


# ---------------------------------------------------------------------------
# operand field specs: list of (name, low_bit, width)  (§4–§8)
# ---------------------------------------------------------------------------
_DMA_FIELDS = [
    ("SrcAR", 98, 6), ("DstAR", 92, 6), ("RowBytes", 76, 16),
    ("NumRows", 60, 16), ("StrideC", 55, 5), ("mode", 54, 1),
]

_MATRIX_FIELDS = [
    ("ARa", 98, 6), ("ARb", 92, 6), ("ARc", 86, 6),
    ("M", 78, 8), ("N", 70, 8), ("K", 54, 16), ("batch", 48, 6),
    ("CA", 43, 5), ("CB", 38, 5), ("CC", 33, 5), ("CD", 28, 5),
    ("acc_init", 27, 1), ("bsrc", 26, 1), ("dequant", 25, 1),
    ("transpose_A", 24, 1), ("transpose_B", 23, 1),
]

_VECTOR_FIELDS = [
    ("ARa", 98, 6), ("ARb", 92, 6), ("ARd", 86, 6),
    ("len", 70, 16), ("CV", 65, 5), ("imm", 33, 32),
]

# (mnemonic, opcode, operand_fields)
OPSPEC: dict[str, tuple[int, list[tuple[str, int, int]]]] = {
    "MODE": (OP_MODE, [("mode", 102, 2)]),
    "CONFIG": (OP_CONFIG, [("REG", 98, 6), ("reg_class", 97, 1), ("IMM64", 33, 64)]),
    "BARRIER": (OP_BARRIER, []),
    "WAIT": (OP_WAIT, [("eng_mask", 100, 4)]),
    "NOP": (OP_NOP, []),
    "DMA.LOAD": (OP_DMA_LOAD, _DMA_FIELDS),
    "DMA.STORE": (OP_DMA_STORE, _DMA_FIELDS),
    "DMA.PREFETCH": (OP_DMA_PREFETCH, _DMA_FIELDS),
    "GEMM": (OP_GEMM, _MATRIX_FIELDS),
    "GEMV": (OP_GEMV, _MATRIX_FIELDS),
    "BMM": (OP_BMM, _MATRIX_FIELDS),
    "VADD": (OP_VADD, _VECTOR_FIELDS),
    "VSUB": (OP_VSUB, _VECTOR_FIELDS),
    "VMUL": (OP_VMUL, _VECTOR_FIELDS),
    "VDIV": (OP_VDIV, _VECTOR_FIELDS),
    "VRECIP": (OP_VRECIP, _VECTOR_FIELDS),
    "VEXP": (OP_VEXP, _VECTOR_FIELDS),
    "VRSQRT": (OP_VRSQRT, _VECTOR_FIELDS),
    "VSILU": (OP_VSILU, _VECTOR_FIELDS),
    "VMAX": (OP_VMAX, _VECTOR_FIELDS),
    "VMOV": (OP_VMOV, _VECTOR_FIELDS),
    "VSCALE": (OP_VSCALE, _VECTOR_FIELDS),
    "VMASK": (OP_VMASK, _VECTOR_FIELDS),
    "VREDUCE_SUM": (OP_VREDUCE_SUM, _VECTOR_FIELDS),
    "VREDUCE_MAX": (OP_VREDUCE_MAX, _VECTOR_FIELDS),
    "ROPE": (OP_ROPE, _VECTOR_FIELDS),
    "RMSNORM": (OP_RMSNORM, _VECTOR_FIELDS),
    "QUANT": (OP_QUANT, _VECTOR_FIELDS),
    "DEQUANT": (OP_DEQUANT, _VECTOR_FIELDS),
    "KV.APPEND": (OP_KV_APPEND, [
        ("srcK", 98, 6), ("srcV", 92, 6), ("layer", 86, 6), ("head", 83, 3)]),
    "KV.STORE_BLOCK": (OP_KV_STORE_BLOCK, [
        ("srcK", 98, 6), ("srcV", 92, 6), ("layer", 86, 6), ("head", 83, 3),
        ("pos_start", 70, 13), ("count", 56, 14)]),
    "KV.LOAD": (OP_KV_LOAD, [
        ("dstK", 98, 6), ("dstV", 92, 6), ("layer", 86, 6), ("head", 83, 3),
        ("sel", 81, 2), ("pos_start", 68, 13), ("count", 54, 14)]),
    "KV.GATHER": (OP_KV_GATHER, [
        ("dst", 98, 6), ("dst2", 92, 6), ("layer", 86, 6), ("head", 83, 3),
        ("sel", 82, 1), ("broadcast", 81, 1), ("pos_start", 68, 13),
        ("count", 54, 14), ("Cstride", 49, 5)]),
}

# 33-instruction frozen set (SYS 5 + DMA 3 + MATRIX 3 + VECTOR 18 + KV 4)
_TOTAL = sum(1 for _ in OPSPEC)
assert _TOTAL == 33, f"expected 33 instructions, got {_TOTAL}"

OPCODE_BY_NAME = {mn: spec[0] for mn, spec in OPSPEC.items()}
NAME_BY_OPCODE = {op: mn for mn, (op, _) in OPSPEC.items()}


# ---------------------------------------------------------------------------
# core encode / decode
# ---------------------------------------------------------------------------
def encode_inst(mnemonic: str, srcA: int = 0, srcB: int = 0, acc: int = 0,
                **operands: int) -> int:
    """Encode one instruction to a 128-bit little-endian integer."""
    if mnemonic not in OPSPEC:
        raise ValueError(f"unknown mnemonic {mnemonic!r}")
    opcode, fields = OPSPEC[mnemonic]
    eng = _engine_for_opcode(opcode)

    word = 0
    word |= eng << 120
    word |= opcode << 112
    word |= (srcA & 0x7) << 109
    word |= (srcB & 0x7) << 106
    word |= (acc & 0x3) << 104

    provided = set(operands)
    known = {n for n, _, _ in fields}
    for n, lo, w in fields:
        v = operands.get(n, 0)
        if v < 0 or v >= (1 << w):
            raise ValueError(f"{mnemonic}.{n} = {v} out of range [0, {1 << w})")
        word |= (v & ((1 << w) - 1)) << lo
    extra = provided - known
    if extra:
        raise ValueError(f"{mnemonic}: unknown operand(s) {sorted(extra)}")
    return word


def inst_to_bytes(word: int) -> bytes:
    return word.to_bytes(16, "little")


def decode_inst(word: int) -> dict:
    """Decode a 128-bit integer back to a dict of fields."""
    eng = (word >> 120) & 0xFF
    opcode = (word >> 112) & 0xFF
    srcA = (word >> 109) & 0x7
    srcB = (word >> 106) & 0x7
    acc = (word >> 104) & 0x3
    if opcode not in NAME_BY_OPCODE:
        raise ValueError(f"reserved opcode 0x{opcode:02X}")
    mnemonic = NAME_BY_OPCODE[opcode]
    fields = OPSPEC[mnemonic][1]
    out = {
        "mnemonic": mnemonic, "opcode": opcode, "engine": eng,
        "engine_name": ENGINE_NAMES.get(eng, "RESERVED"),
        "srcA": srcA, "srcB": srcB, "acc": acc,
    }
    for n, lo, w in fields:
        out[n] = (word >> lo) & ((1 << w) - 1)
    expected_eng = _engine_for_opcode(opcode)
    out["engine_tag_valid"] = (eng == expected_eng)
    return out


@dataclass
class Inst:
    """Convenience builder object used by lowering and the assembler."""
    mnemonic: str
    srcA: int = 0
    srcB: int = 0
    acc: int = 0
    operands: dict = field(default_factory=dict)

    def encode(self) -> int:
        return encode_inst(self.mnemonic, self.srcA, self.srcB, self.acc,
                           **self.operands)

    def to_bytes(self) -> bytes:
        return inst_to_bytes(self.encode())


# ---------------------------------------------------------------------------
# assembler / disassembler (text <-> bytes)
# ---------------------------------------------------------------------------
def assemble(text: str) -> list[Inst]:
    """Parse a Q-ISA asm text program into Inst objects."""
    insts: list[Inst] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        mn = tok[0].upper()
        rest = tok[1:]
        if mn not in OPSPEC:
            raise ValueError(f"line {lineno}: unknown mnemonic {mn!r}")
        srcA = srcB = acc = 0
        operands: dict[str, int] = {}

        if mn == "CONFIG":
            # CONFIG ARn = val  |  CONFIG Cn = val
            if len(rest) < 3 or rest[1] != "=":
                raise ValueError(f"line {lineno}: bad CONFIG syntax: {raw!r}")
            target, _, valstr = rest[0], rest[1], rest[2]
            val = _parse_int(valstr)
            if target.upper().startswith("AR"):
                operands["reg_class"] = 1
                operands["REG"] = int(target[2:])
                operands["IMM64"] = val
            elif target.upper().startswith("C"):
                operands["reg_class"] = 0
                operands["REG"] = int(target[1:])
                if val >= (1 << 32):
                    raise ValueError(
                        f"line {lineno}: C{target[1:]} value exceeds 32b")
                operands["IMM64"] = val
            else:
                raise ValueError(f"line {lineno}: bad CONFIG target {target!r}")
        elif mn == "MODE":
            if len(rest) != 1:
                raise ValueError(f"line {lineno}: MODE needs PF|DC")
            operands["mode"] = {"PF": 0, "DC": 1}[rest[0].upper()]
        elif mn == "WAIT":
            mask = 0
            if rest:
                for part in rest[0].split("|"):
                    part = part.upper()
                    mask |= {"DMA": 1, "MATRIX": 2, "VECTOR": 4,
                             "KV": 8}.get(part, 0)
            operands["eng_mask"] = mask
        elif mn in ("BARRIER", "NOP"):
            pass
        else:
            # generic: KEY=VALUE pairs only (dtype via srcA=/srcB=/acc=)
            for kv in rest:
                if "=" not in kv:
                    raise ValueError(
                        f"line {lineno}: bad token {kv!r} (use KEY=VALUE)")
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k == "srcA":
                    srcA = _dtype(v, lineno)
                elif k == "srcB":
                    srcB = _dtype(v, lineno)
                elif k == "acc":
                    code = ACC_CODE.get(v.upper())
                    if code is None:
                        raise ValueError(f"line {lineno}: bad acc {v!r}")
                    acc = code
                else:
                    operands[k] = _parse_int(v)
        insts.append(Inst(mn, srcA, srcB, acc, operands))
    return insts


def _dtype(v: str, lineno: int) -> int:
    code = DTYPE_CODE.get(v.upper())
    if code is None:
        raise ValueError(f"line {lineno}: bad dtype {v!r}")
    return code


def _parse_int(s: str) -> int:
    s = s.strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(s, 10)


def disassemble(word: int) -> str:
    d = decode_inst(word)
    mn = d["mnemonic"]
    srcA, srcB, acc = d["srcA"], d["srcB"], d["acc"]
    if mn == "CONFIG":
        cls = d["reg_class"]
        reg = d["REG"]
        imm = d["IMM64"]
        target = f"AR{reg}" if cls == 1 else f"C{reg}"
        return f"CONFIG {target} = 0x{imm:X}"
    if mn == "MODE":
        return f"MODE {'PF' if d['mode'] == 0 else 'DC'}"
    if mn == "WAIT":
        mask = d["eng_mask"]
        parts = []
        for bit, name in ((0, "DMA"), (1, "MATRIX"), (2, "VECTOR"), (3, "KV")):
            if mask & (1 << bit):
                parts.append(name)
        return "WAIT " + ("|".join(parts) if parts else "0")
    if mn in ("BARRIER", "NOP"):
        return mn

    parts = [mn]
    if srcA:
        parts.append(f"srcA={DTYPE_NAMES[srcA]}")
    if srcB:
        parts.append(f"srcB={DTYPE_NAMES[srcB]}")
    if acc:
        parts.append(f"acc={ACC_NAMES[acc]}")
    for n, _lo, _w in OPSPEC[mn][1]:
        if n in d:
            parts.append(f"{n}={d[n]}")
    return " ".join(parts)


def assemble_bytes(text: str) -> bytes:
    return b"".join(i.to_bytes() for i in assemble(text))


def disassemble_program(data: bytes) -> str:
    assert len(data) % 16 == 0, "program must be a multiple of 16 bytes"
    lines = []
    for off in range(0, len(data), 16):
        w = int.from_bytes(data[off:off + 16], "little")
        lines.append(f"{off // 16:04d}: " + disassemble(w))
    return "\n".join(lines)
