#!/usr/bin/env python3
"""Preprocess the frozen RTL snapshot for the Yosys 0.44 Verilog frontend.

Yosys 0.44's SystemVerilog support predates several constructs used by the
frozen RTL.  All rewrites below are semantics-neutral (pure desugaring):

1. `return <full-expr>;`  -> `return (<full-expr>);`
   (Yosys `return` accepts only a primary expression).
2. `import <pkg>::*;`     -> removed; every referenced member is qualified
   `<pkg>::<member>` (Yosys does not parse `import` statements, but does
   support qualified `pkg::name` references).
3. Unpacked-array `localparam` constant tables -> `case` ROM functions.
4. SV type-cast operators `32'(x)` / `int'(x)` / `signed'(x)` -> equivalent
   Verilog-2001 expressions.
5. `x inside {a, b, c}`   -> `((x == a) || (x == b) || (x == c))`.
6. Unpacked-array *ports* -> flattened to packed vectors
   (`logic [31:0] X [S]` -> `logic [S*32-1:0] X`; `X[i]` -> `X[(i)*32 +: 32]`);
   unpacked-array *variables* are supported and left alone.

Verilator 4.038 (the project lint/elaboration tool) parses the original
unchanged.  This script copies rtl/ref/asicsnap/*.sv -> asic/gen/*.sv applying
only those rewrites; the snapshot and rtl/ are left untouched.
"""
import re
import pathlib

import return_elim

SRC = pathlib.Path("rtl/ref/asicsnap")
DST = pathlib.Path("asic/gen")

RETURN_RE = re.compile(r"return\s+([^;]+?)\s*;")
INSIDE_RE = re.compile(r"(\w+)\s+inside\s*\{([^}]*)\}", re.DOTALL)

# ---------------------------------------------------------------------------
# Package member lists (used to qualify references after dropping `import`).
# MAX_M/MAX_N/MAX_K are intentionally NOT qualified: in the synthesised files
# they are the matrix_engine module parameters (which shadow the package
# constants), never package-value references.
# ---------------------------------------------------------------------------
QCORE_MEMBERS = [
    "MATRIX_N", "MATRIX_ROWS", "DC_LANES", "DC_LANE_ROWS", "LANES",
    "N_BANK", "BANK_BYTES", "SRAM_BYTES", "WORD_BYTES", "SRAM_WORDS", "BANK_WORDS",
    "HBM_BYTES", "HBM_BURST", "HBM_ADDR_BITS",
    "T_FIRST", "HBM_READ_BPC", "HBM_WRITE_BPC", "SRAM_READ_BPC", "SRAM_WRITE_BPC",
    "MODE_SWITCH", "ARRAY_MAC",
    "ENG_SYS", "ENG_DMA", "ENG_MATRIX", "ENG_VECTOR", "ENG_KV",
    "OP_MODE", "OP_CONFIG", "OP_BARRIER", "OP_WAIT", "OP_NOP",
    "OP_DMA_LOAD", "OP_DMA_STORE", "OP_DMA_PREFETCH",
    "OP_GEMM", "OP_GEMV", "OP_BMM",
    "OP_VADD", "OP_VSUB", "OP_VMUL", "OP_VDIV", "OP_VRECIP", "OP_VEXP",
    "OP_VRSQRT", "OP_VSILU", "OP_VMAX", "OP_VMOV", "OP_VSCALE", "OP_VMASK",
    "OP_VREDUCE_SUM", "OP_VREDUCE_MAX", "OP_ROPE", "OP_RMSNORM", "OP_QUANT",
    "OP_DEQUANT", "OP_KV_APPEND", "OP_KV_STORE_BLOCK", "OP_KV_LOAD", "OP_KV_GATHER",
    "DT_BF16", "DT_FP16", "DT_INT8", "DT_INT4", "DT_INT32", "DT_INT16", "DT_FP8",
    "ACC_INT32", "ACC_FP32", "ACC_FP16",
    "AR_KV_BASE", "C_KV_POS", "C_SLAB_SHIFT",
    "ENG_HI", "ENG_LO", "OP_HI", "OP_LO", "DT_HI", "DT_LO",
    "PRIO_MATRIX_A", "PRIO_MATRIX_B", "PRIO_MATRIX_C", "PRIO_VECTOR", "PRIO_DMA",
    "PRIO_KV",
    "sram_word_t",
    "ceil_div", "matrix_pf_cycles", "matrix_dc_batch_cycles", "vector_latency",
    "hbm_read_cycles", "hbm_write_cycles", "sram_write_cycles", "sram_read_cycles",
    "dtype_size",
]

SOFTFLOAT_MEMBERS = [
    "F_POS_INF", "F_NEG_INF", "F_QNAN", "F_ONE", "F_TWO", "F_HALF", "F_NZERO",
    "bf16_to_fp32", "fp32_to_bf16", "i32_to_f32", "f32_to_i32_rne",
    "fp32_add", "fp32_sub", "fp32_max", "fp32_mul", "fp32_recip", "fp32_div",
    "fp32_rsqrt", "fp32_exp2", "fp32_exp", "fp32_log2", "fp32_pow", "fp32_sin",
    "fp32_cos", "fp16_to_fp32", "fp32_to_fp16",
]

ROPE_LUT_MEMBERS = [
    "ROPE_LUT_N", "INV_2PI", "TWO_PI_HI", "TWO_PI_LO", "N_OVER_2PI",
    "rope_sincos", "rope_lut_sin_fn", "rope_lut_cos_fn", "rope_invf_fn",
]

# softfloat identifiers used by rope_lut_pkg (its package-scope import is
# dropped; only these six are referenced).
ROPE_SOFTFLOAT_IDS = [
    "i32_to_f32", "f32_to_i32_rne", "fp32_add", "fp32_sub", "fp32_mul", "F_HALF",
]

# Unpacked-array localparam constant tables -> case ROM functions.
TABLE_RE = re.compile(
    r"localparam\s+logic\s*\[31:0\]\s+(\w+)\s*\[0:(\d+)\]\s*=\s*'\{(.*?)\};",
    re.DOTALL,
)
USAGE_RE = re.compile(r"\b(ROPE_LUT_SIN|ROPE_LUT_COS|ROPE_INVF)\s*\[\s*(\w+)\s*\]")

CAST_REPLACEMENTS = [
    ("32'(n)", "(n)"),
    ("32'(i)", "(i)"),
    ("32'(qmin)", "(qmin)"),
    ("32'(qmax)", "(qmax)"),
    ("int'({1'b1, u[22:0]} >> shf)", "({1'b1, u[22:0]} >> shf)"),
    ("32'(M)", "{24'b0, M}"),
    ("32'(N)", "{24'b0, N}"),
    ("32'(K >> 7)", "{16'b0, (K >> 7)}"),
    ("32'(signed'({1'b0, e}) - 127)", "({24'b0, e} - 32'sd127)"),
]

# Variable-bound `for` loops -> fixed-bound + runtime guard (Yosys requires
# constant loop bounds; these two compute a sticky-bit OR over a runtime-length
# prefix of a bit vector).
LOOP_REPLACEMENTS = [
    (
        "for (integer k = 0; k < lead - 23 - 1; k = k + 1) st = st | mag[k];",
        "for (integer k = 0; k < 8; k = k + 1) st = st | (mag[k] & (k < (lead - 23 - 1)));",
    ),
    (
        "for (integer k = 0; k < 23 - shf - 1; k = k + 1) sty = sty | v[k];",
        "for (integer k = 0; k < 23; k = k + 1) sty = sty | (v[k] & (k < (23 - shf - 1)));",
    ),
]

# 32-bit-element arrays declared as unpacked *ports* (flattened to packed).
PACKED_ARRAYS = ["a_slice", "b_slice", "a_vec", "b_vec", "out_vec", "va", "vb", "vo"]


def wrap_returns(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if "//" in line:
            code, comment = line.split("//", 1)
            comment = "//" + comment
        else:
            code, comment = line, ""
        code = RETURN_RE.sub(lambda m: "return (%s);" % m.group(1).strip(), code)
        out_lines.append(code + comment)
    return "\n".join(out_lines) + "\n"


def apply_casts(text: str) -> str:
    for old, new in CAST_REPLACEMENTS + LOOP_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def convert_inside(text: str) -> str:
    def repl(m):
        lhs = m.group(1)
        items = [x.strip() for x in m.group(2).split(",") if x.strip()]
        return "(" + " || ".join("(%s == %s)" % (lhs, it) for it in items) + ")"

    return INSIDE_RE.sub(repl, text)


def pack_arrays(text: str) -> str:
    for name in PACKED_ARRAYS:
        # declaration: logic [31:0] NAME [SIZE] -> logic [SIZE*32-1:0] NAME
        text = re.sub(
            r"logic\s*\[31:0\]\s+%s\s*\[\s*([^\]]+?)\s*\]" % name,
            lambda m: "logic [%s*32-1:0] %s" % (m.group(1), name),
            text,
        )
    for name in PACKED_ARRAYS:
        # usage: NAME[idx] (no colon in idx) -> NAME[(idx)*32 +: 32]
        text = re.sub(
            r"\b%s\s*\[\s*([^:\]]+?)\s*\]" % name,
            lambda m: "%s[(%s)*32 +: 32]" % (name, m.group(1)),
            text,
        )
    return text


def convert_tables(text: str) -> str:
    def repl(m):
        name = m.group(1)
        n = int(m.group(2))
        body = m.group(3)
        vals = re.findall(r"32'h[0-9a-fA-F]+", body)
        assert len(vals) == n + 1, (name, len(vals), n)
        fn = name.lower() + "_fn"
        lines = [
            "function automatic logic [31:0] %s(input integer i);" % fn,
            "  case (i)",
        ]
        for idx, v in enumerate(vals):
            lines.append("    %d: %s = %s;" % (idx, fn, v))
        lines.append("    default: %s = 32'h0;" % fn)
        lines.append("  endcase")
        lines.append("endfunction")
        return "\n".join(lines)

    return TABLE_RE.sub(repl, text)


def rewrite_usages(text: str) -> str:
    return USAGE_RE.sub(
        lambda m: "%s_fn(%s)" % (m.group(1).lower(), m.group(2)), text
    )


def qualify(text: str, pkg: str, members) -> str:
    text = re.sub(r"[ \t]*import\s+%s::\*;[ \t]*\n" % re.escape(pkg), "", text)
    for m in members:
        text = re.sub(
            r"\b(?:%s::)?%s\b" % (re.escape(pkg), re.escape(m)),
            "%s::%s" % (pkg, m),
            text,
        )
    return text
def transform(text: str, name: str) -> str:
    out = return_elim.eliminate_returns(text)
    out = apply_casts(out)
    out = convert_inside(out)
    out = pack_arrays(out)

    if name == "rope_lut.sv":
        out = out.replace("  import softfloat_pkg::*;\n", "")
        for ident in ROPE_SOFTFLOAT_IDS:
            out = re.sub(r"\b%s\b" % ident, "softfloat_pkg::%s" % ident, out)
        out = convert_tables(out)
        out = rewrite_usages(out)
    elif name == "vector_engine.sv":
        out = rewrite_usages(out)
        out = qualify(out, "rope_lut_pkg", ROPE_LUT_MEMBERS)
        out = qualify(out, "softfloat_pkg", SOFTFLOAT_MEMBERS)
        out = qualify(out, "qcore_pkg", QCORE_MEMBERS)
    elif name == "command_processor.sv":
        out = qualify(out, "softfloat_pkg", SOFTFLOAT_MEMBERS)
        out = qualify(out, "qcore_pkg", QCORE_MEMBERS)
    elif name == "matrix_engine.sv":
        out = qualify(out, "softfloat_pkg", SOFTFLOAT_MEMBERS)
        out = qualify(out, "qcore_pkg", QCORE_MEMBERS)
    elif name == "dma_engine.sv":
        out = qualify(out, "qcore_pkg", QCORE_MEMBERS)
    return out


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    changed = 0
    for f in sorted(SRC.glob("*.sv")):
        src = f.read_text()
        out = transform(src, f.name)
        (DST / f.name).write_text(out)
        if out != src:
            changed += 1
            print("preprocessed: %s" % f.name)
    print("files with rewrites: %d" % changed)


if __name__ == "__main__":
    main()
