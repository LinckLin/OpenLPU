"""QCore qsim functional core — numerically-exact ISA reference executor.

This is the P3 seed: it executes the frozen 33-instruction Q-ISA subset needed
for M2a with *exact* numerical semantics (no timing model — P3 adds that).
Authoritative semantics: docs/spec-src/02-isa.md (encoding) + 04-execution-engines.md
§1.5 (dequant post-processing) + 00-container.md (command stream).

Executed subset: SYS (CONFIG/BARRIER/WAIT/MODE/NOP), DMA (LOAD/STORE/PREFETCH),
MATRIX (GEMM/GEMV/BMM), VECTOR (18 ops: elementwise / softmax primitives / ROPE /
RMSNORM / VMASK / QUANT / DEQUANT), KV (APPEND/STORE_BLOCK/LOAD/GATHER).
GEMM family supports: acc_init, transpose_A, transpose_B, bsrc, dequant=1 with a
CD per-128-group scale descriptor (INT32 in-group accumulation -> fp32 scale ->
fp-domain cross-group accumulation, per 02 §6 / 04 §1.5).
VECTOR numeric convention (plan §1): fp32 internal datapath (acc=FP32), bf16
inputs upcast, output written back in srcA dtype.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from compiler.isa import isa as I

try:
    import ml_dtypes
    _BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    _BF16 = np.float16  # degraded fallback (bfloat16 unavailable)

DTYPE_NP = {
    I.DT_BF16: _BF16,
    I.DT_FP16: np.float16,
    I.DT_INT8: np.int8,
    I.DT_INT32: np.int32,
    I.DT_INT16: np.int16,
}
DTYPE_SIZE = {
    I.DT_BF16: 2, I.DT_FP16: 2, I.DT_INT8: 1, I.DT_INT4: 0.5,
    I.DT_INT32: 4, I.DT_INT16: 2, I.DT_FP8: 1,
}

SRAM_BYTES = 8 * 1024 * 1024  # 8 MiB
HBM_BYTES = 16 * 1024 ** 3    # 16 GiB (sparse — only touched blocks allocated)
BLOCK = 4096


class SparseMemory:
    """Sparse byte-addressable store (models 16 GiB HBM without allocating it)."""

    def __init__(self, size: int = HBM_BYTES):
        self.size = size
        self._blk: dict[int, bytearray] = {}

    def _block(self, idx: int) -> bytearray:
        b = self._blk.get(idx)
        if b is None:
            b = bytearray(BLOCK)
            self._blk[idx] = b
        return b

    def write(self, addr: int, data: bytes):
        i = 0
        n = len(data)
        while i < n:
            idx = addr // BLOCK
            off = addr % BLOCK
            b = self._block(idx)
            m = min(n - i, BLOCK - off)
            b[off:off + m] = data[i:i + m]
            addr += m
            i += m

    def read(self, addr: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            idx = addr // BLOCK
            off = addr % BLOCK
            b = self._blk.get(idx)
            m = min(n, BLOCK - off)
            if b is None:
                out += b"\x00" * m
            else:
                out += b[off:off + m]
            addr += m
            n -= m
        return bytes(out)


class SramMemory:
    def __init__(self):
        self.buf = bytearray(SRAM_BYTES)

    def write(self, addr: int, data: bytes):
        if addr + len(data) > SRAM_BYTES:
            raise ValueError(f"SRAM write 0x{addr:X}+{len(data)} exceeds 8 MiB")
        self.buf[addr:addr + len(data)] = data

    def read(self, addr: int, n: int) -> bytes:
        if addr + n > SRAM_BYTES:
            raise ValueError(f"SRAM read 0x{addr:X}+{n} exceeds 8 MiB")
        return bytes(self.buf[addr:addr + n])


@dataclass
class InstRef:
    d: dict
    word: int


def _bf16_to_f32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32)


def unpack_int4(raw: bytes, n: int) -> np.ndarray:
    """Unpack n packed 4-bit values to signed int8 (even -> low nibble, odd ->
    high nibble, two's-complement -8..7). This is the authoritative packing
    bit order (02 §6 / 04 §1.2): `_read_vector`/`_write_vector` define it, the
    W4A16 GEMM read path and the RTL unpacker must agree with it.
    """
    b = np.frombuffer(raw, dtype=np.uint8)
    lo = (b & 0x0F).astype(np.int8)
    hi = ((b >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    out = np.empty(len(b) * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:n]


def _rope_apply(x: np.ndarray, pos: int, theta: float,
                npdt) -> np.ndarray:
    """RoPE (HF rotate_half form): x [H,128] -> rotated [H,128].

    inv_freq = 1/(theta^(arange(0,128,2)/128)); emb = cat(freqs,freqs);
    out = x*cos + rotate_half(x)*sin (02 §7.2 / ref model.apply_rope).
    cos/sin are computed fp32 then cast to `npdt` (HF stores the rope table in
    the model dtype, bf16), and the rotation runs in `npdt` arithmetic with
    per-op rounding — this bit-matches the HF bf16 reference.
    Returns fp32 values that are exactly representable in `npdt`.

    The inv_freq/angle chain is explicit fp32 (spec 04 §3.2 fp32 internal data
    path, frozen v0): every ufunc is forced to the fp32 loop via `dtype=` so
    numpy cannot silently widen the scalar/array power path to fp64 (the fp64
    deviation the M6 review flagged). Bit-identical to the RTL ROPE_INVF LUT
    (`1/(1e6^(d/64))`, per-op double rounding).
    """
    D = x.shape[1]
    half = D // 2
    i = np.arange(half, dtype=np.float32)
    exp = np.multiply(i, np.float32(2.0 / D), dtype=np.float32)        # d/64
    base = np.power(np.float32(theta), exp, dtype=np.float32)          # θ^(d/64)
    inv_freq = np.divide(np.float32(1.0), base, dtype=np.float32)      # 1/base
    angles = np.multiply(np.float32(pos), inv_freq, dtype=np.float32)  # pos·inv
    emb = np.concatenate([angles, angles]).astype(np.float32)
    cos = np.cos(emb).astype(npdt)
    sin = np.sin(emb).astype(npdt)
    xb = x.astype(npdt)
    x1 = xb[:, :half]
    x2 = xb[:, half:]
    rh = np.concatenate([-x2, x1], axis=1)  # rotate_half
    out = (xb * cos + rh * sin).astype(npdt)
    return out.astype(np.float32)


class Executor:
    def __init__(self):
        self.sram = SramMemory()
        self.hbm = SparseMemory()
        self.AR = [0] * 64
        self.C = [0] * 32
        self.mode = "PF"  # PF | DC
        self.AR_KV_BASE = 63
        self.C_KV_POS = 30
        self.C_SLAB_SHIFT = 31

    # -- addressing -----------------------------------------------------
    def resolve(self, ar_idx: int) -> tuple[str, int]:
        """Resolve an AR register to (memory, byte-address)."""
        val = self.AR[ar_idx]
        if val >> 63:  # HBM
            return "hbm", val & ((1 << 40) - 1)
        return "sram", (val & ((1 << 19) - 1)) * 16  # 19b word addr, 16B words

    def _mem(self, kind: str):
        return self.hbm if kind == "hbm" else self.sram

    def read_bytes(self, kind: str, addr: int, n: int) -> bytes:
        return self._mem(kind).read(addr, n)

    def write_bytes(self, kind: str, addr: int, data: bytes):
        self._mem(kind).write(addr, data)

    # -- dtype matrix IO ------------------------------------------------
    def read_matrix(self, kind: str, addr: int, dtype: int, nrows: int,
                    ncols: int, row_stride: int, transpose: bool) -> np.ndarray:
        """Read a logical nrows x ncols matrix; storage may be transposed."""
        if dtype == I.DT_INT4:
            return self._read_matrix_int4(kind, addr, nrows, ncols, row_stride,
                                          transpose)
        esz = DTYPE_SIZE[dtype]
        npdt = DTYPE_NP[dtype]
        if not transpose:
            out = np.empty((nrows, ncols), dtype=npdt)
            for r in range(nrows):
                raw = self.read_bytes(kind, addr + r * row_stride, ncols * esz)
                out[r] = np.frombuffer(raw, dtype=npdt)
            return out
        # transposed storage: logical[r][c] == storage[c][r]
        out = np.empty((nrows, ncols), dtype=npdt)
        for c in range(ncols):
            raw = self.read_bytes(kind, addr + c * row_stride, nrows * esz)
            out[:, c] = np.frombuffer(raw, dtype=npdt)
        return out

    def _read_matrix_int4(self, kind: str, addr: int, nrows: int, ncols: int,
                          row_stride: int, transpose: bool) -> np.ndarray:
        """Read a logical nrows x ncols INT4 matrix (4-bit packed, 2 per byte,
        even -> low nibble per `unpack_int4`). Returns signed int8 values."""
        if not transpose:
            out = np.empty((nrows, ncols), dtype=np.int8)
            nbytes = (ncols + 1) // 2
            for r in range(nrows):
                raw = self.read_bytes(kind, addr + r * row_stride, nbytes)
                out[r] = unpack_int4(raw, ncols)
            return out
        # transposed storage: logical[r][c] == storage[c][r]; each storage
        # row holds `nrows` packed 4-bit elements.
        out = np.empty((nrows, ncols), dtype=np.int8)
        nbytes = (nrows + 1) // 2
        for c in range(ncols):
            raw = self.read_bytes(kind, addr + c * row_stride, nbytes)
            out[:, c] = unpack_int4(raw, nrows)
        return out

    def write_matrix(self, kind: str, addr: int, dtype: int, data: np.ndarray,
                     row_stride: int):
        esz = DTYPE_SIZE[dtype]
        nrows = data.shape[0]
        ncols = data.shape[1]
        arr = data.astype(DTYPE_NP[dtype])
        for r in range(nrows):
            self.write_bytes(kind, addr + r * row_stride,
                             arr[r].tobytes())

    # -- stride / CD descriptor helpers ---------------------------------
    @staticmethod
    def _stride(cval: int) -> tuple[int, int]:
        """CA/CB/CC: [31:16] row_stride bytes, [15:0] batch_stride bytes."""
        return (cval >> 16) & 0xFFFF, cval & 0xFFFF

    def _cd(self, cdval: int) -> tuple[int, int, int]:
        """CD: [20] mode, [19] scale_dtype, [18:0] scale_base (SRAM word addr)."""
        mode = (cdval >> 20) & 1
        scale_dtype = (cdval >> 19) & 1
        scale_base = (cdval & ((1 << 19) - 1)) * 16  # word addr -> byte addr
        return mode, scale_dtype, scale_base

    # -- vector / KV helpers -------------------------------------------
    @staticmethod
    def _c_f32(cval: int) -> float:
        """Interpret a 32-bit C-register value as an fp32 bit pattern."""
        return struct.unpack("<f", struct.pack("<I", cval & 0xFFFFFFFF))[0]

    @staticmethod
    def _bf16_bits(bits: int) -> float:
        return float(np.frombuffer(
            struct.pack("<H", bits & 0xFFFF), dtype=_BF16)[0])

    def _read_vector(self, kind: str, addr: int, dtype: int,
                     n: int) -> np.ndarray:
        """Read n contiguous elements; return an fp32 (or int32) working array."""
        if dtype == I.DT_INT4:
            b = np.frombuffer(self.read_bytes(kind, addr, (n + 1) // 2),
                              dtype=np.uint8)
            lo = (b & 0x0F).astype(np.int32)
            hi = ((b >> 4) & 0x0F).astype(np.int32)
            lo = np.where(lo >= 8, lo - 16, lo)
            hi = np.where(hi >= 8, hi - 16, hi)
            out = np.empty(len(b) * 2, dtype=np.int32)
            out[0::2] = lo
            out[1::2] = hi
            return out[:n]
        esz = int(DTYPE_SIZE[dtype])
        raw = self.read_bytes(kind, addr, n * esz)
        arr = np.frombuffer(raw, dtype=DTYPE_NP[dtype])
        if dtype in (I.DT_BF16, I.DT_FP16):
            return arr.astype(np.float32)
        if dtype in (I.DT_INT8, I.DT_INT16):
            return arr.astype(np.int32)
        if dtype == I.DT_INT32:
            return arr.astype(np.int32, copy=True)
        raise NotImplementedError(
            f"VECTOR read dtype {I.DTYPE_NAMES[dtype]} unsupported")

    def _write_vector(self, kind: str, addr: int, dtype: int,
                      arr: np.ndarray):
        if dtype == I.DT_INT4:
            v = np.clip(arr.astype(np.int32), -8, 7) & 0x0F
            pairs = np.zeros((v.shape[0] + 1) // 2, dtype=np.uint8)
            pairs[: v[0::2].shape[0]] = v[0::2]
            pairs[: v[1::2].shape[0]] |= (v[1::2] << 4).astype(np.uint8)
            self.write_bytes(kind, addr, pairs.tobytes())
            return
        out = arr.astype(DTYPE_NP[dtype])
        self.write_bytes(kind, addr, out.tobytes())

    def _kv_slab_base(self, layer: int, head: int, kv: int) -> int:
        """HBM byte address of the (layer, head, kv) slab start (05 §1.3)."""
        _, base = self.resolve(self.AR_KV_BASE)
        shift = self.C[self.C_SLAB_SHIFT]
        if shift not in (20, 21, 22):
            raise ValueError(f"SLAB_SHIFT {shift} invalid (must be 20, 21 or 22)")
        slab_index = (layer << 4) | (head << 1) | kv
        return base + (slab_index << shift)

    # -- execution ------------------------------------------------------
    def run(self, program: bytes, trace: list | None = None):
        assert len(program) % 16 == 0
        for off in range(0, len(program), 16):
            word = int.from_bytes(program[off:off + 16], "little")
            d = I.decode_inst(word)
            if not d["engine_tag_valid"]:
                raise ValueError(
                    f"inst {off//16}: engine tag 0x{d['engine']:02X} mismatches "
                    f"opcode 0x{d['opcode']:02X}")
            self._exec(d)
            if trace is not None:
                trace.append((off // 16, d["mnemonic"]))

    def _exec(self, d: dict):
        mn = d["mnemonic"]
        if mn == "NOP":
            return
        if mn == "CONFIG":
            self._exec_config(d)
        elif mn == "BARRIER":
            return  # no timing model — nothing to drain
        elif mn == "WAIT":
            return  # no timing model — nothing to drain
        elif mn == "MODE":
            self.mode = "PF" if d["mode"] == 0 else "DC"
        elif mn == "DMA.LOAD":
            self._dma(d, load=True)
        elif mn == "DMA.STORE":
            self._dma(d, load=False)
        elif mn == "DMA.PREFETCH":
            pass  # advisory, non-blocking: no functional effect
        elif mn in ("GEMM", "GEMV", "BMM"):
            self._matrix(d)
        elif d["engine"] == I.ENGINE_VECTOR:
            self._vector(d)
        elif d["engine"] == I.ENGINE_KV:
            self._kv(d)
        else:
            raise NotImplementedError(
                f"executor does not implement {mn} (M2a subset)")

    def _exec_config(self, d: dict):
        cls = d["reg_class"]
        reg = d["REG"]
        imm = d["IMM64"]
        if cls == 1:
            self.AR[reg] = imm & ((1 << 64) - 1)
        else:
            if imm >> 32:
                raise ValueError(f"CONFIG C{reg}: IMM64[63:32] must be 0")
            self.C[reg] = imm & 0xFFFFFFFF

    def _dma(self, d: dict, load: bool):
        src_ar = d["SrcAR"]
        dst_ar = d["DstAR"]
        row_bytes = d["RowBytes"]
        num_rows = d["NumRows"]
        stride_c = d["StrideC"]
        mode = d["mode"]
        if mode == 0:  # 1D
            num_rows = 1
        src_kind, src_addr = self.resolve(src_ar)
        dst_kind, dst_addr = self.resolve(dst_ar)
        if load:
            assert src_kind == "hbm" and dst_kind == "sram"
        else:
            assert src_kind == "sram" and dst_kind == "hbm"
        stride = self.C[stride_c] if mode == 1 else 0
        for r in range(num_rows):
            data = self.read_bytes(src_kind, src_addr + r * stride, row_bytes)
            # SRAM destination is dense (no stride)
            self.write_bytes(dst_kind, dst_addr + r * row_bytes, data)

    def _matrix(self, d: dict):
        M = d["M"]
        N = d["N"]
        K = d["K"]
        batch = d["batch"]
        if M > 128 or N > 128:
            raise ValueError(f"{d['mnemonic']}: M,N must be <= 128 (got {M},{N})")
        if d["mnemonic"] == "GEMV" and M != 1:
            raise ValueError("GEMV requires M=1")
        if batch > 16:
            raise ValueError(f"batch {batch} > 16")
        if batch == 0:
            batch = 1

        srcA = d["srcA"]
        srcB = d["srcB"]
        acc = d["acc"]
        row_stride_a, batch_stride_a = self._stride(self.C[d["CA"]])
        row_stride_b, batch_stride_b = self._stride(self.C[d["CB"]])
        row_stride_c, batch_stride_c = self._stride(self.C[d["CC"]])

        a_kind, a_base = self.resolve(d["ARa"])
        b_kind, b_base = self.resolve(d["ARb"])
        c_kind, c_base = self.resolve(d["ARc"])

        acc_init = d["acc_init"]
        dequant = d["dequant"]
        transpose_a = d["transpose_A"]
        transpose_b = d["transpose_B"]

        # dequant scale descriptor
        cd_mode, scale_dtype, scale_base = self._cd(self.C[d["CD"]])
        if dequant:
            if cd_mode != 1:
                raise NotImplementedError("only per-128-group dequant (CD mode=1)")
            G = K // 128
            if K % 128:
                raise ValueError("dequant requires K multiple of 128")
            # W4A16 (srcB=INT4) requires BF16/FP16 activations with an fp32
            # accumulator (02 §6 / 04 §1.2): INT4 weights x fp activation, no
            # INT32 in-group accumulation, no activation quantization.
            if srcB == I.DT_INT4:
                if srcA not in (I.DT_BF16, I.DT_FP16) or acc != I.ACC_FP32:
                    raise NotImplementedError(
                        "W4A16 requires srcA=BF16/FP16, acc=FP32 "
                        f"(got srcA={I.DTYPE_NAMES[srcA]} "
                        f"acc={I.ACC_NAMES[acc]})")

        # Output element representation (04 §1.5 post-process "fp32 -> target
        # dtype"): the dequant path writes BF16 (v0 activation dtype); the
        # non-dequant BF16/FP16 path writes srcA dtype (rounded from the fp32
        # accumulator); non-dequant INT8 (acc=INT32) writes the exact INT32
        # accumulator. The executor no longer writes fp32 in either fp path.
        if dequant:
            out_dtype = I.DT_BF16
        elif srcA in (I.DT_BF16, I.DT_FP16):
            out_dtype = srcA
        else:
            out_dtype = I.DT_INT32  # INT8xINT8 -> INT32 (exact)
        out_esz = DTYPE_SIZE[out_dtype]
        if row_stride_c == 0:
            row_stride_c = N * out_esz

        for b in range(batch):
            A = self.read_matrix(a_kind, a_base + b * batch_stride_a, srcA,
                                 M, K, row_stride_a, transpose_a)
            B = self.read_matrix(b_kind, b_base + b * batch_stride_b, srcB,
                                 K, N, row_stride_b, transpose_b)
            if dequant:
                C = self._gemm_dequant(A, B, M, N, K, G, scale_dtype,
                                       scale_base, w4=(srcB == I.DT_INT4))
            elif srcA == I.DT_INT8 and srcB == I.DT_INT8 and acc == I.ACC_INT32:
                C = np.matmul(A.astype(np.int32), B.astype(np.int32),
                              dtype=np.int32)
            elif srcA in (I.DT_BF16, I.DT_FP16) and acc == I.ACC_FP32:
                C = np.matmul(A.astype(np.float32), B.astype(np.float32),
                              dtype=np.float32)
            elif srcA == I.DT_FP16 and acc == I.ACC_FP16:
                C = np.matmul(A.astype(np.float16), B.astype(np.float16),
                              dtype=np.float16)
            else:
                raise NotImplementedError(
                    f"MATRIX dtype combo srcA={I.DTYPE_NAMES[srcA]} "
                    f"srcB={I.DTYPE_NAMES[srcB]} acc={I.ACC_NAMES[acc]}")

            c_addr = c_base + b * batch_stride_c
            if acc_init:
                # fresh: clear C then write (each batch element its own C)
                self.write_matrix(c_kind, c_addr, out_dtype, C, row_stride_c)
            else:
                prev = self.read_matrix(c_kind, c_addr, out_dtype, M, N,
                                       row_stride_c, False)
                C = prev.astype(np.float32) + C.astype(np.float32)
                self.write_matrix(c_kind, c_addr, out_dtype, C, row_stride_c)

    def _gemm_dequant(self, A: np.ndarray, B: np.ndarray, M, N, K, G,
                      scale_dtype, scale_base, w4: bool = False) -> np.ndarray:
        """Per-128-K-group dequant (02 §6 / 04 §1.5).

        W8A8 (w4=False): INT32 in-group accumulation -> fp32 scale -> fp32
        cross-group accumulation.
        W4A16 (w4=True): fp32 in-group accumulation (BF16 activation x INT4
        weight), same per-128-group fp32 scale. The activation stays fp32 — it
        must NOT be cast through int8_group_partials' astype(int32) truncation.
        """
        scale_np = np.float16 if scale_dtype == 1 else _BF16
        scale_esz = 2
        # scale array layout: [N, G] row-major (scale[n*G + g])
        raw = self.read_bytes("sram", scale_base, N * G * scale_esz)
        scales = np.frombuffer(raw, dtype=scale_np).astype(np.float64)
        scales = scales.reshape(N, G)  # (N, G)
        C = np.zeros((M, N), dtype=np.float32)
        if w4:
            Af = A.astype(np.float32)
            for g in range(G):
                partial = np.matmul(
                    Af[:, g * 128:(g + 1) * 128],
                    B[g * 128:(g + 1) * 128, :].astype(np.float32),
                    dtype=np.float32)
                C += (scales[:, g].astype(np.float32)[None, :] * partial)
        else:
            for g, partial in enumerate(int8_group_partials(A, B, G)):
                # fp32 scale (per output column) then fp32 accumulate across groups
                C += (scales[:, g].astype(np.float32)[None, :]
                      * partial.astype(np.float32))
        return C

    def _vector(self, d: dict):
        op = d["opcode"]
        srcA = d["srcA"]
        srcB = d["srcB"]
        acc = d["acc"]
        n = d["len"]
        cv = d["CV"]
        imm = d["imm"]
        a_kind, a_base = self.resolve(d["ARa"])
        d_kind, d_base = self.resolve(d["ARd"])

        # causal mask tile generation (rows/cols + global base from imm/C[CV])
        if op == I.OP_VMASK:
            cval = self.C[cv]
            col_base = (cval >> 16) & 0xFFFF
            row_base = cval & 0xFFFF
            rows = (imm >> 16) & 0xFFFF
            cols = imm & 0xFFFF
            i = np.arange(rows, dtype=np.int64)[:, None]
            j = np.arange(cols, dtype=np.int64)[None, :]
            tile = np.where((col_base + j) <= (row_base + i), 0.0, -np.inf)
            self._write_vector(d_kind, d_base, srcA, tile.reshape(-1))
            return

        # per-group reduction (softmax online max/sum); C[CV] = group count
        if op in (I.OP_VREDUCE_SUM, I.OP_VREDUCE_MAX):
            ngroups = self.C[cv] or 1
            total = ngroups * n
            a = self._read_vector(a_kind, a_base, srcA, total).astype(np.float32)
            a = a.reshape(ngroups, n)
            r = a.sum(axis=1) if op == I.OP_VREDUCE_SUM else a.max(axis=1)
            self._write_vector(d_kind, d_base, srcA, r)
            return
        if op == I.OP_ROPE:
            theta = self._c_f32(self.C[cv])
            pos = imm & 0xFFFF
            a = self._read_vector(a_kind, a_base, srcA, n)
            r = _rope_apply(a.reshape(n // 128, 128), pos, theta,
                            DTYPE_NP[srcA]).reshape(-1)
            self._write_vector(d_kind, d_base, srcA, r)
            return

        # RMSNorm (normal / per-head via imm[31]); eps = fp32 in C[CV]
        if op == I.OP_RMSNORM:
            eps = self._c_f32(self.C[cv])
            per_head = (imm >> 31) & 1
            a = self._read_vector(a_kind, a_base, srcA, n).astype(np.float32)
            g_kind, g_base = self.resolve(d["ARb"])
            g = self._read_vector(g_kind, g_base, srcB, n).astype(np.float32)
            if per_head:
                a2 = a.reshape(-1, 128)
                g2 = g.reshape(-1, 128)
                rms = np.sqrt(np.mean(a2 * a2, axis=1, keepdims=True) + eps)
                r = (a2 / rms) * g2
            else:
                rms = np.sqrt(np.mean(a * a) + eps)
                r = (a / rms) * g
            self._write_vector(d_kind, d_base, srcA, r.reshape(-1))
            return

        # QUANT (fp -> int) / DEQUANT (int -> fp); CV = scale descriptor (CD)
        if op in (I.OP_QUANT, I.OP_DEQUANT):
            mode, scale_dtype, scale_base = self._cd(self.C[cv])
            scale_np = np.float16 if scale_dtype == 1 else _BF16
            a = self._read_vector(a_kind, a_base, srcA, n).astype(np.float32)
            if mode == 0:  # per-tensor scale
                s = np.frombuffer(
                    self.read_bytes("sram", scale_base, 2), dtype=scale_np)[0]
                scales = np.full(n, float(s), dtype=np.float32)
            else:  # per-128-group scale
                gcnt = n // 128
                raw = self.read_bytes("sram", scale_base, gcnt * 2)
                scales = np.repeat(
                    np.frombuffer(raw, dtype=scale_np).astype(np.float32), 128)
            if op == I.OP_QUANT:
                out_dtype = I.DT_INT4 if srcB == I.DT_INT4 else I.DT_INT8
                q = np.round(a / scales)
                if out_dtype == I.DT_INT4:
                    q = np.clip(q, -8, 7)
                else:
                    q = np.clip(q, -127, 127)
                self._write_vector(d_kind, d_base, out_dtype, q)
            else:
                out_dtype = I.DT_FP16 if srcB == I.DT_FP16 else I.DT_BF16
                self._write_vector(d_kind, d_base, out_dtype, a * scales)
            return

        # elementwise binary (VADD/VSUB/VMUL/VDIV/VMAX)
        # cv field == 0 -> ARb holds len contiguous elements; cv field != 0 ->
        # ARb holds a single scalar b[0] broadcast across all len lanes.
        if op in (I.OP_VADD, I.OP_VSUB, I.OP_VMUL, I.OP_VDIV, I.OP_VMAX):
            b_kind, b_base = self.resolve(d["ARb"])
            a = self._read_vector(a_kind, a_base, srcA, n)
            if cv == 0:
                b = self._read_vector(b_kind, b_base, srcB, n)
            else:
                b = np.repeat(self._read_vector(b_kind, b_base, srcB, 1), n)
            if srcA in (I.DT_BF16, I.DT_FP16) or srcB in (I.DT_BF16, I.DT_FP16):
                a = a.astype(np.float32)
                b = b.astype(np.float32)
                if op == I.OP_VADD:
                    r = a + b
                elif op == I.OP_VSUB:
                    r = a - b
                elif op == I.OP_VMUL:
                    r = a * b
                elif op == I.OP_VDIV:
                    r = a / b
                else:
                    r = np.maximum(a, b)
            else:
                if op == I.OP_VDIV:
                    raise ValueError("VDIV requires fp operands")
                if op == I.OP_VADD:
                    r = a + b
                elif op == I.OP_VSUB:
                    r = a - b
                elif op == I.OP_VMUL:
                    r = a * b
                else:
                    r = np.maximum(a, b)
            self._write_vector(d_kind, d_base, srcA, r)
            return

        # elementwise unary (VRECIP/VEXP/VRSQRT/VSILU) + VMOV + VSCALE
        a = self._read_vector(a_kind, a_base, srcA, n).astype(np.float32)
        if op == I.OP_VRECIP:
            r = 1.0 / a
        elif op == I.OP_VEXP:
            r = np.exp(a)  # functional-level: numpy exp
        elif op == I.OP_VRSQRT:
            r = 1.0 / np.sqrt(a)
        elif op == I.OP_VSILU:
            # a * sigmoid(a) = a / (1 + exp(-a)); for a << 0 the exp overflows
            # to +inf and the result collapses to 0 — matching HF's F.silu.
            with np.errstate(over="ignore"):
                r = a * (1.0 / (1.0 + np.exp(-a)))
        elif op == I.OP_VMOV:
            # Same-dtype copy: v0 does no dtype conversion here. The 04 §1.5
            # "fp32 -> BF16 target dtype" conversion already happened in the
            # MATRIX post-process writeback (dequant / BF16 non-dequant path).
            r = a  # copy in srcA dtype
        elif op == I.OP_VSCALE:
            if acc == I.ACC_FP32:
                s = self._c_f32(self.C[cv])
            else:
                s = self._bf16_bits(imm & 0xFFFF)
            r = a * s
        else:
            raise NotImplementedError(f"VECTOR opcode 0x{op:02X}")
        self._write_vector(d_kind, d_base, srcA, r)

    def _kv(self, d: dict):
        op = d["opcode"]
        layer = d["layer"]
        head = d["head"]
        if op == I.OP_KV_APPEND:
            k_kind, k_addr = self.resolve(d["srcK"])
            v_kind, v_addr = self.resolve(d["srcV"])
            pos = self.C[self.C_KV_POS]
            k = self.read_bytes(k_kind, k_addr, 256)
            v = self.read_bytes(v_kind, v_addr, 256)
            self.write_bytes("hbm", self._kv_slab_base(layer, head, 0) + (pos << 8), k)
            self.write_bytes("hbm", self._kv_slab_base(layer, head, 1) + (pos << 8), v)
        elif op == I.OP_KV_STORE_BLOCK:
            k_kind, k_addr = self.resolve(d["srcK"])
            v_kind, v_addr = self.resolve(d["srcV"])
            pos_start = d["pos_start"]
            nbytes = d["count"] * 256
            k = self.read_bytes(k_kind, k_addr, nbytes)
            v = self.read_bytes(v_kind, v_addr, nbytes)
            self.write_bytes("hbm", self._kv_slab_base(layer, head, 0) + (pos_start << 8), k)
            self.write_bytes("hbm", self._kv_slab_base(layer, head, 1) + (pos_start << 8), v)
        elif op == I.OP_KV_LOAD:
            dstK_kind, dstK_addr = self.resolve(d["dstK"])
            dstV_kind, dstV_addr = self.resolve(d["dstV"])
            sel = d["sel"]
            if sel == 3:
                raise ValueError("KV.LOAD sel=3 reserved")
            pos_start = d["pos_start"]
            nbytes = d["count"] * 256
            if sel in (0, 2):
                k = self.read_bytes("hbm",
                                    self._kv_slab_base(layer, head, 0) + (pos_start << 8), nbytes)
                self.write_bytes(dstK_kind, dstK_addr, k)
            if sel in (1, 2):
                v = self.read_bytes("hbm",
                                    self._kv_slab_base(layer, head, 1) + (pos_start << 8), nbytes)
                self.write_bytes(dstV_kind, dstV_addr, v)
        elif op == I.OP_KV_GATHER:
            dst_kind, dst_addr = self.resolve(d["dst"])
            kv = d["sel"]  # 0=K, 1=V
            broadcast = d["broadcast"]
            pos_start = d["pos_start"]
            nbytes = d["count"] * 256
            data = self.read_bytes(
                "hbm", self._kv_slab_base(layer, head, kv) + (pos_start << 8), nbytes)
            self.write_bytes(dst_kind, dst_addr, data)
            if broadcast:
                stride = self.C[d["Cstride"]] * 16  # word addr -> byte stride
                for i in range(1, 4):  # GQA ×4
                    self.write_bytes(dst_kind, dst_addr + i * stride, data)
        else:
            raise NotImplementedError(f"KV opcode 0x{op:02X}")


def int8_group_partials(A_i8: np.ndarray, B_i8: np.ndarray,
                        G: int) -> list[np.ndarray]:
    """Per-128-K-group INT32 partial sums of an INT8xINT8 matmul (exact)."""
    A = A_i8.astype(np.int32)
    B = B_i8.astype(np.int32)
    out = []
    for g in range(G):
        partial = np.matmul(A[:, g * 128:(g + 1) * 128],
                            B[g * 128:(g + 1) * 128, :], dtype=np.int32)
        out.append(partial)
    return out


def load_qbin_into_executor(exe: Executor, qbin) -> None:
    """Place a qbin's weight/scale tensors into HBM at their hbm_off."""
    for t in qbin.tensors:
        exe.write_bytes("hbm", t.hbm_off, t.data)
        if t.scales is not None:
            exe.write_bytes("hbm", t.scales_hbm_off, t.scales)
