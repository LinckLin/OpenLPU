"""qrun runtime: tokenizer, host embedding, host sampling, KV bootstrap, and
the per-token DC program patches (C_KV_POS / ROPE pos / KV.LOAD count / tail
mask) that make the fixed DC program track the decode position.

The DC program is static; per token (position `pos`, window C = pos+1) qrun:
  * patches CONFIG C_KV_POS = pos
  * patches every ROPE imm[15:0] = pos
  * patches every KV.LOAD count = clamp(C - pos_start, 0, 2048)
  * rewrites the [65, 2, 128] BF16 tail mask (0 in-window, -inf out-of-window)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compiler.isa import isa as I
from qrun import bf16 as B
from qrun import program as P

H = 1024
VOCAB = 151936
BLOCK = 128
N_TILE = 128

_CKVPOS_BITS = 33                    # CONFIG IMM64 low bit
_IMM32_BITS = 33                     # VECTOR imm low bit
_KVLOAD_COUNT_BITS = 54              # KV.LOAD count low bit


@dataclass
class DcPatch:
    """Pre-decoded patch locations in the DC program."""
    words: list[int]
    ckv_pos_idx: int | None = None
    rope_idx: list[int] = field(default_factory=list)
    kvload: list[tuple[int, int]] = field(default_factory=list)  # (idx, pos_start)


def build_dc_patch(program: bytes) -> DcPatch:
    words = [int.from_bytes(program[o:o + 16], "little")
             for o in range(0, len(program), 16)]
    p = DcPatch(words=words)
    for i, w in enumerate(words):
        d = I.decode_inst(w)
        if d["mnemonic"] == "CONFIG" and d["reg_class"] == 0 and d["REG"] == 30:
            if p.ckv_pos_idx is None:
                p.ckv_pos_idx = i
        elif d["mnemonic"] == "ROPE":
            p.rope_idx.append(i)
        elif d["mnemonic"] == "KV.LOAD":
            p.kvload.append((i, d["pos_start"]))
    assert p.ckv_pos_idx is not None, "DC program missing C_KV_POS CONFIG"
    return p


def patch_dc(p: DcPatch, pos: int) -> bytes:
    """Return the DC program bytes patched for decode position `pos`."""
    C = pos + 1                       # causal window length (incl. current)
    words = list(p.words)
    words[p.ckv_pos_idx] = ((words[p.ckv_pos_idx]
                             & ~(((1 << 64) - 1) << _CKVPOS_BITS))
                            | (pos << _CKVPOS_BITS))
    for i in p.rope_idx:
        words[i] = ((words[i] & ~(0xFFFF << _IMM32_BITS))
                    | ((pos & 0xFFFF) << _IMM32_BITS))
    for i, ps in p.kvload:
        count = max(0, min(2048, C - ps))
        words[i] = ((words[i] & ~(((1 << 14) - 1) << _KVLOAD_COUNT_BITS))
                    | (count << _KVLOAD_COUNT_BITS))
    return b"".join(w.to_bytes(16, "little") for w in words)


def build_tail_mask(C: int) -> bytes:
    """[65, 2, 128] BF16 tail mask: 0 in-window, -inf out-of-window."""
    mask = np.full((P.N_SUBTILES, 2, N_TILE), -np.inf, dtype=B.BF16_NP)
    for s in range(P.N_SUBTILES):
        base = s * N_TILE
        nvalid = max(0, min(N_TILE, C - base))
        if nvalid > 0:
            mask[s, :, :nvalid] = 0
    return mask.tobytes()


class RunEngine:
    """End-to-end qrun engine over the qsim backend."""

    def __init__(self, qmetal, plan, dtype: str, layouts, st, tokenizer,
                 ref=None):
        self.qmetal = qmetal
        self.plan = plan
        self.dtype = dtype
        self.layouts = layouts
        self.st = st
        self.tokenizer = tokenizer
        self.ref = ref                    # Qwen3Ref (torch) for bootstrap/HF ref
        self.embed = st.get_float32("model.embed_tokens.weight")  # [151936,1024]
        self.pf_program = None
        self.dc_program = None
        self.dc_patch = None

    def build_programs(self):
        self.pf_program = P.lower_transformer("PF", self.layouts, self.dtype,
                                              self.plan)
        self.dc_program = P.lower_transformer("DC", self.layouts, self.dtype,
                                              self.plan)
        self.dc_patch = build_dc_patch(self.dc_program)

    # -- host embedding --------------------------------------------------
    def embed_ids(self, token_ids) -> np.ndarray:
        """token_ids: [seq] -> hidden [seq, 1024] fp32 (exact from BF16 table)."""
        return self.embed[token_ids]

    def write_input_hbm(self, hidden: np.ndarray, M: int):
        assert hidden.shape[1] == H and hidden.shape[0] <= M
        buf = np.zeros((M, H), dtype=np.float32)
        buf[:hidden.shape[0]] = hidden
        self.qmetal.write_hbm(self.plan.input_hbm,
                              np.asarray(buf, dtype=B.BF16_NP).tobytes())

    # -- logits readback -------------------------------------------------
    def read_logits(self, M: int, row: int) -> np.ndarray:
        ntiles = VOCAB // N_TILE
        parts = []
        for t in range(ntiles):
            off = self.plan.logits_hbm + t * M * N_TILE * 2 + row * 256
            parts.append(self.qmetal.read_hbm(off, 256))
        return B.bf16_bytes_to_fp32(b"".join(parts))

    def argmax(self, logits: np.ndarray) -> int:
        return int(np.argmax(logits))

    # -- prefill (real PF, one 128-token block) --------------------------
    def prefill(self, token_ids: np.ndarray) -> np.ndarray:
        """Run the PF program over token_ids (<=128); return logits[last]."""
        seq = token_ids.shape[0]
        hidden = self.embed_ids(token_ids)
        self.write_input_hbm(hidden, BLOCK)
        self.qmetal.run_pf(self.pf_program)
        return self.read_logits(BLOCK, seq - 1)

    # -- KV bootstrap (reference model writes K/V into the HBM slab) -----
    def bootstrap_kv(self, token_ids):
        """Run Qwen3Ref prefill; write K/V slabs. token_ids: [seq] torch int64."""
        assert self.ref is not None
        _, cache = self.ref.forward(token_ids)
        for L in range(28):
            kn = cache[L]["k"].detach().float().cpu().numpy()   # [8, seq, 128]
            vn = cache[L]["v"].detach().float().cpu().numpy()
            for h in range(8):
                self.qmetal.write_hbm(
                    self.qmetal.kv_addr(L, h, 0, 0),
                    np.asarray(kn[h], dtype=B.BF16_NP).tobytes())
                self.qmetal.write_hbm(
                    self.qmetal.kv_addr(L, h, 1, 0),
                    np.asarray(vn[h], dtype=B.BF16_NP).tobytes())

    # -- decode ----------------------------------------------------------
    def decode_step(self, pos: int, token_id: int) -> np.ndarray:
        """One DC step: embed token_id, patch DC program for pos, run, return
        logits [151936] fp32."""
        hidden = self.embed_ids(np.array([token_id], dtype=np.int64))
        self.write_input_hbm(hidden, 1)
        self.qmetal.write_sram(P.MASK_BASE, build_tail_mask(pos + 1))
        self.qmetal.run_dc(patch_dc(self.dc_patch, pos))
        return self.read_logits(1, 0)

    # -- generate --------------------------------------------------------
    def generate(self, prompt_ids: np.ndarray, max_new: int,
                 *, bootstrap: bool = False):
        """Greedy decode. Returns (new_tokens, logits_list).

        bootstrap=False: real PF prefill (prompt <= 128 tokens).
        bootstrap=True:  KV bootstrap (prompt > 128 tokens); the last prompt
                         token's logits are still computed by the qsim DC
                         program (the DC step also re-appends that token's K/V,
                         which for BF16 is ~1-ULP-equal to the bootstrapped K/V).
        """
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        new_tokens: list[int] = []
        logits_list: list[np.ndarray] = []

        if bootstrap:
            import torch
            self.bootstrap_kv(torch.from_numpy(prompt_ids))
            pos = prompt_ids.shape[0] - 1          # position of last prompt token
            logits = self.decode_step(pos, int(prompt_ids[-1]))
            pos += 1
        else:
            logits = self.prefill(prompt_ids)
            pos = prompt_ids.shape[0]

        for _ in range(max_new):
            tok = self.argmax(logits)
            new_tokens.append(tok)
            logits_list.append(logits)
            logits = self.decode_step(pos, tok)
            pos += 1
        return new_tokens, logits_list
