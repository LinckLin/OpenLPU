"""QMetal runtime: QCore device control over the qsim functional backend.

Responsibilities (plans/p5-plan.md §4 task 1):
  * HBM slab allocation (05 §1.3 formula) for INPUT / LOGITS / KV regions
  * tensor loading (INT8 from qbin, BF16 from safetensors)
  * command queue: PF once -> DC per token
  * device control via the qsim `Executor` functional backend

The RTL interface is explicitly deferred to P7; this module drives the qsim
backend (functional, numpy) and keeps the device-control surface small so an
RTL backend can replace it later.

One deliberate divergence from `qsim/executor.py`: the KV slab stride is a
per-instance parameter (`slab_shift`, bytes = 1 << slab_shift). The 8K golden
(`decode_seq1_cache8192`) decodes token position **8192** — i.e. the 8193rd KV
slot — while the frozen v0 slab (SLAB_SHIFT=21, 2 MiB) holds only 8192 slots
(pos 0..8191). qrun therefore uses `slab_shift=22` (4 MiB slabs) so pos 8192
does not wrap into the neighbouring slab. This is a load-time allocation
choice, not an ISA change, and is reported as a review item.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qsim.executor import Executor

# frozen model card (01b) — must match qforge/program.py
H = 1024
LAYERS = 28
KVH = 8
HD = 128


class QMetalExecutor(Executor):
    """Functional executor with a parameterised KV slab stride."""

    def __init__(self, slab_shift: int = 22):
        super().__init__()
        self.slab_shift = slab_shift

    def _kv_slab_base(self, layer: int, head: int, kv: int) -> int:
        _, base = self.resolve(self.AR_KV_BASE)
        slab_index = (layer << 4) | (head << 1) | kv
        return base + (slab_index << self.slab_shift)


@dataclass
class HbmPlan:
    """Final HBM allocation (qrun re-plans the v0 placeholders)."""
    weights_end: int
    input_hbm: int
    input_bytes: int
    logits_hbm: int
    logits_bytes: int
    kv_base: int
    kv_bytes: int
    slab_shift: int


class QMetal:
    """Device handle: owns the executor + the final HBM address plan."""

    def __init__(self, slab_shift: int = 22):
        self.exe = QMetalExecutor(slab_shift)
        self.slab_shift = slab_shift
        self.plan: HbmPlan | None = None

    def slab_bytes(self) -> int:
        return 1 << self.slab_shift

    def kv_addr(self, layer: int, head: int, kv: int, pos: int) -> int:
        """HBM byte address of (layer, head, K/V, pos) per 05 §1.3."""
        return self.exe._kv_slab_base(layer, head, kv) + (pos << 8)

    # -- HBM plan ---------------------------------------------------------
    def plan_hbm(self, weights_end: int, *, input_bytes: int,
                 logits_bytes: int) -> HbmPlan:
        """Place INPUT / LOGITS / KV regions after the weights."""

        def align(x: int, a: int) -> int:
            return (x + a - 1) // a * a

        input_hbm = align(weights_end, 64)
        logits_hbm = align(input_hbm + input_bytes, 64)
        kv_base = align(logits_hbm + logits_bytes, 1 << self.slab_shift)
        n_slabs = LAYERS * KVH * 2
        kv_bytes = n_slabs * self.slab_bytes()
        plan = HbmPlan(weights_end=weights_end, input_hbm=input_hbm,
                       input_bytes=input_bytes, logits_hbm=logits_hbm,
                       logits_bytes=logits_bytes, kv_base=kv_base,
                       kv_bytes=kv_bytes, slab_shift=self.slab_shift)
        self.plan = plan
        # make the KV base visible to bootstrap writes before any program runs
        self.exe.AR[63] = (1 << 63) | kv_base
        return plan

    # -- tensor loading ---------------------------------------------------
    def load_tensor_hbm(self, hbm_off: int, data: bytes):
        self.exe.write_bytes("hbm", hbm_off, data)

    def load_int8_tensors(self, qbin) -> int:
        """Place the qbin's INT8 weight/scale tensors; return weights_end."""
        end = 0
        for t in qbin.tensors:
            self.load_tensor_hbm(t.hbm_off, t.data)
            end = max(end, t.hbm_off + len(t.data))
            if t.scales is not None:
                self.load_tensor_hbm(t.scales_hbm_off, t.scales)
                end = max(end, t.scales_hbm_off + len(t.scales))
        return end

    def write_sram(self, byte_addr: int, data: bytes):
        self.exe.write_bytes("sram", byte_addr, data)

    def write_hbm(self, byte_addr: int, data: bytes):
        self.exe.write_bytes("hbm", byte_addr, data)

    def read_hbm(self, byte_addr: int, n: int) -> bytes:
        return self.exe.read_bytes("hbm", byte_addr, n)

    def read_sram(self, byte_addr: int, n: int) -> bytes:
        return self.exe.read_bytes("sram", byte_addr, n)

    # -- command queue ----------------------------------------------------
    def run_pf(self, program: bytes):
        self.exe.run(program)

    def run_dc(self, program: bytes):
        self.exe.run(program)
