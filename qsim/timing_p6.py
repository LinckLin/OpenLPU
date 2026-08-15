"""qsim P6 timing model — hardware-aware scheduling (M5).

An *optional, additive* layer on top of `qsim/timing.py`.  It models the
KV-staging ⟂ dense-weight-stream overlap (double-buffered weight/KV tiles +
DMA PREFETCH) that the **write roofline** (488 token/s @4K, 257 @8K) assumes,
replays decode 4K/8K single-token full-model + prefill 128-block, and reports a
per-resource cycle decomposition + ablation + M5 acceptance
(write-roofline 80%: 4K ≤ 2.56 M cycles, 8K ≤ 4.86 M cycles).

Baseline-freeze contract (plans/p6-p7-plan.md §3): this module does **not**
modify the v0 cycle口径 in `qsim/timing.py` or `qsim/executor.py`; it imports the
frozen constants and re-derives the schedule.  The INT8 per-128-group activation
quantization's cycle cost (QUANT vector ops) is folded in as a hidden bucket.

Clock = 1 GHz; all figures in cycles.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from qsim.timing import (
    ARRAY_MAC_PER_CYCLE, CFG, HBM_READ_BPC, HBM_WRITE_BPC, SRAM_READ_BPC,
    SRAM_WRITE_BPC, T_FIRST, hbm_read_cycles, hbm_write_cycles,
    sram_write_cycles, matrix_compute_cycles, t_first_overhead,
    decode_layer_buckets, prefill_layer_buckets, VectorBucket,
    softmax_cycles, rmsnorm_normal_cycles, qknorm_cycles, rope_cycles,
    swiglu_cycles,
)

# =============================================================================
# Frozen per-token/per-layer resource demands (Qwen3-0.6B, 01b / roofline §2)
# =============================================================================

WEIGHT_LAYER = CFG.weight_int8_per_layer          # 15,730,944 B / layer
LM_HEAD = CFG.lm_head_params                      # 155,582,464 B (tied, still read)
KV_PER_TOKEN_LAYER = CFG.kv_bytes_per_token_per_layer   # 4096 B (K+V, 8 heads)
DENSE_MAC_LAYER = CFG.dense_params_per_layer      # 15,728,640 MAC (M=1)
KV_LOAD_TMAX = 2048                               # 05 §1.5 single-copy tile cap

# INT8 per-128-group QUANT vector-op element counts (per layer, decode):
# qkv 1024 + o 2048 + gate 1024 + up 1024 + down 3072 = 8192 elements/layer.
QUANT_ELEMS_LAYER = 8192
QUANT_ELEMS_LMHEAD = 1024


def decode_vector_bucket() -> VectorBucket:
    """Vector op cycles per decode token (v0 513) + INT8 QUANT (hidden)."""
    return VectorBucket(
        softmax=softmax_cycles(CFG.q_heads),
        rmsnorm_normal=2 * rmsnorm_normal_cycles(CFG.hidden),
        qknorm=qknorm_cycles(CFG.q_heads + CFG.kv_heads),
        rope=rope_cycles((CFG.q_out + CFG.k_out) // 128),
        swiglu=swiglu_cycles(CFG.intermediate),
        residual=2 * (CFG.hidden // 128),
    )


def quant_cycles(elems: int) -> int:
    """QUANT vector-op cycles = ceil(elems / 128) (128-lane, 1 elem/cycle)."""
    return ceil(elems / 128)


# =============================================================================
# Decode per-layer resource demands (single token, cache = ctx)
# =============================================================================

@dataclass
class DecodeDemands:
    ctx: int
    weight_read: int          # HBM read of dense weights (21,849)
    kv_read: int              # HBM read of KV window (single copy)
    kv_sram_write: int        # SRAM write of KV staging (single copy, LOAD)
    append_write: int         # HBM write of KV.APPEND (8 heads x 512 B)
    matrix: int               # array MAC/peak (dense + attention)
    vector: int               # Vector ops (513) + INT8 QUANT (hidden)
    quant: int                # INT8 QUANT vector-op cycles (hidden)

    @property
    def hbm_read_total(self) -> int:
        return self.weight_read + self.kv_read


def decode_demands(ctx: int) -> DecodeDemands:
    kv_bytes = KV_PER_TOKEN_LAYER * ctx
    attn_mac = CFG.q_heads * CFG.head_dim * ctx * 2        # QK^T + AV
    vec = decode_vector_bucket()
    quant = quant_cycles(QUANT_ELEMS_LAYER)
    return DecodeDemands(
        ctx=ctx,
        weight_read=hbm_read_cycles(WEIGHT_LAYER),
        kv_read=hbm_read_cycles(kv_bytes),
        kv_sram_write=sram_write_cycles(kv_bytes),
        append_write=hbm_write_cycles(KV_PER_TOKEN_LAYER),
        matrix=matrix_compute_cycles(DENSE_MAC_LAYER + attn_mac),
        vector=vec.total,
        quant=quant,
    )


# =============================================================================
# Scheduling strategies (per-layer wall) + full-token aggregation
# =============================================================================

def lm_head_wall() -> int:
    """Final lm_head: HBM read of 155.6 MB (no KV staging to overlap)."""
    return hbm_read_cycles(LM_HEAD)          # 216,087 cyc


def schedule_serial(d: DecodeDemands) -> int:
    """v0 serial: HBM read (weights + KV) and SRAM staging write serialized."""
    return d.weight_read + d.kv_read + d.kv_sram_write


def schedule_overlap(d: DecodeDemands) -> int:
    """KV staging SRAM write ∥ dense weight + KV window HBM read.

    The write roofline's core assumption: the SRAM write port and the HBM read
    bus are independent resources; per-layer wall = max(SRAM write, HBM read)."""
    return max(d.kv_sram_write, d.weight_read + d.kv_read)


def schedule_double_buffer(d: DecodeDemands) -> int:
    """Overlap + double-buffered tiles + DMA PREFETCH.

    Ping-pong weight/KV tiles let the next tile prefetch during the current
    tile's compute; adds per-tile HBM T_first for KV.LOAD tiling."""
    n_tiles = ceil(d.ctx / KV_LOAD_TMAX)
    return schedule_overlap(d) + t_first_overhead(n_tiles)


def schedule_kv_resident(d: DecodeDemands) -> int:
    """KV re-read reduction upper bound: KV window resident in SRAM (no
    re-read, no staging write).  Hypothetical — 8 MiB SRAM cannot hold a
    4K/8K window (16/32 MiB); listed as the ceiling the backlog targets."""
    return d.weight_read                        # HBM-bound on dense weights


def full_token_cycles(d: DecodeDemands, per_layer_wall: int,
                      pipeline_fill: int = 0) -> int:
    """28 layers x per-layer wall + lm_head (216,087) + one-time pipeline fill."""
    return CFG.layers * per_layer_wall + lm_head_wall() + pipeline_fill


# =============================================================================
# Ablation + acceptance
# =============================================================================

def ablation(ctx: int) -> dict:
    d = decode_demands(ctx)
    fill = hbm_read_cycles(WEIGHT_LAYER)      # layer-0 weight prefetch, once
    strategies = {
        "serial": ("no overlap (v0)", schedule_serial(d), 0),
        "overlap": ("KV staging ∥ weight stream", schedule_overlap(d), fill),
        "double_buffer": ("+ double buffer + DMA PREFETCH",
                          schedule_double_buffer(d), fill),
        "kv_resident": ("+ KV re-read reduction (hypothetical)",
                        schedule_kv_resident(d), 0),
    }
    out = {}
    for key, (label, wall, this_fill) in strategies.items():
        total = full_token_cycles(d, wall, this_fill)
        out[key] = {
            "label": label,
            "per_layer_wall": wall,
            "total_cycles": total,
            "token_per_s": round(1e9 / total, 1),
        }
    return out


def acceptance() -> dict:
    """M5 write-roofline 80% targets: 4K ≤ 2.56 M, 8K ≤ 4.86 M cycles."""
    targets = {4096: 2_560_000, 8192: 4_860_000}
    out = {}
    for ctx, tgt in targets.items():
        d = decode_demands(ctx)
        wall = schedule_double_buffer(d)
        total = full_token_cycles(d, wall, hbm_read_cycles(WEIGHT_LAYER))
        out[ctx] = {
            "target_cycles": tgt,
            "scheduled_cycles": total,
            "pass": total <= tgt,
            "margin_pct": round(100.0 * (tgt - total) / tgt, 2),
            "token_per_s": round(1e9 / total, 1),
        }
    return out


def prefill_replay(seq: int = 128) -> dict:
    """Prefill 128-block per-layer decomposition (compute-bound; overlap no-op)."""
    b = prefill_layer_buckets(seq)
    quant = quant_cycles(seq * QUANT_ELEMS_LAYER)  # per-row QUANT, M=128
    return {
        "seq": seq,
        "weight_stream": b.weight_stream,
        "kv_hbm_read": b.kv_hbm_read,
        "kv_sram_write": b.kv_sram_write,
        "matrix_compute": b.matrix_compute,
        "vector_total": b.vector.total,
        "quant_hidden": quant,
        "per_layer_total": b.matrix_compute + b.vector.total,
        "bound": "compute",
    }


def main() -> dict:
    r = {}
    # decode resource demands @4K / @8K
    r["decode_demands"] = {
        str(ctx): {
            "weight_read": decode_demands(ctx).weight_read,
            "kv_read": decode_demands(ctx).kv_read,
            "kv_sram_write": decode_demands(ctx).kv_sram_write,
            "append_write": decode_demands(ctx).append_write,
            "matrix": decode_demands(ctx).matrix,
            "vector": decode_demands(ctx).vector,
            "quant_hidden": decode_demands(ctx).quant,
            "hbm_read_total": decode_demands(ctx).hbm_read_total,
            "lm_head": lm_head_wall(),
        }
        for ctx in (4096, 8192)
    }
    r["ablation_4K"] = ablation(4096)
    r["ablation_8K"] = ablation(8192)
    r["acceptance"] = acceptance()
    r["prefill"] = prefill_replay(128)
    return r


def print_report(r: dict) -> None:
    print("=== decode per-layer resource demands (cycles) ===")
    for ctx, d in r["decode_demands"].items():
        print(f"  ctx {ctx:>4}: weight_read {d['weight_read']:>7} "
              f"kv_read {d['kv_read']:>7} kv_sram_write {d['kv_sram_write']:>7} "
              f"append {d['append_write']:>3} matrix {d['matrix']:>4} "
              f"vector {d['vector']:>4} quant(hidden) {d['quant_hidden']:>3} "
              f"hbm_read_total {d['hbm_read_total']:>7} lm_head {d['lm_head']:>7}")

    print("\n=== ablation (per-token full model, cycles -> token/s) ===")
    for key in ("serial", "overlap", "double_buffer", "kv_resident"):
        a4 = r["ablation_4K"][key]
        a8 = r["ablation_8K"][key]
        print(f"  {key:>15} ({a4['label']:>38}): "
              f"@4K {a4['total_cycles']:>9} cyc {a4['token_per_s']:>7} tok/s | "
              f"@8K {a8['total_cycles']:>9} cyc {a8['token_per_s']:>7} tok/s")

    print("\n=== M5 acceptance (write-roofline 80%) ===")
    for ctx, a in r["acceptance"].items():
        print(f"  ctx {ctx:>4}: target {a['target_cycles']:>9} cyc | "
              f"scheduled {a['scheduled_cycles']:>9} cyc | "
              f"{'PASS' if a['pass'] else 'FAIL'} "
              f"(margin {a['margin_pct']}%, {a['token_per_s']} tok/s)")

    p = r["prefill"]
    print(f"\n=== prefill 128-block per layer: matrix {p['matrix_compute']} + "
          f"vector {p['vector_total']} = {p['per_layer_total']} cyc "
          f"({p['bound']}-bound; QUANT hidden {p['quant_hidden']})")


if __name__ == "__main__":
    print_report(main())
