"""Compute P1.4 roofline and write docs/p1/roofline.md.

All hardware constants are quoted from spec §3/§4; model numbers are recomputed
for Qwen3-0.6B by the same method as spec 04 §2.4 (with the 8B design cap noted).
"""
import os

# ---- hardware constants (spec §3 memory / §4 exec-engines) ----
INT8_TMACS = 32.77e12      # INT8 array peak (spec §4)
BF16_TMACS = 8.19e12       # BF16 array peak (spec §4)
HBM_PEAK = 1.2e12          # aggregate peak B/s (spec §4)
HBM_READ_SUST = 720e9      # sustained read B/s (spec §3.4: 900 x 0.8)
SRAM_WRITE_BPC = 256       # SRAM write B/cycle (spec §4)
SRAM_READ_BPC = 512        # SRAM read B/cycle
DC_WEIGHT_DEMAND = 32.77e12  # DC weight stream demand B/s (spec §2.2)
SPEC_8B_INT8 = 7.57e9      # 8B design cap decode weight read (spec §4)

# ---- Qwen3-0.6B model card ----
HIDDEN = 1024
INTER = 3072
N_HEADS = 16
N_KV = 8
HEAD_DIM = 128
VOCAB = 151936
LAYERS = 28

Q = N_HEADS * HEAD_DIM     # 2048
KV = N_KV * HEAD_DIM       # 1024

per_layer_dense = Q * HIDDEN + KV * HIDDEN + KV * HIDDEN + HIDDEN * Q \
    + INTER * HIDDEN + INTER * HIDDEN + HIDDEN * INTER          # 15,728,640
rmsnorm_per_layer = HIDDEN + HIDDEN + HEAD_DIM + HEAD_DIM       # 2304
lm_head = VOCAB * HIDDEN                                       # 155,582,464
total_unique = per_layer_dense * LAYERS + rmsnorm_per_layer * LAYERS + lm_head

GB = 1e9
MiB = 2**20

# ---- decode per-token weight read ----
dec_int8 = per_layer_dense * LAYERS + rmsnorm_per_layer * LAYERS + lm_head
dec_bf16 = dec_int8 * 2

dec_int8_tok_s_sust = HBM_READ_SUST / dec_int8
dec_int8_tok_s_peak = HBM_PEAK / dec_int8
dec_bf16_tok_s_sust = HBM_READ_SUST / dec_bf16
dec_bf16_tok_s_peak = HBM_PEAK / dec_bf16

# ---- per-layer decode (INT8, sustained) ----
layer_weight_int8 = per_layer_dense + rmsnorm_per_layer
layer_read_sust_s = layer_weight_int8 / HBM_READ_SUST
layer_compute_s = per_layer_dense / INT8_TMACS
lm_read_sust_s = lm_head / HBM_READ_SUST
lm_compute_s = lm_head / INT8_TMACS
total_decode_sust_s = LAYERS * layer_read_sust_s + lm_read_sust_s

# ---- prefill per layer (seq=128) ----
S = 128
prefill_attn_mac = N_HEADS * S * S * HEAD_DIM * 2   # QK^T + AV
prefill_dense_mac = S * per_layer_dense
prefill_total_mac = prefill_dense_mac + prefill_attn_mac
prefill_compute_s = prefill_total_mac / INT8_TMACS
prefill_weight_read_s = layer_weight_int8 / HBM_PEAK

# ---- per-op table (decode M=1) ----
ops = [
    ("q_proj", 1, Q, HIDDEN),
    ("k_proj", 1, KV, HIDDEN),
    ("v_proj", 1, KV, HIDDEN),
    ("o_proj", 1, HIDDEN, Q),
    ("gate_proj", 1, INTER, HIDDEN),
    ("up_proj", 1, INTER, HIDDEN),
    ("down_proj", 1, HIDDEN, INTER),
]


def op_flops(m, n, k):
    return 2 * m * n * k


L = []
a = L.append

a("# P1.4 Roofline — Qwen3-0.6B（验证模型）")
a("")
a("> 口径：硬件常数引用 spec §3/§4（冻结）；模型数字按 spec 04 §2.4 同一方法对 0.6B 重算，")
a("> 并注明与 8B 设计上限的关系。所有单位 1 GHz 时钟。")
a("")
a("## 1. 硬件常数（spec 引用）")
a("")
a("| 量 | 值 | 出处 |")
a("|---|---|---|")
a(f"| INT8 阵列峰值 | {INT8_TMACS/1e12:.2f} TMAC/s | spec §4 速查 |")
a(f"| BF16 阵列峰值 | {BF16_TMACS/1e12:.2f} TMAC/s | spec §4 速查 |")
a(f"| HBM 聚合峰值 | {HBM_PEAK/1e12:.2f} TB/s（读 900 / 写 300 GB/s） | spec §4 |")
a(f"| HBM sustained 读 | {HBM_READ_SUST/1e9:.0f} GB/s（900×80%） | spec §3.4 |")
a(f"| SRAM 读 / 写 | {SRAM_READ_BPC} / {SRAM_WRITE_BPC} B/cyc | spec §4 |")
a(f"| DC 权重流需求 | {DC_WEIGHT_DEMAND/1e12:.2f} TB/s = {DC_WEIGHT_DEMAND/HBM_PEAK:.1f}× HBM | spec §2.2 |")
a("")
a("## 2. 模型参数（Qwen3-0.6B，与 01b 模型卡一致）")
a("")
a("| 量 | 值 |")
a("|---|---|")
a(f"| hidden / intermediate | {HIDDEN} / {INTER} |")
a(f"| Q heads / KV heads / head_dim | {N_HEADS} / {N_KV} / {HEAD_DIM}（GQA 2:1） |")
a(f"| vocab / layers | {VOCAB} / {LAYERS} |")
a(f"| q_proj / k_proj / v_proj / o_proj | {Q}×{HIDDEN} / {KV}×{HIDDEN} / {KV}×{HIDDEN} / {HIDDEN}×{Q} |")
a(f"| gate / up / down | {INTER}×{HIDDEN} / {INTER}×{HIDDEN} / {HIDDEN}×{INTER} |")
a(f"| 每层 dense | {per_layer_dense:,} 参数 |")
a(f"| 28 层 dense | {per_layer_dense*LAYERS:,} 参数 |")
a(f"| lm_head（tied，仍每 token 全读） | {lm_head:,} 参数 |")
a(f"| 每层 RMSNorm（in/post/q/k） | {rmsnorm_per_layer} 参数 |")
a(f"| **总参数量** | **{total_unique:,} ≈ 0.596B** |")
a("")
a("## 3. decode 每 token 权重读（0.6B vs 8B 设计上限交叉验证）")
a("")
a("| dtype | 每层 (B) | 28 层 (B) | lm_head (B) | **每 token 合计** | 8B 上限占比 |")
a("|---|---|---|---|---|---|")
a(f"| INT8 | {layer_weight_int8:,} | {per_layer_dense*LAYERS:,} | {lm_head:,} | **{dec_int8:,} B = {dec_int8/GB:.3f} GB** | {dec_int8/SPEC_8B_INT8*100:.2f}% |")
a(f"| BF16 | {layer_weight_int8*2:,} | {per_layer_dense*LAYERS*2:,} | {lm_head*2:,} | **{dec_bf16:,} B = {dec_bf16/GB:.3f} GB** | {dec_bf16/(SPEC_8B_INT8*2)*100:.2f}% |")
a("")
a(f"> spec §4 的 8B 设计上限 = **7.57 GB/token（INT8，dense 6.95G + lm_head 622M，不含 embedding）**。")
a(f"> 0.6B 的 {dec_int8/GB:.3f} GB = 8B 的 **{dec_int8/SPEC_8B_INT8*100:.2f}%**。embedding 仅首 token 读单行 1024 元素（2 KB），忽略；")
a(f"> RMSNorm 每层 {rmsnorm_per_layer} B 计入（占比 {rmsnorm_per_layer*LAYERS/dec_int8*100:.3f}%）。")
a("")
a("### 3.1 decode token/s 天花板（HBM 带宽 ÷ 每 token 字节）")
a("")
a("| dtype | 峰值 1.2 TB/s | sustained 720 GB/s |")
a("|---|---|---|")
a(f"| INT8 | {dec_int8_tok_s_peak:.0f} token/s | {dec_int8_tok_s_sust:.0f} token/s |")
a(f"| BF16 | {dec_bf16_tok_s_peak:.0f} token/s | {dec_bf16_tok_s_sust:.0f} token/s |")
a("")
a(f"> 对比 8B 设计上限：INT8 sustained {HBM_READ_SUST/SPEC_8B_INT8:.0f} token/s（spec §4）。")
a(f"> 0.6B 放大 {dec_int8_tok_s_sust/(HBM_READ_SUST/SPEC_8B_INT8):.1f}×（每 token 权重读小 {SPEC_8B_INT8/dec_int8:.1f}×，token/s 高约 {SPEC_8B_INT8/dec_int8:.1f}×，二者互为倒数）。")
a("")
a("## 4. 每层 GEMM op FLOPs / Bytes（decode M=1）")
a("")
a("| op | M×N×K | MAC | FLOPs | 权重字节 (INT8) | 算术强度 (FLOP/B) |")
a("|---|---|---|---|---|---|")
for name, m, n, k in ops:
    mac = m * n * k
    fl = op_flops(m, n, k)
    wb = n * k
    a(f"| {name} | {m}×{n}×{k} | {mac:,} | {fl:,} | {wb:,} | {fl/wb:.2f} |")
a(f"| **每层 dense 合计** | — | {per_layer_dense:,} | {2*per_layer_dense:,} | {per_layer_dense:,} | 2.00 |")
a("")
a("attention（decode cache=L，无权重、读 KV）：")
a("")
a("| op | per-head GEMM | 每层 MAC (16 head) | KV 读 (BF16) |")
a("|---|---|---|---|")
a(f"| QK^T | 1×L×128 | 16×1×L×128 | 8×(L+1)×128×2 B |")
a(f"| AV | 1×128×L | 16×1×128×L | 8×(L+1)×128×2 B |")
a(f"| L=4096 合计 | — | {N_HEADS*4096*HEAD_DIM*2:,} MAC | {N_KV*HEAD_DIM*2*(4096+1)*2:,} B/层 |")
a("")
a("## 5. prefill / decode 逐层瓶颈")
a("")
a("### 5.1 prefill（seq=128，MODE PF，整块 GEMM）")
a("")
a("| 量 | 值 |")
a("|---|---|")
a(f"| 每层 dense MAC | {prefill_dense_mac:,}（128×每层参数） |")
a(f"| 每层 attention MAC（QK^T+AV） | {prefill_attn_mac:,} |")
a(f"| 每层总 MAC | {prefill_total_mac:,} ≈ {prefill_total_mac/1e9:.2f} G |")
a(f"| 计算时间（INT8 峰值） | {prefill_compute_s*1e6:.1f} µs |")
a(f"| 权重读时间（读一次，峰值） | {prefill_weight_read_s*1e6:.1f} µs |")
a(f"| 权重复用因子 | M=128（每权重被 128 输出行复用） |")
a(f"| **瓶颈结论** | **计算受限**（{prefill_compute_s*1e6:.1f} µs > {prefill_weight_read_s*1e6:.1f} µs） |")
a("")
a("> 与 spec §2.1 结论一致：prefill M≫1 权重复用，HBM 权重流量 = 计算流量/M，远低于 1.2 TB/s，")
a("> 阵列利用率 ~94–99%。")
a("")
a("### 5.2 decode（seq=1，MODE DC，16 lane GEMV）")
a("")
a("| 量 | 值 |")
a("|---|---|")
a(f"| 每层权重读（INT8） | {layer_weight_int8:,} B |")
a(f"| 每层权重读时间（sustained） | {layer_read_sust_s*1e6:.2f} µs |")
a(f"| 每层计算（INT8 峰值） | {layer_compute_s*1e6:.2f} µs |")
a(f"| 读:算比（sustained 口径） | {layer_read_sust_s/layer_compute_s:.1f}× |")
a(f"| lm_head 权重读 / 计算 | {lm_read_sust_s*1e6:.1f} µs / {lm_compute_s*1e6:.2f} µs |")
a(f"| 每 token 总时长（sustained） | {total_decode_sust_s*1e6:.0f} µs → {1/total_decode_sust_s:.0f} token/s |")
a(f"| 阵列利用率 | {HBM_PEAK/DC_WEIGHT_DEMAND*100:.2f}%（1200/32768） |")
a(f"| **瓶颈结论** | **HBM 权重流受限**（每层读 {layer_read_sust_s*1e6:.2f} µs ≫ 算 {layer_compute_s*1e6:.2f} µs） |")
a("")
a("> 与 spec §2.2 一致：DC 权重流需求 32.77 TB/s vs HBM 1.2 TB/s = **27.3× 短缺**（硬件常数，与模型无关），")
a("> 阵列被迫降至 1.2 TMAC/s，利用率 3.66%。读:算比 45.5× 为 sustained（720 GB/s）口径，")
a("> 27.3× 为峰值（1.2 TB/s）口径，二者同源。")
a("")
a("## 6. 需评审项：decode KV 重读流量")
a("")
a("> spec 04 §2.4 注「KV 流量每层 ≈ 8 KB（KV.APPEND 写 + KV.GATHER 读当前 token）」与 05 §5.2 步 5")
a("> 「KV.GATHER 载入窗口 [0,pos]」不一致：decode 每 token 实际从 HBM 重读**整个** K/V 窗口，")
a("> 而非仅当前 token。按 05 口径，长上下文下 KV 重读不可忽略：")
a("")
a("| context | KV 重读/层 (K+V, BF16) | 28 层重读/token | vs 权重读 0.596 GB |")
a("|---|---|---|---|")
for Lctx in [128, 512, 1024, 2048, 4096, 8192]:
    per_layer = N_KV * HEAD_DIM * 2 * (Lctx + 1) * 2
    total = per_layer * LAYERS
    a(f"| {Lctx} | {per_layer:,} B | {total/GB:.3f} GB | {total/dec_int8*100:.0f}% |")
a("")
a(f"> 8K 时 KV 重读 = {N_KV*HEAD_DIM*2*(8192+1)*2*LAYERS/GB:.2f} GB/token，约为权重读的 {N_KV*HEAD_DIM*2*(8192+1)*2*LAYERS/dec_int8*100:.0f}%。")
a("> 若计入，decode token/s 天花板需下调；建议 P3 时序模型按 05 口径核算 KV 重读，并回填 04 §2.4 的 KV 注记。")
a("")
a("## 7. 结论")
a("")
a("- **prefill 每层 = 计算受限**（M=128 权重复用，阵列利用率高），与 spec §2.1/§2.5 一致。")
a("- **decode 每层 = HBM 权重流受限**（27.3× 短缺，利用率 3.66%），与 spec §2.2/§2.5 一致，为硬件常数、与模型无关。")
a(f"- decode INT8 sustained 天花板 ≈ **{dec_int8_tok_s_sust:.0f} token/s**（0.6B），8B 设计上限为 {HBM_READ_SUST/SPEC_8B_INT8:.0f} token/s；")
a(f"  0.6B 每 token 权重读 {dec_int8/GB:.3f} GB = 8B 7.57 GB 的 {dec_int8/SPEC_8B_INT8*100:.1f}%。")
a("- 需评审项：decode KV 重读（05 §5.2 全窗口 vs 04 §2.4「≈8 KB」）影响长上下文 token/s 口径，待 P3 精化。")

os.makedirs("docs/p1", exist_ok=True)
with open("docs/p1/roofline.md", "w") as f:
    f.write("\n".join(L) + "\n")

print(f"per-layer dense = {per_layer_dense:,}")
print(f"decode INT8/token = {dec_int8:,} B = {dec_int8/GB:.4f} GB (8B cap ratio {dec_int8/SPEC_8B_INT8*100:.2f}%)")
print(f"decode INT8 sustained = {dec_int8_tok_s_sust:.1f} token/s, peak = {dec_int8_tok_s_peak:.1f}")
print(f"layer decode: read {layer_read_sust_s*1e6:.2f} us vs compute {layer_compute_s*1e6:.2f} us (x{layer_read_sust_s/layer_compute_s:.0f})")
print(f"prefill/layer: compute {prefill_compute_s*1e6:.1f} us vs weight read {prefill_weight_read_s*1e6:.1f} us")
print("WROTE docs/p1/roofline.md")
