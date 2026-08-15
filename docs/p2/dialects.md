# P2.2/P2.3 方言定义（qnn + qisa）

三级 lowering 路线（PLAN.md P2）：`qnn（Transformer 语义）→ qisa（加速器 IR）→ Q-ISA asm → qbin`。
TableGen 源：`compiler/mlir/qnn.td`、`compiler/mlir/qisa.td`。

## qnn 方言（层语义层）

保留 Transformer 语义，每个 op 对应一条 lowering pass → qisa 序列。语义对齐 02-isa §12
（Qwen3 decode 最小步序列）。

| op | 语义 | 降为 qisa（02 §12 步） |
|----|------|------------------------|
| `qnn.matmul` | `y = x @ W^T`（线性层，`W [N,K]` PyTorch [out,in]） | GEMM（PF）/ GEMV（DC）`transpose_B=1`，N≤128 tiling、K 流式（§12 第 1 步） |
| `qnn.rmsnorm` | `y = (x/rms(x))⊙γ`，mode=normal/per-head | RMSNORM（§12 第 0/2 步） |
| `qnn.rope` | 旋转位置编码，`θ_i=pos·θ^(-2i/128)` | ROPE（§12 第 3 步） |
| `qnn.attention` | causal GQA 注意力（QK^T→mask→softmax→AV） | BMM+VMASK+VECTOR softmax+KV.GATHER（§12 第 4–5 步） |
| `qnn.swiglu` | `h = silu(gate)⊙up` | VSILU + VMUL（§12 第 7 步） |

## qisa 方言（加速器 IR 层）

与 02-isa 33 条指令**一一对应**，op 属性即 ISA 操作数字段（AR 引用 / C 寄存器索引 / 立即数）。

| 组 | 计数 | op |
|----|------|-----|
| SYS | 5 | `mode` `config` `barrier` `wait` `nop` |
| DMA | 3 | `dma.load` `dma.store` `dma.prefetch` |
| MATRIX | 3 | `gemm` `gemv` `bmm` |
| VECTOR | 18 | `vadd vsub vmul vdiv vrecip vexp vrsqrt vsilu vmax vmov vscale vmask vreduce_sum vreduce_max rope rmsnorm quant dequant` |
| KV | 4 | `kv.append` `kv.store_block` `kv.load` `kv.gather` |

关键字段映射（与 `compiler/isa/isa.py` 的 OPSPEC 一致，来源 02-isa §4–§8）：

- **MATRIX**（gemm/gemv/bmm 共用）：`ara/arb/arc, m, n, k, batch, ca/cb/cc/cd, acc_init, bsrc,
  dequant, transpose_a, transpose_b, src_a, src_b, acc`（§6.1）。
- **CD dequant 描述符**（C 寄存器 32b）：`[31:21] reserved | [20] mode | [19] scale_dtype | [18:0] scale_base`
  —— per-128-group scale 数组（BF16/FP16）在 SRAM 的 16B 字地址（§6.1）。
- **VECTOR** 共用：`ara/arb/ard, len, cv, imm, src_a, src_b, acc`（§7.1）；`cv`/`imm` 按 opcode 解释（§7.2）。
- **KV**：层号/head/sel/broadcast/pos_start/count/cstride 逐 op（§8）。

## 与 ISA 编码器的边界

qisa → Q-ISA asm 的 lowering 在 `compiler/lowering.py`（本 M2a 直接以 Python 产 asm 文本 →
`isa.py` assembler → 128-bit 字节 → qbin）。方言 op 的每个属性值直接写入对应 128-bit 字段，
无中间格式（00-container §2「P3/P4 共用同一解析器，不做中间格式」）。
