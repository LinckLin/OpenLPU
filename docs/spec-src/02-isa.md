# 02 — Tensor Command ISA v0

> 本切片为指令语义与编码的权威定义（契约第 9 条）。所有指令编码逐位可查、无字段重叠；
> 内存行为归 03-memory-system，执行单元行为归 04-execution-engines，KV 协议归 05-kv-cache。
> 本文件中的语义描述是各切片之间的 binding 接口；冲突时以 P0 冻结契约为准并报主会话。

---

## 1. 定位与范围

newlpu 的 ISA 是 **tensor command ISA**，不是 scalar CPU ISA。指令流的执行模型（直线序列、无跳转、
控制仅经 `BARRIER`/`WAIT`、程序入口由 `CONFIG` 批量写寄存器）见 00-container。本文件定义：

1. 128-bit 定长指令的完整字段布局（契约第 1 条展开）；
2. 《强制指令全集》33 条指令逐条的 opcode、引擎、操作数、dtype 约束、语义、PF/DC 双模行为；
3. 全局 opcode 编码分配表；
4. 明确禁止项与非法指令行为；
5. 文末「ISA 完备性证明：Qwen3 decode 最小步序列」。

**冻结约束（不可更改）**：目标模型 Qwen3-8B dense（decoder-only，batch=1，4K/8K ctx，BF16/INT8/INT4）；
路线 HF→MLIR→加速器 IR→Tensor Command ISA→ASIC；时钟 1 GHz。

---

## 2. 指令格式（128-bit 模板）

### 2.1 总布局

```
 bit:  127 ─────── 120 | 119 ────── 112 | 111 109 | 108 106 | 105 104 | 103 ─────────────── 0
 field:  engine tag     |  opcode        |  srcA   |  srcB   |  acc    |  engine-specific
 width:     8b          |     8b         |   3b    |   3b    |   2b    |      104b
```

- 总宽度校验：$8 + 8 + 3 + 3 + 2 + 104 = 128$ bit。✓
- 指令定长 128-bit，小端序（与 00-container 的 `.qbin` 数值序一致）。
- 头 24 bit（`[127:104]`）全引擎通用；`[103:0]` 为引擎自定义操作数（各引擎一张布局表，见 §4–§8）。

### 2.2 通用头字段

**`[127:120] engine tag`（8b，路由标签）**

| 值 | 引擎 |
|----|------|
| `0x00` | SYS |
| `0x01` | DMA |
| `0x02` | MATRIX |
| `0x03` | VECTOR |
| `0x04` | KV |
| `0x05–0xFF` | reserved |

`engine tag` 与 opcode 的全局区间（§3）是**冗余一致**的双重编码：Command Processor 按 opcode 区间
判定引擎并分发，`engine tag` 必须与该区间一致；不一致 → 非法指令（§10）。

**`[119:112] opcode`（8b，全局编码）**

全局 0x00–0xFF，见 §3 分配表。

**`[111:104] dtype flags`（8b）**

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[111:109]` | srcA | 3b | 操作数 A 的 dtype（3-bit 编码） |
| `[108:106]` | srcB | 3b | 操作数 B 的 dtype（一元/单操作数指令忽略） |
| `[105:104]` | acc | 2b | 累加/计算数据通路 dtype（2-bit 编码） |

3-bit dtype 编码（契约第 3 条，固定）：

| 编码 | dtype |
|------|-------|
| `0` | BF16 |
| `1` | FP16 |
| `2` | INT8 |
| `3` | INT4 |
| `4` | INT32 |
| `5` | INT16 |
| `6` | FP8E4M3 |
| `7` | reserved |

2-bit acc 编码：

| 编码 | 含义 |
|------|------|
| `00` | INT32 |
| `01` | FP32 |
| `10` | FP16 |
| `11` | reserved |

> 注：3-bit srcA/srcB 表不含 FP32。v0 中 FP32 是**计算数据通路/累加宽度**（由 `acc` 选择），
> 不是片上存储格式；softmax 等以 `acc=FP32` 走 fp32 内部数据通路、BF16 落盘。
> 各引擎对 `acc` 的细化见 §4–§8。SYS/DMA/KV 的 dtype 字段约束见各自章节。

### 2.3 寄存器文件（契约第 2 条，binding）

**地址寄存器 AR0–AR63（64×64b）**：指令通过 6-bit 索引引用 AR。

| 位 | 含义 |
|----|------|
| `[63]` | 地址类型：`1` = HBM；`0` = SRAM |
| `[62:40]` | 置零 |
| `[39:0]` | `bit63=1` 时：40b HBM 字节地址（可寻址 $2^{40}$ B = 1 TiB；设备实现 16 GiB = $2^{34}$ B，用低 34 位） |
| `[18:0]` | `bit63=0` 时：19b SRAM 字地址（16B 字，高位置零），可寻址 $2^{19}\times16$ B = 8 MiB ✓ |

- SRAM 字地址 `[18:0]` 与 bank 划分：bank = `addr[7:4]`（16B 粒度 16 路交错），bank 内字 = `addr[22:8]`（32K 字 = 512 KiB）✓（契约第 4 条）。
- **约定（v0）**：`AR63` = `AR_KV_BASE`（KV cache HBM 基址，bit63=1），供全部 KV 指令隐式引用（05-kv-cache 地址公式的基址）。

**配置寄存器 C0–C31（32×32b）**：指令通过 5-bit 索引引用 C。C 寄存器承载维度/stride/KV 位置/scale
（契约第 2 条），其逐字段布局在本文件作为操作数编码定义（各引擎 §4–§8）。**约定（v0）**：
`C30` = `C_KV_POS`（KV.APPEND 当前 token 位置，13b 有效）；`C31` = `C_SLAB_SHIFT`
（KV slab 步长参数 ∈{20,21,22}，load-time，默认 21，见 §8）。

> AR/C 均由 `CONFIG`（§4）写入；`CONFIG` 是 v0 唯一的寄存器写入口（Host 侧另有加载器直写，
> 但指令流内仅 `CONFIG`）。

---

## 3. 编码分配总表

| opcode 区间 | 引擎 | 指令 | opcode | engine tag |
|-------------|------|------|--------|-----------|
| `0x00–0x1F` | SYS | MODE | `0x00` | `0x00` |
| | | CONFIG | `0x01` | |
| | | BARRIER | `0x02` | |
| | | WAIT | `0x03` | |
| | | NOP | `0x04` | |
| | | reserved | `0x05–0x1F` | |
| `0x20–0x3F` | DMA | DMA.LOAD | `0x20` | `0x01` |
| | | DMA.STORE | `0x21` | |
| | | DMA.PREFETCH | `0x22` | |
| | | reserved | `0x23–0x3F` | |
| `0x40–0x7F` | MATRIX | GEMM | `0x40` | `0x02` |
| | | GEMV | `0x41` | |
| | | BMM | `0x42` | |
| | | reserved | `0x43–0x7F` | |
| `0x80–0xBF` | VECTOR | VADD / VSUB / VMUL / VDIV / VRECIP / VEXP / VRSQRT / VSILU | `0x80–0x87` | `0x03` |
| | | VMAX / VMOV / VSCALE / VMASK | `0x88–0x8B` | |
| | | VREDUCE_SUM / VREDUCE_MAX / ROPE / RMSNORM | `0x8C–0x8F` | |
| | | QUANT / DEQUANT | `0x90–0x91` | |
| | | reserved | `0x92–0xBF` | |
| `0xC0–0xDF` | KV | KV.APPEND | `0xC0` | `0x04` |
| | | KV.STORE_BLOCK | `0xC1` | |
| | | KV.LOAD | `0xC2` | |
| | | KV.GATHER | `0xC3` | |
| | | reserved | `0xC4–0xDF` | |
| `0xE0–0xFF` | — | reserved | `0xE0–0xFF` | reserved |

VECTOR 逐条 opcode：`VADD=0x80, VSUB=0x81, VMUL=0x82, VDIV=0x83, VRECIP=0x84, VEXP=0x85,
VRSQRT=0x86, VSILU=0x87, VMAX=0x88, VMOV=0x89, VSCALE=0x8A, VMASK=0x8B, VREDUCE_SUM=0x8C,
VREDUCE_MAX=0x8D, ROPE=0x8E, RMSNORM=0x8F, QUANT=0x90, DEQUANT=0x91`。

全集计数：SYS 5 + DMA 3 + MATRIX 3 + VECTOR 18 + KV 4 = **33 条**，与《强制指令全集》一一对应，无增删（仅加 NOP 属允许项）。✓

---

## 4. SYS 引擎

- dtype 字段 `[111:104]`：**必须为 0**（SYS 无数据操作数）。
- 操作数字段 `[103:0]`（104b），逐条布局如下。

### 4.1 MODE（opcode 0x00）— 双模切换

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:102]` | mode | 2b | `00`=PF（prefill），`01`=DC（decode），`10/11`=reserved |
| `[101:0]` | reserved | 102b | 必须为 0 |

语义：全局双模开关。`PF`：MATRIX 阵列以整块 128×128 GEMM 工作；`DC`：阵列拆 16 lane×8 行，每 lane 独立 GEMV
（契约第 7 条）。切换前必须先 `BARRIER`（所有引擎排空）；切换代价见 04-execution-engines。

### 4.2 CONFIG（opcode 0x01）— 写 AR/C 寄存器

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | REG | 6b | 目标寄存器索引：AR0–AR63（class=1）或 C0–C31（class=0，用低 5b） |
| `[97]` | class | 1b | `0`=C（32b），`1`=AR（64b） |
| `[96:33]` | IMM64 | 64b | 写入值（小端序） |
| `[32:0]` | reserved | 33b | 必须为 0 |

语义：`class=1` 时把 `IMM64` 写入 `AR[REG]`（含 bit63 类型标志 + 40b HBM 地址或 19b SRAM 字地址）；
`class=0` 时把 `IMM64[31:0]`（低 32b）写入 `C[REG]`，`IMM64[63:32]` 必须为 0。
这是 v0 唯一的指令流内寄存器写入口；程序入口由一串 `CONFIG` 写基址/stride/维度/KV 位置/scale。

### 4.3 BARRIER（opcode 0x02）

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:0]` | reserved | 104b | 必须为 0 |

语义：全引擎同步栅栏——所有先发射指令（MATRIX/VECTOR/DMA/KV）全部排空并 retire 后，后续指令才开始发射。
用于 PF/DC 切换前、双缓冲边界、以及任何跨引擎顺序依赖。

### 4.4 WAIT（opcode 0x03）— 按引擎等待

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:100]` | eng_mask | 4b | 位向量：bit0=DMA，bit1=MATRIX，bit2=VECTOR，bit3=KV |
| `[99:0]` | reserved | 100b | 必须为 0 |

语义：阻塞后续发射，直到 `eng_mask` 指示的引擎队列全部排空（空闲）。`eng_mask=0` 等价于 NOP。
`WAIT` 只等引擎，不等内存一致性（SRAM 写对 MATRIX/VECTOR 可见由引擎顺序保证）。

### 4.5 NOP（opcode 0x04）

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:0]` | reserved | 104b | 必须为 0 |

语义：无操作，占位/对齐用。NOP 是《强制指令全集》允许唯一新增的指令（本切片即按全集定义，无新增）。

---

## 5. DMA 引擎

- dtype 字段：`srcA` = 传输数据 dtype（用于 16B 粒度与行内字节解释；不改变数据）；`srcB`/`acc` 必须为 0。
- 三条指令（LOAD/STORE/PREFETCH）共享同一 2D 操作数字段布局（契约：源 AR + 目标 AR + 行数/行内字节；stride 放 C 寄存器）。

### 5.1 操作数字段布局（LOAD/STORE/PREFETCH 共用）

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | SrcAR | 6b | 源基址 AR |
| `[97:92]` | DstAR | 6b | 目标基址 AR |
| `[91:76]` | RowBytes | 16b | 每行字节数（1D 即总字节数） |
| `[75:60]` | NumRows | 16b | 行数（1D 必须为 1） |
| `[59:55]` | StrideC | 5b | C 寄存器：源侧行 stride（字节） |
| `[54]` | mode | 1b | `0`=1D 连续；`1`=2D 带 stride |
| `[53:0]` | reserved | 54b | 必须为 0 |

校验：$6+6+16+16+5+1+54=104$ bit。✓

语义：

- **2D 地址**：源行 $r$ 的基址 = `SrcAR + r × Stride`（`Stride` 取自 `C[StrideC]`，字节）；
  每行拷贝 `RowBytes` 字节；共 `NumRows` 行。目标在 SRAM 内**密集**存放（连续无 stride）。
- **1D 地址**：`NumRows=1`，连续拷贝 `RowBytes` 字节，忽略 `Stride`。
- **方向**：
  - `DMA.LOAD`：`SrcAR`=HBM（bit63=1），`DstAR`=SRAM（bit63=0）。
  - `DMA.STORE`：`SrcAR`=SRAM，`DstAR`=HBM。
  - `DMA.PREFETCH`：`SrcAR`=HBM，`DstAR`=SRAM，与 LOAD 同向但**非阻塞/建议性**——提前把数据搬进 SRAM
    （权重/KV 预取），不建立数据依赖，供后续 MATRIX/VECTOR 消费。
- **对齐**：`RowBytes` 必须为 16 的倍数（SRAM 访问粒度 16B）；HBM 源/目标地址 64B 突发对齐由 DMA 引擎
  以 masking/partial-burst 处理（详见 03-memory-system）。跨 bank 由 16 bank 交错自然覆盖。

### 5.2 逐条

| 指令 | opcode | 引擎 | 操作数 | dtype 约束 | 语义 |
|------|--------|------|--------|-----------|------|
| DMA.LOAD | `0x20` | DMA | SrcAR, DstAR, RowBytes, NumRows, StrideC, mode | srcA=数据 dtype | HBM→SRAM 拷贝（1D/2D） |
| DMA.STORE | `0x21` | DMA | 同上 | srcA=数据 dtype | SRAM→HBM 拷贝（1D/2D） |
| DMA.PREFETCH | `0x22` | DMA | 同上 | srcA=数据 dtype | HBM→SRAM 非阻塞预取（1D/2D） |

PF/DC：DMA 不依赖阵列模式，两种模式下行为一致（DC 下 KV/权重预取与计算 overlap 是调度优化，见 P6）。

---

## 6. MATRIX 引擎

- dtype 字段：`srcA` = A dtype，`srcB` = B dtype，`acc` = 累加/输出 dtype（`00`=INT32，`01`=FP32，`10`=FP16）。
- 阵列：128×128 MAC，weight-stationary，INT8 MAC + INT32 累加；INT4 打包双倍吞吐；BF16 用 2×INT8 分解仿真
  （契约第 6 条）。1 GHz → INT8 峰值 32.77 TMAC/s（推导见 §11）。
- 通用语义（GEMM/GEMV/BMM 共享）：
  $C[m,n] = \sum_{k=0}^{K-1} A[m,k]\cdot B[k,n]$，维度以**元素**计，行主序；`M,N \le 128`，`K` 任意（流式，每拍进 128 个 K）。
  A=激活（SRAM），B=权重（SRAM 或 HBM 流式），C=累加。
- dtype 组合（覆盖 BF16 / W8A8 / W4A8 / W4A16）：
  - BF16：`srcA=srcB=BF16, acc=FP32`（2×INT8 分解仿真，8.19 TMAC/s，见 04 §1.2）；
  - W8A8：`srcA=srcB=INT8, acc=INT32`，`dequant=1`（per-128-group scale → fp）；
  - W4A8：`srcA=INT8, srcB=INT4, acc=INT32`，`dequant=1`（INT4 打包双倍吞吐 65.54 TMAC/s）；
  - W4A16：`srcA=BF16/FP16, srcB=INT4, acc=FP32`，`dequant=1`（BF16 尾数路径，吞吐与 BF16 同量级 8.19 TMAC/s，见 04 §1.2）。

### 6.1 操作数字段布局（GEMM/GEMV/BMM 共用）

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | ARa | 6b | A 基址（激活，SRAM） |
| `[97:92]` | ARb | 6b | B 基址（权重，SRAM/HBM） |
| `[91:86]` | ARc | 6b | C 基址（累加/输出） |
| `[85:78]` | M | 8b | 输出行数，1..128（GEMV 必须=1） |
| `[77:70]` | N | 8b | 输出列数，1..128 |
| `[69:54]` | K | 16b | 归约维度长度，1..65535（流式） |
| `[53:48]` | batch | 6b | 批次数，1..16（GEMM/GEMV 必须=1） |
| `[47:43]` | CA | 5b | C 寄存器：A 行/批 stride 描述符 |
| `[42:38]` | CB | 5b | C 寄存器：B 行/批 stride 描述符 |
| `[37:33]` | CC | 5b | C 寄存器：C 行/批 stride 描述符 |
| `[32:28]` | CD | 5b | C 寄存器：dequant scale 描述符 |
| `[27]` | acc_init | 1b | `0`=累加进现有 C；`1`=先清 0 再累加 |
| `[26]` | bsrc | 1b | B 来源：`0`=SRAM，`1`=HBM 流式 |
| `[25]` | dequant | 1b | `0`=不反量化；`1`=按 CD 反量化 scale |
| `[24]` | transpose_A | 1b | `0`=A 为 M×K；`1`=A 存为 K×M（读转置） |
| `[23]` | transpose_B | 1b | `0`=B 为 K×N；`1`=B 存为 N×K（读转置） |
| `[22:0]` | reserved | 23b | 必须为 0（**BMM 时 `[20:5]` 复用为 `pos_base[15:0]`**，仅 CD.KV_QUANT=1 时有效；`[22:21]`、`[4:0]` 仍必须为 0） |

校验：$6\times3 + 8 + 8 + 16 + 6 + 5\times4 + 5\times1 + 23 = 18+8+8+16+6+20+5+23 = 104$ bit。✓

C 寄存器描述符（32b）：

**CA / CB / CC（stride 描述符）**：`[31:16]`=row_stride（字节），`[15:0]`=batch_stride（字节）。

**CD（dequant 描述符）**：

| 位 | 字段 | 含义 |
|----|------|------|
| `[31]` | KV_QUANT | `1`=B 操作数为量化 KV（B-feed 在线去量化）；`0`=常规权重/去量化路径 |
| `[30]` | ROTATE_K | `1`=对 K 施加绝对位置 RoPE（B-feed）；`0`=不旋转（V 或常规路径） |
| `[29:21]` | KV_IDX | `(layer*8+head)`，选择 k_norm/scale slab（仅 KV_QUANT=1 时有效） |
| `[20]` | mode | `0`=per-tensor，`1`=per-128-group（KV_QUANT=1 时忽略） |
| `[19]` | scale_dtype | `0`=BF16，`1`=FP16 |
| `[18:0]` | scale_base | 19b SRAM 字地址（scale 数组基址） |

> per-128-group 的 group 大小固定为 128（契约 per-128-group）。
> B-feed 融合（rotator-impl）：BMM 指令 reserved `[20:5]` 复用为 `pos_base[15:0]`
> （绝对位置 0..40959）；KV_QUANT=1 时 B 操作数直接从 HBM 量化 KV slab 读取
> （K：INT8 折叠 + ROTATE_K；V：INT4 per-token），去量化 + 旋转在 B-feed 内联完成，
> 不再经 KV.LOAD staged BF16 写入。无新指令（33 条计数不变）。

### 6.2 逐条

**GEMM（opcode 0x40）** — 整块矩阵乘
- 语义：`C[M×N] += A[M×K] × B[K×N]`；`M,N≤128`，`K` 流式；`batch=1`。
- `acc_init=1` 清 C；`bsrc=1` 时 B 从 HBM 流式（大权重 tile）；`dequant=1` 时按 per-128-group scale 反量化
  （$C = \sum_{g}\text{scale}[g]\cdot\sum_{k\in\text{group}_g}A\cdot B$），覆盖 W8A8 / W4A16。
  bias 不在 INT32 累加内——dequant 后在 fp 域用 `VADD` 加（见 04-execution-engines 后处理路径）。
- PF：整块 128×128 阵列一次算一个 M×N tile（K 逐 128 流式）。
- DC：合法（decode 主用 GEMV/BMM）；K 经 lane 内 8 行流式的分片细节见 04-execution-engines。

**GEMV（opcode 0x41）** — M=1 特例
- 语义：`C[1×N] += A[1×K] × B[K×N]`；`M=1` 强制（否则非法），`N≤128`，`K` 流式；`batch=1`。
- DC：阵列 128 行拆 16 lane×8 行，每个 lane = 一个独立 GEMV（该 lane 的 128 列出 N、8 行驻留 K，K 经 8 行/拍流式）；
  单条 GEMV 占 1 个 lane；`BMM batch≤16` 时 16 lane 各跑一个独立 GEMV（每 lane 一个 batch 元素）。
  `transpose_B` 支持 QK^T（B 存 `[N×K]`，即 KV cache 的 `[seq×128]` 自然布局）。
- PF：整块阵列，行为同 GEMM 的 M=1 情形。

**BMM（opcode 0x42）** — batched 矩阵乘（attention 多 head）
- 语义：`C[b][M×N] += A[b][M×K] × B[b][K×N]`，`b=1..16`；`A[b]` 在 `ARa + b×batch_stride_A`（CA 的 `[15:0]`），
  `B[b]` 在 `ARb + b×batch_stride_B`，`C[b]` 在 `ARc + b×batch_stride_C`。
- DC：`batch` 映射到 16 lane（每 lane 一个 batch 元素）；GQA 注意力（QK^T/AV）用 `batch=4`（1 KV head 服务 4 Q head）
  或 `batch=16`（4 个 KV head 同拍）。
- PF：`batch` 在阵列上时分复用（逐批计算）。

> DC 下 GQA 组内 K/V 共享经 P7 内部广播总线；B 侧地址派生规则由 04 §2.2 在 P7 落地时定义。

---

## 7. VECTOR 引擎

- dtype 字段：`srcA`=A dtype，`srcB`=B dtype（二元 op；一元 op 忽略），`acc`=计算数据通路
  （`00`=INT32，`01`=FP32，`10`=FP16）。输出落盘 dtype = `srcA`（就地格式）；`acc=FP32` 表示 fp32 内部数据通路。
- 语义单位：`len` 以**元素**计；向量在 SRAM 行主序、按 `srcA` dtype 连续存放。

### 7.1 操作数字段布局（全部 VECTOR 指令共用）

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | ARa | 6b | 操作数 A 基址（一元 src / 二元左操作数） |
| `[97:92]` | ARb | 6b | 操作数 B 基址（二元 op / RMSNORM 的 γ 权重） |
| `[91:86]` | ARd | 6b | 目标基址 |
| `[85:70]` | len | 16b | 元素个数 |
| `[69:65]` | CV | 5b | C 寄存器：op 特定描述符（见 7.2 表） |
| `[64:33]` | imm | 32b | op 特定立即数（见 7.2 表） |
| `[32:0]` | reserved | 33b | 必须为 0 |

校验：$6\times3 + 16 + 5 + 32 + 33 = 18+16+5+32+33 = 104$ bit。✓

### 7.2 逐条语义（CV/imm 按 opcode 解释）

| 指令 | opcode | 类别 | 语义（逐元素，$i\in[0,len)$） | CV 含义 | imm 含义 | dtype 约束 |
|------|--------|------|------------------------------|---------|----------|-----------|
| VADD | `0x80` | 二元 | `d[i]=a[i]+b[i]` | cv=0 → ARb 连续 len 元素；cv≠0 → ARb[0] 标量广播（v0 约定细化） | 未用 | srcA/srcB 同 dtype；acc=数据通路 |
| VSUB | `0x81` | 二元 | `d[i]=a[i]-b[i]` | 同上 | 未用 | 同上 |
| VMUL | `0x82` | 二元 | `d[i]=a[i]×b[i]` | 同上 | 未用 | 同上 |
| VDIV | `0x83` | 二元 | `d[i]=a[i]/b[i]` | 同上 | 未用 | fp：srcA/srcB∈{BF16,FP16}，acc=FP32 |
| VRECIP | `0x84` | 一元 | `d[i]=1/a[i]` | 未用 | 未用 | fp |
| VEXP | `0x85` | 一元 | `d[i]=exp(a[i])` | 未用 | 未用 | fp；softmax 用 acc=FP32 |
| VRSQRT | `0x86` | 一元 | `d[i]=1/sqrt(a[i])` | 未用 | 未用 | fp |
| VSILU | `0x87` | 一元 | `d[i]=a[i]·sigmoid(a[i])` | 未用 | 未用 | fp |
| VMAX | `0x88` | 二元 | `d[i]=max(a[i],b[i])` | 同上 | 未用 | 同 dtype |
| VMOV | `0x89` | 一元 | `d[i]=a[i]`（拷贝，可做 dtype 转换） | 未用 | 未用 | srcA→acc 输出 |
| VSCALE | `0x8A` | 一元+标量 | `d[i]=a[i]×s` | 未用 | `s`=标量（`imm[15:0]` BF16；acc=FP32 时用 CV 存 fp32 标量） | fp |
| VMASK | `0x8B` | 生成 | 因果掩码 tile：`d[i,j]=0 if (col_base+j)≤(row_base+i) else -inf`，`rows×cols` 行主序 | `[31:16]`=col_base，`[15:0]`=row_base（全局 tile 偏移） | `imm[31:16]`=rows，`imm[15:0]`=cols | acc=数据通路（-inf 按 acc dtype 编码） |
| VREDUCE_SUM | `0x8C` | 归约 | 每组 `len` 元素求和 → 1 标量（按 CV 的 group stride 对多组各输出 1 标量） | group_stride（组数×组内长度；默认 1 组） | 未用 | acc=FP32（softmax 求和） |
| VREDUCE_MAX | `0x8D` | 归约 | 每组 `len` 元素求 max → 1 标量 | 同上 | 未用 | acc=FP32 |
| ROPE | `0x8E` | 逐元素 | 对每个 head_dim=128 的 head 施加旋转（偶奇对旋转角 $\theta_i=\text{pos}\cdot\theta^{-2i/128}$） | θ（fp32，`rope_theta`=1e6） | `imm[15:0]`=pos（0..40959） | srcA∈{BF16,FP16}，acc=FP32 |
| RMSNORM | `0x8F` | 归一化 | `d=(a/rms(a))⊙γ`，`rms=sqrt(mean(a²)+eps)`；per-head 模式按 128 元素分组各算 rms | eps（fp32，`rms_eps`=1e-6） | `mode`=imm[31]（0=normal，1=per-head，见下） | srcA∈{BF16,FP16}，acc=FP32 |
| QUANT | `0x90` | 量化 | `d=round(a/s)` → INT8/INT4（`srcA` fp → `acc`/输出 int） | scale 基址（SRAM）+ mode（per-tensor / per-128-group） | 未用 | srcA∈{BF16,FP16}，输出 INT8/INT4 |
| DEQUANT | `0x91` | 反量化 | `d=a×s`（INT8/INT4 → fp） | scale 基址 + mode | 未用 | srcA∈{INT8,INT4}，输出 BF16/FP16 |

> 说明：
> - **RMSNORM mode 位**（契约「normal 与 per-head 两种模式位」）：`imm[31]=0` 对全部 `len` 元素算一个 rms
>   （输入 RMSNorm，len=4096）；`imm[31]=1` 按 head_dim=128 分组各算 rms（QK-norm，len=4096→32 组 / len=1024→8 组）。
>   γ（`ARb`）始终逐元素相乘；无权重时 `ARb` 指向 ones 向量。
> - **VMASK** 生成 causal 掩码 tile：全局条件 `col_base+j ≤ row_base+i` 取 0、否则 `-inf`；softmax 前 `VADD` 到 scores 上。
>   长序列按 score tile 分块（`rows/cols` 分块 + 全局 base 偏移），8K 因果掩码不整块物化（$8192^2\times2$ B = 134 MB > SRAM）。
> - **ROPE** v0 为**单 position**（decode 常见路径）；prefill 块 RoPE 由编译器按 position 逐条发（或 strided 变体入 backlog）。
> - **QUANT/DEQUANT** 覆盖 W8A8（INT8）与 W4A16（INT4，2b 打包）；per-128-group scale 与 GEMM 的 dequant 描述符同构。

---

## 8. KV 引擎

- dtype 字段：**固定 BF16**（`srcA=0`，`srcB`/`acc` 必须为 0）——KV cache 元素 = pos×128×2B（bf16，契约第 8 条）。
- 地址公式（05-kv-cache 权威，本切片引用）：
  $\text{addr} = \text{AR\_KV\_BASE} + ((\text{layer}\ll25)\ |\ (\text{head}\ll22)\ |\ (\text{kv}\ll21)) + (\text{pos}\ll8) + (d\ll1)$（SLAB_SHIFT=21 情形）
  - slab = $2^{\text{SLAB\_SHIFT}}$ B 每 (layer, head, K/V)；SLAB_SHIFT ∈ {20,21,22} 为 load-time 参数（C31，默认 21 → 2 MiB、容量 $2^{13}$ token（8K）×128×2B ✓；20 → 1 MiB、4K token，BF16 参考模式用，见 03-memory §7.3；22 → 4 MiB、容量 $2^{14}$ token（16K），`KV.LOAD/STORE_BLOCK` 的 13-bit `pos_start` 仍限 $[0,8192)$，pos=8192 尾 token 由 DC 的 VMOV 尾 subtile 覆盖）。上式为 SHIFT=21 位域；SHIFT=20 时 slab 步长 $2^{20}$、pos 12b；SHIFT=22 时 slab 步长 $2^{22}$、pos 13b（容量 16K，寻址仍 13b）。
  - 36 layer × 8 head × 2(K/V) × 2 MiB = 1152 MiB = 1.125 GiB < 16 GiB ✓。
- 全局：`AR63`=AR_KV_BASE，`C30`=C_KV_POS（见 §2.3）。

### 8.1 KV.APPEND（opcode 0xC0）— 追加单 token 的 K/V

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | srcK | 6b | 新 K 向量（1×128 bf16，SRAM） |
| `[97:92]` | srcV | 6b | 新 V 向量（1×128 bf16，SRAM） |
| `[91:86]` | layer | 6b | 层号 0..35 |
| `[85:83]` | head | 3b | KV head 0..7 |
| `[82:0]` | reserved | 83b | 必须为 0 |

语义：把 SRAM 里的新 K/V 写入 KV cache 的 `(layer, head)` slab 的 `pos=C_KV_POS` 位置（256B 对齐）。
decode 每 token 每层每 KV head 各发一条（36×8=288 条/token；K、V 同时写）。

### 8.2 KV.STORE_BLOCK（opcode 0xC1）— 写 K/V 块（prefill）

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | srcK | 6b | K 块（`count×128` bf16，SRAM） |
| `[97:92]` | srcV | 6b | V 块（`count×128` bf16，SRAM） |
| `[91:86]` | layer | 6b | 层号 |
| `[85:83]` | head | 3b | KV head |
| `[82:70]` | pos_start | 13b | 首 token 位置 |
| `[69:56]` | count | 14b | token 数（默认 128，≤8192） |
| `[55:0]` | reserved | 56b | 必须为 0 |

语义：prefill 阶段把整块 K/V（`count` 个 token）写入 slab `[pos_start, pos_start+count)`。

### 8.3 KV.LOAD（opcode 0xC2）— 读 K/V 到 SRAM

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | dstK | 6b | K 目标（SRAM） |
| `[97:92]` | dstV | 6b | V 目标（SRAM） |
| `[91:86]` | layer | 6b | 层号 |
| `[85:83]` | head | 3b | KV head |
| `[82:81]` | sel | 2b | `0`=K，`1`=V，`2`=both，`3`=reserved |
| `[80:68]` | pos_start | 13b | 首 token 位置 |
| `[67:54]` | count | 14b | token 数（语义上限 ≤2048，分块加载） |
| `[53:0]` | reserved | 54b | 必须为 0 |

语义：从 slab 读 `count` 个 token 的 K（和/或 V）到 SRAM；`sel=both` 时 K 进 `dstK`、V 进 `dstV`。

### 8.4 KV.GATHER（opcode 0xC3）— 收集 + GQA 广播

| 位 | 字段 | 宽度 | 含义 |
|----|------|------|------|
| `[103:98]` | dst | 6b | 广播数据目标（SRAM） |
| `[97:92]` | dst2 | 6b | reserved（both 扩展预留；v0 单 sel，K/V 各发一条） |
| `[91:86]` | layer | 6b | 层号 |
| `[85:83]` | head | 3b | KV head 0..7 |
| `[82]` | sel | 1b | `0`=K，`1`=V |
| `[81]` | broadcast | 1b | `0`=1:1 收集；`1`=GQA 广播 ×4 |
| `[80:68]` | pos_start | 13b | 首 token 位置 |
| `[67:54]` | count | 14b | token 数（字段 ≤8192；`broadcast=1` 时 v0 footprint ≤512，见 03-memory §9） |
| `[53:49]` | Cstride | 5b | C 寄存器：Q head 广播 stride（SRAM 字地址间距，与 05 §4.4 一致） |
| `[48:0]` | reserved | 49b | 必须为 0 |

语义：把 `(layer, head)` 的 K（或 V）在 `[pos_start, pos_start+count)` 区间的数据收集到 SRAM `dst`。
`broadcast=1` 时按 GQA（1 KV head 服务 4 Q head，Qwen3-8B 的 32/8=4）把同一份 K/V 复制 4 份，
份间距 = `C[Cstride]`（Q head 块 stride，SRAM 字地址间距），供后续 BMM（batch=4）直接读。
分页 KV（PagedAttention）入 backlog（契约第 8 条）。

---

## 9. 明确禁止项（ISA 边界）

以下**不进 ISA v0**（契约约束，保持 tensor command 形态）：

| 禁止项 | 理由 |
|--------|------|
| scalar 整数/浮点寄存器运算 | 无标量 ALU；所有数值经 tensor 指令在 SRAM 上进行 |
| 分支 / 跳转 / 条件执行 | 命令流是直线序列（00-container）；控制只经 `BARRIER`/`WAIT` |
| 标量 load/store（字节/字粒度访存） | 最小访存粒度 16B（SRAM 字）；数据搬移一律 DMA |
| 程序计数器相对/寄存器间接跳转 | 无 PC；指令定长顺序发射 |
| 新指令（除 NOP） | 全集固定，语义只能细化不能增删（契约） |

---

## 10. 非法指令与 reserved 行为

- opcode 落在 reserved 区间（§3）→ 非法指令，Command Processor **halt + 报错**（P3 模拟器同义）。
- `engine tag` 与 opcode 区间不一致 → 非法指令。
- 任何标 `必须为 0` 的字段非 0 → 实现可忽略或报错（v0 约定：报错，防前向兼容静默偏差）。
- **豁免**：MATRIX `CD[31:30]`（KV_QUANT/ROTATE_K）、`CD[29:21]`（KV_IDX）与 BMM 指令
  `[20:5]`（pos_base）在 rotator-impl（B-feed 融合）中改为语义字段，不再要求为 0；仅当
  `CD[31]`（KV_QUANT）=1 时该语义生效，`CD[31]=0` 时保持 v0 逐字节向后兼容。
- `M>128`、`N>128`、GEMV 且 `M≠1`、`batch>16`、`dtype=7`（reserved）→ 非法。
- AR 引用越界（`[63]=1` 时 `[39:0]` 超出 16 GiB；`[63]=0` 时 `[18:0]` 的 bank/字越界）→ 由 03-memory-system 定义 fault。

---

## 11. 关键数值推导

| 量 | 推导 | 值 |
|----|------|-----|
| INT8 峰值 | $128\times128\ \text{MAC}\times2\ \text{op}\times1\ \text{GHz} = 2^{14}\times2\times10^9$ | $3.2768\times10^{13}$ = **32.77 TMAC/s** ✓ |
| INT4 峰值 | 2b 打包 → 每 cell 2×INT4 MAC/拍 | 65.54 TMAC/s |
| BF16 峰值 | 2×INT8 分解 = 每尾数 4 部分积 / 2 乘法器 → INT8/4 | 8.19 TMAC/s |
| SRAM 容量 | $16\ \text{bank}\times512\ \text{KiB}=16\times2^{19}$ B | 8 MiB ✓ |
| SRAM 字地址 | 访问粒度 16B → $2^{19}$ 字，19b `[18:0]` | 8 MiB ✓ |
| bank 划分 | `addr[7:4]`=4b→16 bank（16B 粒度 16 路交错）；`addr[22:8]`=15b→$2^{15}\times16$ B | 512 KiB/bank ✓ |
| HBM | 40b 字节地址，16 GiB 实现（$2^{34}$ B，用低 34 位） | 1.2 TB/s 聚合带宽 |
| KV slab | $8\text{K}\times128\times2\ \text{B}=2^{13}\times2^8=2^{21}$ | 2 MiB ✓ |
| KV 总量 | $36\times8\times2\times2^{21}$ B | 1152 MiB ✓ |
| Qwen3-8B | hidden 4096, layers 36, q_heads 32, kv_heads 8, head_dim 128, intermediate 12288, vocab 151936 | 契约第 10 条（待 ModelGrounding 核验） |

---

## 12. ISA 完备性证明：Qwen3 decode 最小步序列

> 目标：证明 Qwen3-8B 每 token 的 decode 前向路径每一步都有指令覆盖（batch=1，GQA 4:1，
> head_dim=128，hidden=4096，intermediate=12288，vocab=151936）。数据流记号：`x` 为 1×4096 hidden。

### 第 0 步：输入 RMSNorm（层入口，normal 模式）

```
RMSNORM  ARa=x  ARb=γ_in  ARd=xn  len=4096  CV={eps=1e-6}  mode=normal(0)
```

### 第 1 步：QKV 投影（3 路 GEMV，M=1，K=4096）

```
q = GEMV  ARa=xn  ARb=Wq  ARc=q   M=1 N=4096 K=4096 transpose_B=1   # 32 个 N=128 tile
k = GEMV  ARa=xn  ARb=Wk  ARc=k   M=1 N=1024 K=4096 transpose_B=1   # 8 个 N=128 tile
v = GEMV  ARa=xn  ARb=Wv  ARc=v   M=1 N=1024 K=4096 transpose_B=1
```
（线性层权重按 PyTorch `[out,in]` 存，`transpose_B=1` 读 `W^T`；N>128 由编译器 tiling。）

### 第 2 步：QK-norm（per-head RMSNorm，RoPE 前）

```
RMSNORM  ARa=q  ARb=γ_qk  ARd=q  len=4096  CV={eps}  mode=per-head(1)   # 32 组×128
RMSNORM  ARa=k  ARb=γ_qk  ARd=k  len=1024  CV={eps}  mode=per-head(1)   # 8 组×128
```

### 第 3 步：RoPE（单 position）

```
ROPE  ARa=q  ARd=q  len=4096  CV={θ=1e6}  imm[15:0]=pos
ROPE  ARa=k  ARd=k  len=1024  CV={θ=1e6}  imm[15:0]=pos
```

### 第 4 步：KV 追加

```
CONFIG  class=0 REG=C30(KV_POS)  IMM=pos           # 更新当前 token 位置（C 寄存器必须 class=0）
KV.APPEND  srcK=k  srcV=v  layer=L  head=0..7      # 每 KV head 一条
```

### 第 5 步：注意力（32 Q head，8 KV head，GQA 4:1）

记 `seq = pos+1`（已缓存 token 数）。因果掩码只依赖位置、与 head 无关，每层生成一次复用：

```
VMASK  ARd=mask  CV={row_base=0, col_base=0}  imm={rows=seq, cols=seq}   # (i,j): j≤i→0, j>i→-inf
```

对每个 KV head `g∈[0,8)`（服务 4 个 Q head）：

```
KV.LOAD  dstK=Kbuf  dstV=Vbuf  layer=L head=g sel=both pos_start=0 count=seq
# count>2048 时按 2048 分 tile（KV.LOAD 单副本，见 03-memory §9）
# K/V 由 P7 内部广播总线供 GQA 组内各 lane 共享（0.6B 2 个 / 8B 4 个）
# QK^T：scores[b=4][1×seq] = q_g[b][1×128] × K_g^T[128×seq]
BMM  ARa=q_buf ARb=Kbuf ARc=scores M=1 N=seq K=128 batch=4 transpose_B=1 acc_init=1
# causal mask + softmax（fp32 数据通路，每 head 一行）
VADD         ARa=scores ARb=mask   ARd=scores                        # scores+=mask
VREDUCE_MAX  ARa=scores ARd=maxv    len=seq  CV={group_stride}       # 每 head 行 max
VSUB         ARa=scores ARb=maxv    ARd=scores                       # scores-=max
VEXP         ARa=scores ARd=es      acc=FP32
VREDUCE_SUM  ARa=es     ARd=sumv    len=seq  CV={group_stride}
VRECIP       ARa=sumv   ARd=rinv    acc=FP32                         # rinv=1/sum
VMUL         ARa=es     ARb=rinv    ARd=softmax                      # softmax=es×rinv
# AV：ctx[b=4][1×128] = softmax[b][1×seq] × V_g[seq×128]
BMM  ARa=softmax ARb=Vbuf ARc=ctx M=1 N=128 K=seq batch=4 acc_init=1
```

说明：
- decode 中 KV cache 仅含 `0..pos` 的合法 key（无 future），因果掩码在纯 decode 下退化为全 0；
  上列 `VMASK` 为一般性覆盖（prefill 及固定窗口 + padding 场景必需），满足验收步「causal mask+softmax」。
- `seq>128` 时：QK^T 的 `N=seq` 与 softmax 的 `len=seq` 按 128 分块 tiling（BMM 流式 + VMASK 分块，`row_base/col_base` 给全局偏移）；
  AV 的 `K=seq` 由 BMM 流式。

### 第 6 步：O 投影 + 残差

```
o = GEMV  ARa=ctx_cat(32×128=4096)  ARb=Wo  ARc=o  M=1 N=4096 K=4096 transpose_B=1
VADD      ARa=x ARb=o ARd=x  len=4096                                      # 残差 x+=o
```

### 第 7 步：MLP（SwiGLU）+ 残差

```
RMSNORM  ARa=x ARb=γ_mlp ARd=xn2 len=4096  mode=normal
gate = GEMV  ARa=xn2 ARb=Wg  M=1 N=12288 K=4096 transpose_B=1
up   = GEMV  ARa=xn2 ARb=Wu  M=1 N=12288 K=4096 transpose_B=1
VSILU      ARa=gate ARd=h   len=12288
VMUL       ARa=h ARb=up  ARd=h   len=12288                                 # h = SiLU(gate)⊙up
down = GEMV ARa=h   ARb=Wd  M=1 N=4096  K=12288 transpose_B=1
VADD       ARa=x ARb=down ARd=x len=4096                                   # 残差
```

### 第 8 步：末层 LM head + 采样

```
RMSNORM  ARa=x ARb=γ_fin ARd=xn len=4096  mode=normal
logits = GEMV  ARa=xn ARb=Wlm  M=1 N=151936 K=4096 transpose_B=1          # 1187 个 N=128 tile
VREDUCE_MAX  ARa=logits ARd=lmax   len=151936                              # logits 峰值（可选数值校验）；argmax/采样由 host 完成（logits 经 DMA.STORE 回传，v0 决策）
```

### 覆盖结论

decode 路径用到的全部语义原语 —— RMSNorm（normal/per-head）、GEMM 族（GEMV/BMM，含 transpose_B）、
RoPE、KV.APPEND/GATHER、causal mask（VMASK）、softmax（VEXP/VREDUCE_*/VRECIP/VMUL）、SwiGLU（VSILU/VMUL）、
残差（VADD）、logits 生成—— 均被 §3 全集覆盖；采样（argmax/temperature）由 host 完成（logits 经 DMA.STORE 回传 HBM，ISA 不设 argmax 指令）。**无一步需要本 ISA 之外的指令。** ✓

---

## 13. 与其它切片接口

| 切片 | 接口点 |
|------|--------|
| 00-container | 命令流模型、`CONFIG` 写 AR/C、`BARRIER`/`WAIT` 语义、直线无跳转 |
| 03-memory-system | AR 地址语义、16B 粒度、bank 冲突、HBM 突发 64B、fault 定义 |
| 04-execution-engines | 128×128 阵列 PF/DC 行为、GEMV lane 映射、DMA/向量数据通路时序 |
| 05-kv-cache | KV 地址公式、slab 布局、GQA 广播、`AR_KV_BASE`/`C_KV_POS` 约定 |
