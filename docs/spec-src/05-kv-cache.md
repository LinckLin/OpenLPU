# 05 — KV Cache 寻址协议与 prefill/decode 流程

> 本切片负责 KV cache 的**协议层**：存储布局、地址公式、容量核算、KV 指令的操作数级语义、prefill/decode 时序与一致性。
> 指令的**位级编码**（128-bit 内 [103:0] 的字段打包、opcode/engine tag 取值、AR/C 寄存器索引）归 **IsaSpec**（02-isa）；SRAM/HBM 的**物理行为**（bank 冲突仲裁、burst 拆分）归 **MemorySystem**；MAC 阵列的**执行单元行为**归 **ExecEngines**。本文件给出的语义是四者之间的 binding 接口，冲突时以 P0 冻结契约为准并报主会话。

---

## 1. 存储布局 v0

### 1.1 决策：K 与 V 各自独立 slab（不交错）

每个 `(layer, kv_head)` 拥有**两块连续 slab**：K slab 与 V slab，K 在前、V 紧随其后。理由：

1. K 与 V 的消费时机不同——attention 中 K 先用于 `QKᵀ`，V 要到 softmax 之后才用于 `AV`；独立 slab 允许对 K、V 分别 tiling 与 prefetch，不互相污染 burst。
2. 独立 slab 使地址公式保持 `slab_base + pos×256B` 的纯线性形式，2 MiB 对齐天然成立。
3. v0 下 KV cache 元素为 BF16，无 per-group scale 元数据，交错不会带来任何节省，只增加地址解码复杂度。

### 1.2 全局 KV 区域与 slab 布局

以 Qwen3-8B 为准（`LAYERS=36, KV_HEADS=8, HEAD_DIM=128, KV 元素 = BF16`）：

- 单 token 单 head 的 K（或 V）= `HEAD_DIM × 2B = 256 B`。
- 每块 slab 容量 = `8K token × 256 B = 2 MiB`（`SLAB_SIZE = 2²¹ B`，默认 `SLAB_SHIFT=21`；BF16 参考模式用 4K：`SLAB_SHIFT=20`，见 03-memory §7.3）。
- 全局 slab 数 = `LAYERS × KV_HEADS × 2(K,V) = 36 × 8 × 2 = 576`。
- 全局 KV 区域总大小 = `576 × 2 MiB = 1,152 MiB = 1.125 GiB`。

slab 在区域内的排布顺序（线性索引）：

```
slab_index(layer, head, kv) = (layer × KV_HEADS + head) × 2 + kv
                            = (layer << 4) | (head << 1) | kv        // kv: 0=K, 1=V
slab_index ∈ [0, 576)
```

即：同一 `(layer, head)` 的 K、V 相邻（K 在前），同一 layer 内 8 个 head 连续，layer 递增。

### 1.3 地址公式（bit 精确，可直接实现）

以 HBM 40-bit 字节地址 `A[39:0]` 表示。区域基址存于寄存器 `AR_KV_BASE`（= `AR63`，详见 §2）。

```
A(layer, head, kv, pos, d) = AR_KV_BASE
                           | (slab_index << 21)
                           | (pos << 8)
                           | (d << 1)
```

其中（等价展开 `slab_index << 21 = (layer<<25) | (head<<22) | (kv<<21)`，与 IsaSpec 编码一致）：

| 项 | 含义 | 位域 | 范围 |
|----|------|------|------|
| `AR_KV_BASE` | KV 区域基址（HBM 字节） | `A[39:21]`（低位强制为 0） | 2 MiB 对齐 |
| `slab_index` | `(layer<<4)\|(head<<1)\|kv` | `A[30:21]` | `[0,576)` |
| `pos` | token 位置（KV 位置） | `A[20:8]` | `[0,8192)` |
| `d` | head 内维度索引 | `A[7:1]`，`d∈[0,128)` | 元素 = `d<<1` 字节 |

等价写法（任务给定形式，显式展开 kv）：

```
slab_base(layer, head) = AR_KV_BASE + (layer×16 + head×2) × 2²¹   // K slab 基址
slab_base(layer, head, V) = slab_base(layer, head) + 2²¹           // V slab 基址 = K + SLAB_SIZE
addr(K) = slab_base(layer, head)    + pos × 256B + d × 2B
addr(V) = slab_base(layer, head, V) + pos × 256B + d × 2B
```

> 位域拆分说明：`pos<<8` 恰好覆盖 `A[20:8]`（13 bit），`slab_index<<21` 覆盖 `A[30:21]`（10 bit，`slab_index<576<2¹⁰`），`d<<1` 覆盖 `A[7:0]`（8 bit，token 行内 256 B）。三者位域无重叠，故可用按位或拼装，也可用加法；`AR_KV_BASE` 已保证 `A[20:0]=0`，加法与按位或等价。

> **SLAB_SHIFT 参数化（load-time，配合 03-memory §7.3）**：默认 SLAB_SHIFT=21（8K slab、pos 13b、
> slab 步长 $2^{21}$）；BF16 参考模式用 SLAB_SHIFT=20（4K slab、pos 12b、slab 步长 $2^{20}$、
> `AR_KV_BASE` 1 MiB 对齐）。SLAB_SHIFT=22（4 MiB slab、容量 $2^{14}$=16K token、pos 仍 13b）用于
> 容纳 decode 第 8193 个 KV slot（pos=8192）：`KV.LOAD/STORE_BLOCK` 的 13-bit `pos_start` 仍限
> $[0,8192)$，pos=8192 的当前 token 由 DC 程序的 VMOV 尾 subtile 从 post-RoPE SRAM K/V 拷入 staging。
> 地址生成器按 SLAB_SHIFT（C31）取 pos 位宽与 slab 步长，公式结构不变。

### 1.4 对齐规则

| 对象 | 对齐要求 | 来源 |
|------|----------|------|
| `AR_KV_BASE` | 2 MiB（`A[20:0]=0`），隐含满足 64 B burst | HBM 40-bit，契约 §5 |
| 每块 slab | 2 MiB（`slab_index<<21`） | `SLAB_SIZE = 2²¹` |
| 每 token 行 | 256 B = 4 × 64 B burst，行首落在 burst 边界 | `pos<<8` |
| 每元素 | 2 B（BF16，`d<<1`） | 元素粒度 |
| HBM burst | 64 B 对齐 | 契约 §5 |

`pos` 从 `A[20:8]` 读取（13 bit），`pos=8191` 时 `pos×256 + 255 = 2²¹ − 1`，slab 恰好填满，无溢出。

### 1.5 SRAM 暂存区（KV 窗口 staging）

KV cache 常驻 HBM；attention 计算前，窗口经 `KV.LOAD`（或 `KV.GATHER`）载入 SRAM **KV staging 区域**（K 区基址 `0x30000` 字、V 区基址 `0x38000` 字）。`KV.LOAD` 的 `dstK/dstV` 与 `KV.GATHER` 的 `dst` 均为 AR 可寻址，**v0 编译器默认将其指向该 KV staging 区域**；物理 bank 由 SRAM 字节地址决定（契约 §4）：`bank = addr[7:4]`（16B 粒度 16 路交错）。

| 用途 | 区域 | SRAM 字节范围 | SRAM 字范围 | 大小 | 布局 |
|------|------|---------------|-------------|------|------|
| K 窗口 tile | KV staging K 区 | `[0x300000, 0x380000)` | `[0x30000, 0x38000)` | 512 KiB | token-major，token stride 256 B |
| V 窗口 tile | KV staging V 区 | `[0x380000, 0x400000)` | `[0x38000, 0x40000)` | 512 KiB | token-major，token stride 256 B |

- 每 token 行 256 B = 16 个 16-B word；K tile 内 token `t` 的第 `d` 个元素位于 SRAM 字 `0x30000 + t×16 + (d>>3)`，元素在 word 内偏移 `d&7`（自然升序；word 内部 2B 半字的微调归 MemorySystem 做 bank 冲突优化）。
- **tile 容量**：每区 512 KiB。`KV.LOAD`（单副本）`T_max = 512 KiB / 256 B = 2048` token；`KV.GATHER`（广播 ×4，4 副本）`T_max = 512 KiB / (4 × 256 B) = 512` token。窗口按 `⌈window/T⌉` 次流式载入（`T` 取对应上限）。

---

## 2. 寄存器 ABI

契约 §2 规定地址走 AR（64×64b）间接、维度/stride/KV 位置/scale 走 C（32×32b）。寄存器**索引由 IsaSpec 冻结**（02-isa），本协议只定语义：

| 寄存器 | 语义 | 宽度/模式 | 说明 |
|--------|------|-----------|------|
| `AR63` = `AR_KV_BASE` | KV 区域基址 | HBM 模式（`bit63=1`，`[39:0]` 40-bit 字节地址） | 必须 2 MiB 对齐；KV 指令所有 HBM 地址由它派生 |
| `C30` = `C_KV_POS` | 当前 KV 位置 `pos`（已存 token 数，0-based） | 32-bit（有效 13-bit） | decode 每 token +1；`KV.APPEND` 隐式读取 |
| `Cstride`（`KV.GATHER` `[53:49]` 选中的 C 寄存器） | Q head 广播目的 stride（SRAM 字地址间距） | 32-bit | 默认 `16`（=`128×2B/16B`）；compiler 设 |
| 各 KV 指令的 `srcK/srcV`/`dstK/dstV`/`dst` | SRAM 地址 | SRAM 模式（`bit63=0`，`[18:0]` 字地址） | 通过 AR 索引指定（见 §4 各指令字段） |
| `layer` / `head` / `pos_start` / `count` / `sel` | 指令操作数 | 立即数（imm） | 见 §4 各指令 |

> `AR_KV_BASE` 是唯一 HBM 基址真值：所有 KV 指令只携带 `layer/head/pos` 等小字段，HBM 大地址由 DMA/KV 地址生成单元按 §1.3 公式计算，避免为 288 个 slab 预存 288 个 AR。

---

## 3. 容量核算

每 token（一个位置，全层全头，K+V）：

```
B/token = LAYERS × KV_HEADS × HEAD_DIM × 2B × 2(K,V)
        = 36 × 8 × 128 × 2 × 2
        = 147,456 B = 144 KiB
```

| context 长度 | token 数 | 总字节 | MiB | GiB | 占 16 GiB HBM 比例 |
|-------------|----------|--------|-----|-----|--------------------|
| 4K | 4,096 | 603,979,776 | 576 | 0.5625 | 3.52 % |
| 8K | 8,192 | 1,207,959,552 | 1,152 | 1.125 | 7.03 % |
| 32K | 32,768 | 4,831,838,208 | 4,608 | 4.5 | 28.13 % |
| 40,960（`max_position`） | 40,960 | 6,039,797,760 | 5,760 | 5.625 | 35.16 % |

推导示例（8K）：`8 × 1024 × 147,456 B = 1,207,959,552 B`；`÷ 2³⁰ = 1.125 GiB`；`÷ 16 GiB = 7.03 %`。4K/32K/40K 均为 8K 的 0.5×/4×/5× 线性缩放（`147,456 B/token` 恒定）。

**容量压力曲线结论**：

- v0 每 slab 容量固定 **8K token**，对应冻结的 4K→8K context 预算（D5），8K 时 KV 占 HBM **7.03%**，模型权重（BF16，十亿参数量级 ×2B）仍是 HBM 主消费者。
- **32K / 40K 超出 v0 slab 容量**（需 4×/5× slab），v0 不可原生支持。缓解方向（均为 backlog）：① 增大 slab 至 32K/40K（slab 尺寸 ×4/×5，KV 占比升至 28%/35%）；② 分页 KV（paged KV，契约 §8 backlog）。v0 只保证 8K。

> **0.6B 参数注记**（验证模型，01b 权威）：28 层 × 8 head → 每 token 114,688 B（112 KiB），
> 4K 448 MiB / 8K 896 MiB；GQA = 2:1（16 Q / 8 KV）；KV.APPEND = 28×8 = 224 条/token；
> `broadcast=1`（×4）对 2:1 冗余 2 副本——功能无碍、SRAM footprint 加倍，P6 列为优化项。
> 8K slab（SLAB_SHIFT=21）对 0.6B 同样适用（每 (layer,head) slab 2 MiB）。

---

## 4. 指令语义（操作数级，位级编码归 IsaSpec）

KV 指令在 128-bit 编码中占用 engine tag（KV/DMA 类）与 opcode；`[111:104]` dtype flags 在 v0 固定为 **BF16（code 0）**（KV 元素 BF16）。`[103:0]` 引擎自定义操作数的**位级编码由 IsaSpec 冻结**（02-isa），下表为操作数语义与位域对照。

### 4.1 `KV.APPEND` — decode 单 token 追加

将一个 token 的 K、V（各 `128×2B=256B`，已在 SRAM）追加到 HBM 中 `(layer, head)` 对应 slab 的 `pos` 位置。

| 操作数 | 位域 | 宽度 | 来源 | 语义 |
|--------|------|------|------|------|
| `srcK` | `[103:98]` | 6 bit | AR 索引（SRAM 模式） | 新 token K 的 SRAM 字地址（256 B） |
| `srcV` | `[97:92]` | 6 bit | AR 索引（SRAM 模式） | 新 token V 的 SRAM 字地址（256 B） |
| `layer` | `[91:86]` | 6 bit | imm | slab 行选择 `[0,36)` |
| `head` | `[85:83]` | 3 bit | imm | KV head `[0,8)` |
| `pos` | — | — | `C30`（隐式） | 追加位置（`C_KV_POS`） |

写地址：K → `A(layer,head,0,pos,*)`，V → `A(layer,head,1,pos,*)`，各连续 256 B。decode 每 token 每个 head 各 1 次 APPEND（`36×8 = 288` 次/step）。

### 4.2 `KV.STORE_BLOCK` — prefill 128-token 块写入

将 prefill 一个 block 的 K、V（每 head `128 token × 256 B = 32 KiB`）写入 slab 的连续区间 `[pos_start, pos_start+count)`。

| 操作数 | 位域 | 宽度 | 来源 | 语义 |
|--------|------|------|------|------|
| `srcK` | `[103:98]` | 6 bit | AR 索引（SRAM 模式） | K 块基址（`count×256 B`） |
| `srcV` | `[97:92]` | 6 bit | AR 索引（SRAM 模式） | V 块基址 |
| `layer` | `[91:86]` | 6 bit | imm | slab 行选择 |
| `head` | `[85:83]` | 3 bit | imm | KV head |
| `pos_start` | `[82:70]` | 13 bit | imm | 块起始 token 位置 |
| `count` | `[69:56]` | 14 bit | imm | token 数，**默认 128**（v0 冻结），字段上限 8192 |

写地址：`A(layer,head,kv,pos,d)` 中 `pos ∈ [pos_start, pos_start+count)`。prefill 每层每 head 1 次 STORE_BLOCK（每 block `8` 次）。

### 4.3 `KV.LOAD` — attention 前窗口载入（单副本 staging）

将 `(layer, head)` 的 K、V 窗口 tile `[pos_start, pos_start+count)` 从 HBM 载入 SRAM staging（`dstK`/`dstV` 指向，默认指向 KV staging 区域：K 区基址 `0x30000` 字、V 区基址 `0x38000` 字）。

| 操作数 | 位域 | 宽度 | 来源 | 语义 |
|--------|------|------|------|------|
| `dstK` | `[103:98]` | 6 bit | AR 索引（SRAM 模式） | K 目的基址（默认 KV staging K 区基址 `0x30000` 字） |
| `dstV` | `[97:92]` | 6 bit | AR 索引（SRAM 模式） | V 目的基址（默认 KV staging V 区基址 `0x38000` 字） |
| `layer` | `[91:86]` | 6 bit | imm | slab 行选择 |
| `head` | `[85:83]` | 3 bit | imm | KV head |
| `sel` | `[82:81]` | 2 bit | imm | `0=K`、`1=V`、`2=both`（默认 both）、`3=resv` |
| `pos_start` | `[80:68]` | 13 bit | imm | tile 起始 token 位置 |
| `count` | `[67:54]` | 14 bit | imm | tile token 数 `≤ T_max=2048`（字段上限 8192） |

载入地址：K → `A(layer,head,0,pos_start+d,*)` 写 `dstK`；V → `A(layer,head,1,pos_start+d,*)` 写 `dstV`，`d∈[0,count)`。窗口大于 tile 时按 `⌈window/T⌉` 次调用流式覆盖。

### 4.4 `KV.GATHER` — GQA stride 广播

将 `kv_head` 的 K/V 窗口 `[pos_start, pos_start+count)` 从 HBM 读出，按 GQA 广播到同组 4 个 Q head 的目的 SRAM 缓冲。

| 操作数 | 位域 | 宽度 | 来源 | 语义 |
|--------|------|------|------|------|
| `dst` | `[103:98]` | 6 bit | AR 索引（SRAM 模式） | 广播目的基址（第 0 个 Q head 的 K/V 缓冲） |
| `AR2` | `[97:92]` | 6 bit | reserved | both 扩展预留；v0 单 `sel` 下发两条（K、V 各一条） |
| `layer` | `[91:86]` | 6 bit | imm | slab 行选择 |
| `head` | `[85:83]` | 3 bit | imm | 源 KV head |
| `sel` | `[82]` | 1 bit | imm | `0=K`、`1=V` |
| `broadcast` | `[81]` | 1 bit | imm | `1` = GQA 广播 ×4（`g=4`） |
| `pos_start` | `[80:68]` | 13 bit | imm | 窗口起始 token |
| `count` | `[67:54]` | 14 bit | imm | token 数（字段 ≤8192；v0 按 4-副本 footprint ≤512） |
| `Cstride` | `[53:49]` | 5 bit | C 寄存器索引 | Q head 目的 stride（SRAM 字地址间距），默认 `16` |

> **Cstride 与 count 联动（规格完整性）**：默认 `16` 字仅对 `count=1` 成立（单 token 行 256 B = 16 字）；`count>1` 时编译器须设 `Cstride = count×16` 字，使相邻 Q head 目的 tile 相隔 `count` 个 token 行、互不重叠。GATHER 为非默认路径，此注补全规格。

**stride 表达**（GQA，`Q_HEADS/KV_HEADS = 32/8 = 4`）：

- Q head → KV head 映射：`kv_head = q_head >> 2`；反向 `q_head(i) = (kv_head << 2) | i`，`i ∈ [0,4)`。
- 源（KV 地址空间）：同组 4 个 Q head 读**同一** `kv_head` slab，`slab_index` 不变 → **源 stride = 0（HBM 只读一次）**。
- 目的（SRAM 地址空间）：`dst_i = dst + i × Cstride`（SRAM 字地址），`Cstride` 默认 = `HEAD_DIM × 2B / 16B = 128 × 2 / 16 = 16` 字。
- 计算（score/output 空间）：各 Q head 的 score 行/输出行 stride = `HEAD_DIM = 128` 元素。

> 净效果：一次 `KV.GATHER` 对 HBM 产生 1 次窗口读，在 datapath 内复制 `g=4` 份写向 4 个 Q head 目的缓冲，避免 4 次独立 HBM 读。

---

## 5. 流程时序

`MODE` 由 SYS 指令设置：`MODE PF`（GEMM 模式，prefill）与 `MODE DC`（16×8 GEMV lane 模式，decode）。**v0 batch=1，全程只有一次 PF→DC 切换**；若后续多轮对话需再次 prefill，需重新 `MODE PF`。

### 5.1 prefill（`MODE PF`，prompt 分 128-token block）

约定 `B=128`，block `b` 覆盖 token `[s, s+128)`，`s = b×128`。窗口（causal）= `[0, s+128)`。

| 步 | 操作 / 指令 | 引擎 | MODE | 说明 |
|----|-------------|------|------|------|
| 0 | `MODE PF`；`CONFIG` 设 `C_KV_POS=0`、写 `AR_KV_BASE`、`Cstride` | CP/SYS | **PF** | 初始化；PF 为复位默认 |
| 1 | `DMA.LOAD` 载入 block 输入 hidden `128×4096` → SRAM | DMA | PF | 首个 block 读 embedding |
| 2 | QKV 投影 `GEMM` → Q/K/V | Matrix | PF | Q `128×4096`，K/V 各 `128×1024` |
| 3 | `RMSNORM`(QK-norm) + `ROPE` 于 Q、K | Vector | PF | Q/K 的 per-head RMSNorm 在 RoPE 前（契约 §10） |
| 4 | `KV.STORE_BLOCK`（head 0..7，共 8 次）写 K/V 块 | KV/DMA | PF | 每 head 写 32 KiB |
| 5 | `BARRIER`（KV 写已提交、可见） | CP | PF | 写后读依赖（§6） |
| 6 | 每 `kv_head h`（0..7）：`KV.LOAD`（K 窗 + V 窗 → KV staging 区域，单副本，tiled） | KV/DMA | PF | 窗口 `≤ s+128`，超 2048 分 tile；K/V 由阵列在 GQA 组内共享，无需 GATHER |
| 7 | `GEMM` QKᵀ：`(128×128)×(128×window)` → scores | Matrix | PF | 每 Q head 一次（4 Q head 共享 K） |
| 8 | `VMASK`（causal 上三角 `-inf`）+ softmax | Vector | PF | block 内各 token 前缀长度不同，需掩码 |
| 9 | `GEMM` AV：`scores(128×window)×V(window×128)` → 输出 | Matrix | PF | 每 Q head 一次 |
| 10 | 输出投影 `GEMM` → 残差 + `RMSNORM` | Matrix/Vector | PF | 得下一 block / 下一层输入 |
| 11 | 全部 block、全部 layer 完成后：`BARRIER`，**`MODE DC`** | CP/SYS | **PF→DC** | **唯一模式切换点**，进入 decode |

> 本表为 attention/KV 相关步序；每层 MLP（gate/up GEMM → VSILU/VMUL → down GEMM → 残差）与
> 末层 LM head（RMSNORM + 151,936 列 GEMV + host 采样）的完整步序见 02-isa §12 第 7–8 步。
> O 投影产出 hidden，logits 由末层 LM head 产生。

### 5.2 decode（`MODE DC`，单 token）

每 token 位置 `pos = C_KV_POS`。窗口（causal）= `[0, pos]`（`pos+1` token，含新 token 自身）。

| 步 | 操作 / 指令 | 引擎 | MODE | 说明 |
|----|-------------|------|------|------|
| 0 | （已在 DC）`CONFIG` 使 `C_KV_POS = pos` | CP/SYS | DC | — |
| 1 | QKV 投影 `GEMV`（`1×4096×W`）→ Q/K/V | Matrix | DC | 16 lane × 8 行 GEMV，K 流式 |
| 2 | `RMSNORM`(QK-norm) + `ROPE` 于 Q、K | Vector | DC | 新 token 的 Q/K |
| 3 | `KV.APPEND`（head 0..7，共 8 次）追加 K/V 至 `pos` | KV/DMA | DC | 每 head 写 256 B×2 |
| 4 | `BARRIER`（APPEND 已提交、可见） | CP | DC | 写后读依赖（§6） |
| 5 | 每 `kv_head h`（0..7）：`KV.LOAD`（`sel=both`，tile ≤2048） | KV/DMA | DC | 窗口 `[0,pos]`，超 2048 分 tile；K/V 由 P7 内部广播总线供 GQA 组内各 lane 共享（0.6B 2 个 / 8B 4 个） |
| 6 | `GEMV` QKᵀ：`(1×128)×(128×(pos+1))` → scores | Matrix | DC | 每 Q head 一次（DC 独立 lane） |
| 7 | `VMASK`（可选，窗口已是 causal 前缀，掩码全通过）+ softmax | Vector | DC | decode 无未来位置，掩码可省 |
| 8 | `GEMV` AV：`scores(1×(pos+1))×V((pos+1)×128)` → 输出 | Matrix | DC | 每 Q head 一次 |
| 9 | 输出投影 `GEMV` → 残差 + `RMSNORM` | Matrix/Vector | DC | 得 logits / 下一层输入 |
| 10 | `CONFIG`：`C_KV_POS += 1` | CP/SYS | DC | 下一 token 位置 |

> 本表为 attention/KV 相关步序；MLP 与 LM head 步序见 02-isa §12 第 7–8 步（同 prefill 表注）。

> **双模切换点汇总**：PF→DC 在 prefill 全部完成之后、首个 decode token 之前（5.1 步 11）发生一次。decode 全程维持 DC，无 DC→PF。prefill 内部无论 block 大小均用 PF（GEMM）；decode 每步均用 DC（GEMV）。attention 的 QKᵀ 在 prefill 是 GEMM（block×window），在 decode 是 GEMV（1×window），由 `MODE` 统一选择阵列形态，无需额外指令。
>
> **LOAD vs GATHER 的 v0 分工（已冻结）**：prefill 用 `KV.LOAD`（GEMM 路径，K/V 为 weight-stationary 共享矩阵操作数，单副本即可，tile ≤2048）；decode 也已冻结走 `KV.LOAD` 单副本（tile ≤2048），K/V 由 P7 内部广播总线在 GQA 组内共享（ISA 不变）。`KV.GATHER` 保留于 ISA（8B 4:1 GQA 或多副本场景，P7 可再评估）。依据：LOAD 65,736 vs GATHER 262,944 cyc/层@4K（4.00×，0.6B 2:1 GQA 下 4 副本中 2 份冗余）。已冻结（2026-08-13，P3 裁决）：decode = KV.LOAD 单副本。

---

## 6. 一致性（写后读依赖与并发读）

### 6.1 写后读依赖（BARRIER / WAIT 配合）

KV 的写（`KV.APPEND` / `KV.STORE_BLOCK`）与读（`KV.LOAD` / `KV.GATHER`）之间存在 HBM 写可见性依赖，必须显式同步：

```
KV.APPEND / KV.STORE_BLOCK    // 写 HBM
BARRIER                       // 确保写入已提交并对后续读可见
KV.LOAD / KV.GATHER           // 读 HBM → SRAM（staging / Q-head 广播）
WAIT                          // 确保读完成、SRAM 已填满
(attention 计算 GEMV/GEMM)    // 消费 SRAM 数据
```

- `BARRIER`：程序序屏障，保证其之前所有 KV/DMA 写在其之后的读之前完成并可见（跨引擎 ordering）。
- `WAIT`：阻塞至指定引擎/事件完成——上例 `WAIT KV`（或 `WAIT DMA`）确保 `KV.LOAD`/`KV.GATHER` 的 SRAM 填充完成，attention 计算才能开始。
- 时序表中 5.1 步 5、5.2 步 4 的 `BARRIER` 即此写后读边界；`KV.LOAD`/`KV.GATHER` 与 attention 计算之间由 `WAIT` 衔接（可在软件流水/双缓冲下放宽，但 v0 语义上等价于此顺序）。

### 6.2 多 Q head 共享 KV head 的并发读约束

- 4 个 Q head（同一 GQA group）**并发读同一 KV head**：读操作无副作用、幂等，故允许并发；`KV.GATHER` 把「4 次逻辑读」收敛为「1 次 HBM 读 + 4 路 SRAM 目的写」，窗口数据只从 HBM 取一次。
- **稳定约束**：并发读期间，该 slab 上不得有在途的 `APPEND`/`STORE_BLOCK`。v0 batch=1 单序列，唯一写者是当前 decode/prefill step，且写后紧跟 `BARRIER`（6.1），故 attention 读窗口期间 KV 内容已稳定。
- **层间边界**：layer `L` 的 KV 写与 layer `L` 的 attention 读同层内完成；跨层无共享 KV 依赖（每层独立 slab），`BARRIER` 仅需覆盖同层写→读。
- **SRAM 写冲突**：×4 副本的写放大为带宽成本；交错方案下每行 256 B 已跨 16 bank、1W/bank 并行，无单 bank 串行化（GATHER 为非默认路径）。

---

## 7. 与契约一致性自查

| 契约项 | 本切片落实 |
|--------|-----------|
| §1 指令 128-bit，HBM 大地址经 AR 间接 | §4：KV 指令只携 `layer/head/pos` 小字段，HBM 地址由 `AR_KV_BASE` + 公式生成 |
| §2 AR 64×64b / C 32×32b | §2：`AR63`（HBM 模式）+ `C30`（KV 位置）+ `Cstride`（广播 stride） |
| §4 SRAM 16 bank、bank=addr[7:4]（16B 粒度 16 路交错） | §1.5：staging 默认 KV staging 区域（K/V 区），字节/字范围与 bank 位域一致 |
| §5 HBM 16 GiB、40-bit、64B burst | §1.3/1.4：40-bit 地址、2 MiB 对齐、256B 行 = 4 burst |
| §8 每 (layer,kv_head) 连续 slab、8K 容量、BF16 元素、GQA stride 广播、分页 KV backlog | §1.1/1.2、§3、§4.4；分页 KV 与 32K+ slab 列为 backlog |
| §7 MODE PF/DC | §5：PF=prefill GEMM，DC=decode GEMV，切换点唯一 |
| 数值 §10（hidden 4096/36 层/8 kv_head/128 dim/32 q_head） | §3 容量公式与 §5 流程参数全部采用 |

**Backlog（不进 v0）**：分页 KV（paged KV）、32K/40K slab 扩容、多轮对话的二次 prefill（需 `MODE PF` 重入）、KV 量化（INT8/INT4 KV 元素 + per-group scale 元数据，届时 `dtype` 字段与 `d<<1` 步长需相应改写）。
