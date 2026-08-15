# 03 — 内存系统 (SRAM / HBM / DMA) v0

> 本切片为 **SRAM/HBM/DMA 物理行为**的权威定义（契约第 9 条）：bank 组织与冲突仲裁、HBM 带宽/延迟/突发、
> DMA 2D 搬移、SRAM 容量论证与 bank 划分、HBM 驻留布局。
> 指令的**位级编码与语义**归 **IsaSpec**（02-isa）；KV 的**地址公式与协议**归 **KvProtocol**（05-kv-cache）；
> 阵列/向量的**执行单元行为**归 **ExecEngines**（04）。本文件给出的带宽/延迟/容量是各切片之间的 binding 接口，
> 冲突时以 P0 冻结契约为准并报主会话。

---

## 1. 定位与范围

newlpu 的存储层次只有两层：8 MiB 片上 Scratchpad SRAM（计算本地）与 16 GiB HBM（权重 + KV cache 主存）。
无 L1/L2 cache、无标量访存——所有数据搬移一律经 DMA 或 KV 引擎的 16B 粒度 burst 完成。

本文件定义：

1. SRAM 16 bank 组织、bank 选择位、16B 粒度、bank 冲突仲裁（何时 stall、服务能力）、引擎端口分配；
2. HBM 1.2 TB/s 读/写拆分、延迟模型公式、64B 突发对齐、效率假设；
3. DMA 引擎 2D transfer（严格对应 ISA 的 `DMA.LOAD/STORE/PREFETCH` 操作数）、双缓冲、PREFETCH+WAIT、in-flight 上限；
4. SRAM 容量论证（权重 tile / prefill QK / decode GEMV 权重流 / KV 窗口）与最紧约束；
5. SRAM bank 用途划分表；
6. HBM 驻留布局（含 BF16 模式的约束声明与张力解决）。

---

## 2. SRAM 组织

### 2.1 容量与地址

契约第 4 条：8 MiB = 16 bank × 512 KiB；字节寻址，访问粒度 16 B；bank 由 **字节地址** B[7:4] 选择（16B 粒度 16 路交错）。

| 量 | 推导 | 值 |
|----|------|-----|
| 总容量 | $16 \times 512\ \text{KiB} = 16 \times 2^{19}$ B | $2^{23}$ B = 8 MiB ✓ |
| 字粒度 | 访问粒度 16 B → 1 字 = 16 B | $2^{23}/16 = 2^{19}$ 字 |
| 字地址宽度 | $\log_2 2^{19} = 19$ bit | `[18:0]` ✓ |
| bank 选择 | 字地址 `[3:0]` = 字节地址 `B[7:4]`（4 bit） | 16 bank ✓ |
| bank 内字 | 字地址 `[18:4]` = 字节地址 `B[22:8]`（15 bit） | $2^{15} \times 16$ B = 512 KiB ✓ |

**字节地址 ↔ 字地址换算**（byte 地址 23 bit `B[22:0]`）：

```
字地址 W[18:0] = B[22:4]              // 16B 内字节偏移 = B[3:0]
bank   = W[3:0] = B[7:4]
bank 内字 = W[18:4] = B[22:8]
```

**16B 交错**：连续 16B 字映射到连续 bank（16 路交错），故 **连续 256 B = 16 字恰好命中 16 个 bank 各一次**。
顺序流式访问（权重行、KV token 行 256 B）天然无 bank 冲突。

### 2.2 带宽与端口

**物理端口（契约固定）**：每 bank 每周期 ≤ 2 读 + 1 写，每口 16 B/拍。

| 方向 | 推导 | 峰值带宽 |
|------|------|----------|
| 读 | $16\ \text{bank} \times 2\ \text{口} \times 16$ B | **512 B/cycle** = 512 GB/s @1 GHz |
| 写 | $16\ \text{bank} \times 1\ \text{口} \times 16$ B | **256 B/cycle** = 256 GB/s @1 GHz |

> **关键设计点**：SRAM 写带宽（256 GB/s）是 HBM 读带宽（900 GB/s）的 1/3.5。因此 `DMA.LOAD`（HBM→SRAM）
> 恒为 **SRAM 写受限**；decode 权重流（`bsrc=1`）不经 SRAM 写、直接 HBM→阵列，才能吃到 900 GB/s（见 §3、§8）。

**引擎端口分配表（v0，banked crossbar，每逻辑端口跨全部 16 bank）：**

| 逻辑端口 | 服务引擎 | 方向 | word-port（16B/个） | 峰值 B/cycle | 用途 |
|---------|---------|------|--------------------|--------------|------|
| W（权重口） | MATRIX B | R | 16（1/bank） | 256 | 权重 tile 载入/流式；INT8 用 128、BF16 用满 256 |
| A（激活口） | MATRIX A | R | 16（1/bank） | 256 | 激活 A 流式（128 元素/拍）；INT8 128、BF16 256 |
| C（累加口） | MATRIX C | W | 16（1/bank） | 256 | C tile 写出（INT32 512 B/行 = 2 拍/行；tile 边界、amortized） |
| V（向量口） | VECTOR | R/W | 16R + 16W | 256 R / 256 W | `ARa`/`ARb` 读 + `ARd` 写 |
| D（搬移口） | DMA | R/W | 16R + 16W | 256 R / 256 W | `LOAD` 写 SRAM / `STORE` 读 SRAM |
| K（KV 口） | KV | R/W | 复用 D | 复用 D | KV staging（K/V 区 0x300000–0x3FFFFF），复用 D 口带宽 |

**过载（over-subscribed）设计**：逻辑端口峰值之和（读 $4\times256=1024$、写 $4\times256=1024$ B/cycle）
大于物理预算（512 R / 256 W）。crossbar 按 §2.3 仲裁，任何 bank 单周期超过 2R+1W 即 stall。
**关键并发场景（软件流水，必须零冲突）**——DMA 写下一 tile 权重到权重缓冲 B 区，同时 MATRIX 从权重缓冲 A 区读权重、
从激活区读激活——权重刷新+激活各占 1 读、DMA 权重写占 1 写，每 bank 恰 2R+1W，保证 **256 R + 256 W 同时达成**。

### 2.3 bank 冲突仲裁

- **冲突判据**：某 bank 单周期收到 **>2 个读** 或 **>1 个写** 请求 → 冲突。
- **stall 行为**：按固定优先级只服务前 2 读 + 1 写，落选请求方插 bubble（下一周期重试）；不重排、不缓冲旁路。
- **固定优先级**（v0，确定性 P3 仿真）：`MATRIX.A > MATRIX.B > MATRIX.C > VECTOR > DMA > KV`。
- **无冲突配置**（compiler 目标）：每 bank 2R+1W 预算：权重刷新+激活各占 1 读、DMA 权重写占 1 写，恰满（各 256 B/cyc 流）；INT8 权重流 128 B/cyc 为余量情形；再加 Vector 读则 3R 超限 → 固定优先级仲裁 stall（§2.3）。
- **跨 bank 由 16 路交错自然覆盖**；`DMA.LOAD` 的目标地址若非 16B 对齐则由 DMA 以 masking 补齐（见 §4.1）。

---

## 3. HBM

### 3.1 容量与地址

| 量 | 值 |
|----|-----|
| 容量 | 16 GiB = $2^{34}$ B |
| 地址 | 40-bit 字节地址（设备实现用低 34 位） |
| 突发 | 64 B 对齐 |
| 聚合带宽 | 1.2 TB/s |

### 3.2 读/写拆分与理由

**决策：读 900 GB/s + 写 300 GB/s（3:1）**，共享 1.2 TB/s 总线（HBM 为半双工 DQ 总线，读写分时复用，
任一时刻 读+写 ≤ 1.2 TB/s，读写切换的 turnaround 开销计入 §3.4 效率）。

理由（以 Qwen3-8B 逐 token 流量论证）：

| 阶段 | 读流量/step | 写流量/step | 读:写 |
|------|-------------|-------------|-------|
| decode（每 token） | 权重 $7.57$ GB（INT8，36 层 dense $6.95$G + lm_head $622$M；embedding 仅首 token 读单行）+ KV 窗口重读（4K 时 $576$ MiB） | KV.APPEND $147{,}456$ B | $\approx 6\times10^4 : 1$ |
| prefill（每 128-token block 每层） | 该层权重 $\approx 217$ MiB（QKV+O+MLP）+ KV 读 | KV.STORE_BLOCK $512$ KiB + 输出 staging | $\gg 100 : 1$ |

decode 由 KV.APPEND 产生的写仅为 $147{,}456$ B/token，与 $7.63$ GiB/token 的权重读相比可忽略；
prefill 的写主要是 KV 块落盘，同为读主导。**读:写 = 3:1 已保守**，保留 300 GB/s 写余量以覆盖
`KV.STORE_BLOCK` 突发与输出 staging。这是 HBM 控制器的独立峰值读/写上限，非同时满。

### 3.3 延迟模型（DMA 发起到首字到达）

$$
T_{\text{first}} = C_{\text{fixed}} + C_{\text{align}} + Q \quad [\text{cycles @1 GHz}]
$$

| 项 | 值 | 说明 |
|----|-----|------|
| $C_{\text{fixed}}$ | **100** | HBM 固有延迟：$t_{\text{RCD}}+t_{\text{CAS}}+t_{\text{RL}}$ ≈ 60 ns + 控制器/PHY 流水 ≈ 40 ns → 100 ns = 100 cycles |
| $C_{\text{align}}$ | **0**（64B 对齐）/ **≤3**（非对齐） | 首段 partial-burst 补齐（≤ 64B/16B = 4 拍，首拍之后 3 拍对齐） |
| $Q$ | **0**（总线空闲）/ 排队 | 否则 $Q = \lceil \text{在途剩余字节} / 900 \rceil$（读） |

**传输时间**：

$$
T_{\text{xfer}} = \max\left( T_{\text{hbm}},\ T_{\text{sram}} \right),\qquad
T_{\text{hbm}} = \left\lceil \frac{\text{Bytes}}{BW_{\text{eff}}} \right\rceil,\quad
T_{\text{sram}} = \left\lceil \frac{\text{Bytes}}{BW_{\text{sram side}}} \right\rceil
$$

| 方向 | $BW_{\text{eff}}$（HBM 侧） | $BW_{\text{sram side}}$ | 瓶颈 |
|------|---------------------------|------------------------|------|
| LOAD（HBM→SRAM） | 读 720 B/cycle | SRAM 写 256 B/cycle | **SRAM 写**（256 < 720） |
| STORE（SRAM→HBM） | 写 240 B/cycle | SRAM 读 512 B/cycle | **HBM 写**（240 < 512） |

总完成周期：$T_{\text{done}} = T_{\text{first}} + T_{\text{xfer}} + T_{\text{drain}}$（$T_{\text{drain}}$ = 末字写 SRAM 收尾，≈ 数拍）。

> 效率口径：$BW_{\text{eff}}$ 为 sustained（峰值 ×80%，见 §3.4）。**LOAD 恒 SRAM 写受限**、**STORE 恒 HBM 写受限**，
> 是 P3 时序仿真的两个固定结论。

> **v0 周期模型简化声明**：周期口径按 §3.3 公式计费 $T_{\text{first}} + T_{\text{xfer}} + T_{\text{drain}}$，**不计**
> (a) 非对齐传输的 head/tail partial burst（§3.4，≤2 burst/传输，64 B 对齐流为 0）、
> (b) 多 in-flight 传输的带宽重叠（§4.4）。二者量级 ≤0.05%（256 KiB tile，$2\times64\ \text{B}/262144$）／
> <0.03%（512 KiB tile），被 §3.4 的 80% sustained 余量覆盖，**roofline 锚点（读 720 / 写 240 GB/s）不变**；
> qsim timing 与 RTL co-sim 同口径；重标定另立计划。

### 3.4 突发与对齐、效率假设

- **64 B 突发**：所有 HBM 访问按 64 B 对齐 burst 进行。非 64 B 对齐的传输由 DMA 拆为
  `head(partial, masking) + body(64B 对齐) + tail(partial, masking)`，最多损失 2 个 partial burst。
- **对齐规则**：`.qbin` 权重 section 64B 对齐（00-container）；KV token 行 256 B = 4×64 B 行首对齐（05-kv-cache）；
  SRAM 侧 `RowBytes` 必须 16B 倍数（ISA §5.1）。三者满足则 burst 效率 ≈ 100%。
- **效率假设（roofline 口径，P1/P6 统一引用）**：
  - HBM sustained/peak = **80%**（刷新 ~3–5%、bank 冲突、读↔写 turnaround、行激活）。
  - 读 sustained = $900 \times 0.8$ = **720 GB/s**；写 sustained = $300 \times 0.8$ = **240 GB/s**。
  - 64B 对齐的 dense 权重/KV 流 burst 效率 ≈ 100%；非对齐 partial burst 额外开销 ≤ 2 burst/传输。

---

## 4. DMA 引擎

### 4.1 2D transfer 语义（与 ISA §5.1 一一对应）

操作数**严格取** ISA 的 `SrcAR, DstAR, RowBytes, NumRows, StrideC, mode`，不新增操作数：

| 操作数 | 语义（ISA 权威） | 本切片物理实现 |
|--------|------------------|----------------|
| `SrcAR` | 源基址 AR | bit63 决定 HBM/SRAM 地址空间 |
| `DstAR` | 目标基址 AR | 同上 |
| `RowBytes` | 每行字节数（16-bit，必须 16 倍数） | 每行拆 16B 字，跨 bank 交错 |
| `NumRows` | 行数（16-bit；1D 必须=1） | 行循环 |
| `StrideC` | C 寄存器：源侧行 stride（字节） | 2D 源地址步进 |
| `mode` | `0`=1D 连续 / `1`=2D 带 stride | 选择地址生成模式 |

**地址生成**（与 ISA §5.1 严格一致）：

```
1D:  src_addr = SrcAR，连续拷贝 RowBytes 字节（忽略 Stride）
2D:  src_row(r) = SrcAR + r × C[StrideC]；每行 RowBytes 字节；共 NumRows 行
目标（SRAM 内）密集存放（连续无 stride）
```

**方向**：`LOAD`＝HBM→SRAM；`STORE`＝SRAM→HBM；`PREFETCH`＝HBM→SRAM、非阻塞/建议性。
**对齐处理**：`RowBytes % 16 == 0`（SRAM 粒度，非法则 fault）；HBM 侧 64B 由 DMA 以
`head/body/tail` 拆分 + masking 处理（§3.4），跨 bank 由 16 路交错覆盖。

> `RowBytes` 为 16-bit（≤ 65535 B），故**单行 ≤ 64 KiB**；512 KiB 权重 tile 用 **2D**（`RowBytes=128`、
> `NumRows=4096`、`Stride=128`）或拆多行 1D，两种编码等价（见 §8.1）。

### 4.2 双缓冲

- 权重 tile 双缓冲固定占 **权重缓冲 A/B 两区**（0x000000–0x07FFFF / 0x080000–0x0FFFFF）各 512 KiB（INT8）：DMA 填 B 区时 MATRIX 从 A 区算，
  下一拍交换。交换点由 `WAIT`（`eng_mask` bit0=DMA）/ `BARRIER` 同步，compiler 只切换 `DstAR` 与 MATRIX 的 `ARb` 指针。
- 每 bank 2R+1W 预算：DMA 写 B 区（1 写/bank）与 MATRIX 读 A 区（1 读/bank）**零 bank 冲突**（§2.3 无冲突配置）。
- 推广：任何 HBM→SRAM 预取（KV 窗口、下一层权重）都用同构 ping-pong。

### 4.3 PREFETCH 与 WAIT

- `DMA.PREFETCH` 与 `LOAD` 同向（HBM→SRAM）但**非阻塞/建议性**：CP 不等它完成、不建立数据依赖，供后续
  MATRIX/VECTOR 消费；资源冲突时可丢弃，绝不 fault。
- 消费方在读取预取数据前必须 `WAIT`（`eng_mask` bit0=DMA）——阻塞至 DMA 队列排空，保证 SRAM 已填满。
  典型序列（权重/KV 预取）：

```
DMA.PREFETCH  ...            // 提前把下一 tile 搬进 SRAM（非阻塞）
<计算当前 tile>
WAIT  eng_mask=0b0001        // 等 DMA 排空 → 预取数据已就绪
MATRIX.GEMM / VECTOR.*       // 消费 SRAM 数据
```

- `WAIT` 只等引擎排空，不等内存一致性（SRAM 写对 MATRIX/VECTOR 可见由引擎顺序保证，ISA §4.4）。

### 4.4 最大 in-flight 传输

- DMA 硬件描述符池 **4 个 in-flight 传输**（每传输 = 一条 LOAD/STORE/PREFETCH），FIFO 顺序完成（不重排）。
- v0 编译器实际使用 ≤ **2**（双缓冲 ping-pong，§4.2）；PREFETCH 可额外在途（建议性，不占硬依赖）。
- 单传输上限由 SRAM 容量约束：`LOAD` 目标 ≤ 8 MiB（SRAM 上限）、`STORE` 源 ≤ 8 MiB；实际 tile ≤ 1 MiB。

---

## 5. SRAM 容量论证（契约第 10 条数值）

目标模型 Qwen3-8B：`hidden=4096, layers=36, q_heads=32, kv_heads=8, head_dim=128, intermediate=12288`。

### 5.1 权重 tile 双缓冲

单投影权重 tile（K=hidden，N=阵列宽 128）：

| dtype | 推导 | 单 tile | 双缓冲 |
|-------|------|---------|--------|
| INT8 | $4096 \times 128 \times 1$ B | 512 KiB | **1 MiB** |
| BF16 | $4096 \times 128 \times 2$ B | 1 MiB | **2 MiB** |
| INT4 | $4096 \times 128 \times 0.5$ B | 256 KiB | **512 KiB** |

### 5.2 prefill QK 计算驻留

prefill 一个 128-token block 的 Q、K：

| 张量 | 推导 | 大小 |
|------|------|------|
| Q（128 token × 32 head × 128 dim，BF16） | $128 \times 32 \times 128 \times 2$ B | **1 MiB** |
| K（128 token × 8 head × 128 dim，BF16） | $128 \times 8 \times 128 \times 2$ B | 256 KiB |
| K+V | $\times 2$ | 512 KiB |

### 5.3 decode GEMV 权重流

decode DC 模式：阵列 128 行拆 16 lane × 8 行，每 lane 独立 GEMV（每 lane 128 N 列 × 8 K 行），K 维流式（契约第 7 条）：

$$
1\ \text{lane} \times 4096\ \text{K} \times 128\ \text{N} \times 1\ \text{B(INT8)} = 524{,}288\ \text{B} = \mathbf{512\ KiB}\ (\text{每 lane})
$$

16 lane 全批 = 8 MiB，经 HBM 直连流式（`bsrc=1`）进入阵列，**不占 SRAM 双缓冲**；
SRAM 权重双缓冲区（0x000000–0x0FFFFF，1 MiB）只服务 PF 模式权重 tile。

### 5.4 KV 窗口驻留

| 张量 | 推导 | 大小 |
|------|------|------|
| 128-token K+V 窗口 | $128 \times 8 \times 128 \times 2 \times 2$ B | 512 KiB |
| 完整 KV staging（K 区 0x300000–0x37FFFF + V 区 0x380000–0x3FFFFF） | $2 \times 512$ KiB | 1 MiB = **2048 token**（$2048 \times 256$ B = 各 512 KiB） |

> 单副本（KV.LOAD）tile ≤ 2048 token；`KV.GATHER` `broadcast=1` 写 4 份副本（非默认路径，8B 4:1 GQA 或多副本场景）→ 见 §6 与 §9 的跨切片标注。

### 5.5 汇总与最紧约束

| 用途 | 大小（INT8 默认） | 区域（字节区间） |
|------|-------------------|------------------|
| 权重 tile 双缓冲 | 1 MiB | 0x000000–0x0FFFFF |
| 激活区（hidden+Q+ctx+scores） | ≤ 2 MiB | 0x100000–0x2FFFFF |
| KV 窗口 | 1 MiB | 0x300000–0x3FFFFF |
| vector 工作区 | 2 MiB | 0x400000–0x5FFFFF |
| DMA 暂存/扩展 | 2 MiB | 0x600000–0x7FFFFF |
| **合计** | **8 MiB = 16 bank** | ✓ |

**最紧约束结论**：

1. **单对象最紧 = 权重 tile 双缓冲（BF16 形态 2 MiB）**：BF16 下权重区 2 MiB（0x000000–0x1FFFFF）挤压激活区至 1 MiB，
   与 prefill Q+ctx 需求冲突——这是 BF16 被压出默认 dtype 的**片上**原因（与 §7 的 HBM 容量结论一致）。
2. **多对象联合最紧 = prefill 激活驻留**：hidden（1 MiB）+ Q（1 MiB）+ ctx（1 MiB）+ scores tile（≤1 MiB）
   峰值约 3 MiB > 激活区 2 MiB，必须靠 **in-place 复用**（Q 覆盖 hidden、ctx 覆盖 Q）与 **N 维 tiling** 压缩。
3. **形状增长最紧 = attention scores**（$128 \times 32 \times \text{window}$，窗口 2048 时 = 16 MiB），
   必须把 window 维（N）tile 到 ≤128 列 → scores 驻留 ≤ 1 MiB。

**结论**：8 MiB / 16 bank 在 **INT8 默认下够用且有双缓冲余量**（激活/vector/DMA 共 6 MiB 弹性区）。
BF16 权重（2 MiB 双缓冲）是唯一把预算顶到边界的 dtype，与 §7 的 HBM 结论共同把 v0 部署 dtype 压向 INT8（默认）/INT4。

---

## 6. SRAM 地址映射表（bank 用途划分）

16B 字地址 `W[18:0]`，byte = word ×16。bank = `B[7:4]`（16B 粒度 16 路交错）；下表按**区域**（字节区间）划分，各区域与 bank 无一一对应（连续 256 B 跨 16 bank）。

| 区域 | 字范围 `[18:0]` | 字节范围 | 大小 | 用途（INT8 默认） |
|------|-----------------|----------|------|-------------------|
| 0 | `0x00000–0x07FFF` | `0x000000–0x07FFFF` | 512 KiB | 权重缓冲 A（tile ping-pong） |
| 1 | `0x08000–0x0FFFF` | `0x080000–0x0FFFFF` | 512 KiB | 权重缓冲 B（tile ping-pong） |
| 2 | `0x10000–0x17FFF` | `0x100000–0x17FFFF` | 512 KiB | 激活区 |
| 3 | `0x18000–0x1FFFF` | `0x180000–0x1FFFFF` | 512 KiB | 激活区 |
| 4 | `0x20000–0x27FFF` | `0x200000–0x27FFFF` | 512 KiB | 激活区 |
| 5 | `0x28000–0x2FFFF` | `0x280000–0x2FFFFF` | 512 KiB | 激活区 |
| 6 | `0x30000–0x37FFF` | `0x300000–0x37FFFF` | 512 KiB | **K 窗口**（KV staging） |
| 7 | `0x38000–0x3FFFF` | `0x380000–0x3FFFFF` | 512 KiB | **V 窗口**（KV staging） |
| 8 | `0x40000–0x47FFF` | `0x400000–0x47FFFF` | 512 KiB | vector 工作区 |
| 9 | `0x48000–0x4FFFF` | `0x480000–0x4FFFFF` | 512 KiB | vector 工作区 |
| 10 | `0x50000–0x57FFF` | `0x500000–0x57FFFF` | 512 KiB | vector 工作区 |
| 11 | `0x58000–0x5FFFF` | `0x580000–0x5FFFFF` | 512 KiB | vector 工作区 |
| 12 | `0x60000–0x67FFF` | `0x600000–0x67FFFF` | 512 KiB | DMA 暂存/扩展 |
| 13 | `0x68000–0x6FFFF` | `0x680000–0x6FFFFF` | 512 KiB | DMA 暂存/扩展 |
| 14 | `0x70000–0x77FFF` | `0x700000–0x77FFFF` | 512 KiB | DMA 暂存/扩展 |
| 15 | `0x78000–0x7FFFF` | `0x780000–0x7FFFFF` | 512 KiB | DMA 暂存/扩展 |

- 本表是 **v0 默认布局**（compiler 可调，P4/P6 调度决策），KV 区字节范围与 05-kv-cache §1.5 一致。
- **dtype 变体**：INT4 权重只需 512 KiB（区域 0），激活区可扩至 0x080000–0x2FFFFF；BF16 权重区 2 MiB（0x000000–0x1FFFFF），
  激活区缩至 0x200000–0x2FFFFF（见 §5.5，此为 BF16 非默认的片上原因）。
- 区域 8–11（vector）、12–15（DMA）为弹性区，也承接 `KV.GATHER` 广播副本溢出的情形（非默认路径，见 §9 标注）。

---

## 7. HBM 驻留布局与精度模式（BF16 约束声明）

> 权重总字节以 ModelGrounding 核验值为准：**BF16 权重 = 16,381,470,720 B = 15.26 GiB**（与官方 `total_size` 逐字节一致）。
> 每-128-group 量化 scale（BF16）：$\text{权重元素数}/128 \times 2$ B = $8{,}190{,}735{,}360/128 \times 2 = 127{,}980{,}240$ B ≈ **122.0 MiB**。

### 7.1 布局（load-time 静态划分）

```
HBM[0 .. 16 GiB):
  [0, W)                                    权重（含紧随其后的量化 scale 表）
  [align_up(W, ALIGN), ... + KV_REGION)     KV 区域（576 块 slab）
  剩余                                     reserved（golden 暂存 / 运行时 / 对齐 pad）
```

`ALIGN` = KV slab 对齐：8K slab = 2 MiB、4K slab = 1 MiB（slab 容量是 load-time 参数，见 §7.3）。

### 7.2 各精度模式驻留表

| 模式 | 权重 (GiB) | scale (MiB) | KV slab 容量 | KV 区域 (GiB) | context | 总占用 (GiB) | 余量 (GiB) | 状态 |
|------|-----------|-------------|--------------|---------------|---------|--------------|------------|------|
| BF16 | 15.256 | — | 4K（1 MiB） | 0.5625 | 4K | 15.819 | **0.181** | ✓ golden 参考（限 4K） |
| BF16 | 15.256 | — | 8K（2 MiB） | 1.125 | 8K | 16.381 | **−0.381** | ✗ 溢出 |
| INT8 | 7.628 | 122.0 | 8K（2 MiB） | 1.125 | 8K | 8.872 | 7.128 | ✓ 默认 |
| INT4 | 3.814 | 122.0 | 8K（2 MiB） | 1.125 | 8K | 5.058 | 10.942 | ✓ 激进 |

推导（关键行）：

- BF16@8K 溢出：$16{,}381{,}470{,}720 + 1{,}207{,}959{,}552 = 17{,}589{,}430{,}272$ B，
  超 16 GiB（$17{,}179{,}869{,}184$）达 **$409{,}561{,}088$ B = 0.381 GiB**。✗
- BF16@4K 余量：$17{,}179{,}869{,}184 - (16{,}381{,}470{,}720 + 603{,}979{,}776) = 194{,}418{,}688$ B = **185 MiB**。✓
- INT8@8K：$8{,}190{,}735{,}360 + 127{,}980{,}240 + 1{,}207{,}959{,}552 = 9{,}526{,}675{,}152$ B = 8.872 GiB，余 7.128 GiB。✓
- BF16 权重自身留空 = $17{,}179{,}869{,}184 - 16{,}381{,}470{,}720 = 798{,}398{,}464$ B = 0.744 GiB →
  BF16 模式 KV 硬上限 = $798{,}398{,}464 / 147{,}456$ B/token = **5,414 token**（非整块，取 4K 为干净配置）。

### 7.3 BF16 约束声明与张力解决

**约束**：BF16 权重（15.26 GiB）与 8K KV 区域（1.125 GiB）**无法共同驻留**（超 0.381 GiB）。

**张力解决（v0 策略）**：

1. **8K context 的正式模式 = INT8（默认）/ INT4（激进）权重**。契约 D5 的量化路径本就是 BF16→INT8→INT4，
   与 D8（放量顺序）一致；INT8 是工作默认，余量 7.13 GiB 可承载 >8K context、多 context 或未来分页 KV。
2. **BF16 模式 = golden/正确性参考，v0 限 4K context**：slab 容量 4K（1 MiB/slab）、KV 区域 576 MiB、
   总占用 15.82 GiB、余 185 MiB。此时 KV slab 容量由 load-time 参数选择 **4K**（见下）。
3. **BF16@8K 的解锁路径（backlog，非 v0）**：① KV 元素 INT8 量化（KV 区域减半至 0.5625 GiB →
   总 15.82 GiB ✓，需 KV-quant，05-kv-cache 已列 backlog）；② HBM 扩容 24/32 GiB。
   **v0 无第二存储层可溢出**——SRAM 仅 8 MiB，远小于 0.381 GiB 缺口，不构成溢出目标。

**KV slab 容量参数化（本切片的 load-time 参数）**：KV 地址公式（05-kv-cache §1.3）是 **8K 情形**（`pos` 13-bit、
slab stride $2^{21}$）。BF16@4K 用同构的 **4K 情形**（`pos` 12-bit、slab stride $2^{20}$、`AR_KV_BASE` 1 MiB 对齐）；
8K 解码第 8193 个 KV slot（pos=8192）用 **16K 情形**（slab stride $2^{22}$ = 4 MiB，`KV.LOAD` 13-bit `pos_start`
仍限 $[0,8192)$，当前 token 由 DC 程序 VMOV 尾 subtile 覆盖）：
地址生成器的 `pos` 宽度与 slab 步长由单寄存器 `SLAB_SHIFT ∈ {20, 21, 22}` 选择，结构不变。8K 公式为默认与契约 §8 的
设计上限，4K 为 BF16 模式的物理压缩、16K 为 8K 边界解码的容量扩展——此为对 05-kv-cache 的细化，冲突时报主会话。

---

## 8. 数据搬运动线示例

带宽口径：HBM sustained 读 720 / 写 240 B/cycle；SRAM 写 256 B/cycle；1 GHz。

### 8.1 prefill：DMA.LOAD 权重 tile 双缓冲载入（INT8）

**指令**（2D 编码，因 `RowBytes` ≤ 65535 故按行拆分）：

```
DMA.LOAD  SrcAR=AR_w(权重 HBM 基址, bit63=1)
          DstAR=AR_wbuf0(SRAM 权重缓冲 A, bit63=0, 字地址 0x00000)
          RowBytes=128  NumRows=4096  StrideC=C{128}  mode=1(2D)
```

| 项 | 值 |
|----|-----|
| 搬什么 | W_q 的一个权重 tile（K=4096 × N=128，INT8） |
| 字节 | $4096 \times 128 = 524{,}288$ B = 512 KiB |
| 瓶颈 | **SRAM 写受限**（LOAD 恒 SRAM 写受限，§3.3） |
| cycle 数 | $T_{\text{xfer}} = 524{,}288 / 256 = \mathbf{2048}$ cycles（HBM 读侧仅 $524288/720 = 728$ cycles；$T_{\text{first}} = 100$ cycles 被流水掩盖） |

> 该 tile 载入与 **另一权重缓冲区的计算重叠**（双缓冲，§4.2），稳态暴露延迟 ≈ 0。

### 8.2 decode：KV.LOAD KV 窗口载入（v0 冻结路径）

**指令**（tile = 2048 token，K/V 单副本，`sel=both`）：

```
KV.LOAD  dstK=AR_kbuf(SRAM, bit63=0)  dstV=AR_vbuf(SRAM, bit63=0)
         layer=L  head=h  sel=both(2)  pos_start=0  count=2048
```

| 项 | 值 |
|----|-----|
| 搬什么 | 1 个 KV head 的 K（或 V）窗口 2048 token = 512 KiB，单副本载入 KV staging（K/V 由 P7 内部广播总线在 GQA 组内共享） |
| HBM 读 | $2048 \times 256 = 524{,}288$ B = 512 KiB |
| SRAM 写 | $524{,}288$ B = 512 KiB（单副本） |
| 瓶颈 | **SRAM 写受限**（写 256 < 读 720） |
| cycle 数 | $T_{\text{xfer}} = 524{,}288 / 256 = \mathbf{2048}$ cycles（HBM 读侧仅 $524{,}288 / 720 = \mathbf{728}$ cycles） |

> **decode 主导搬移其实是权重流**：QKV 投影三条 `GEMV`（`bsrc=1`，B 从 HBM 直通阵列，不经 SRAM）流
> W_q 16 MiB + W_k 4 MiB + W_v 4 MiB = **24 MiB**，$24 \times 2^{20} / 720 = \mathbf{34{,}952}$ cycles ≈ 35.0 µs；
> 36 层 dense + lm_head INT8 逐 token 权重流 $7.57\times10^{9} / 720 = \mathbf{10{,}513{,}889}$ cycles ≈ 10.5 ms（decode 不重读 embedding，口径同 04 §2.4）。这印证 decode 是 **HBM 读受限**（P6 的 roofline 上限）。

## 9. 与契约一致性自查

| 契约项 | 本切片落实 |
|--------|-----------|
| §1 指令 128-bit、HBM 大地址经 AR 间接 | §4.1 DMA 操作数严格取 ISA 的 SrcAR/DstAR/RowBytes/NumRows/StrideC/mode，不新增 |
| §2 AR 64×64b（bit63 地址类型）/ C 32×32b | §4.1 源/目标地址类型由 AR bit63 判定；stride 取 C 寄存器 |
| §4 SRAM 16 bank、bank=addr[7:4]（16B 粒度 16 路交错） | §2.1 字地址换算 + 16 路交错；§2.3 冲突规则 |
| §5 HBM 16 GiB、40b、1.2 TB/s、64B 突发 | §3 容量/拆分/延迟/突发对齐 |
| §6 128×128 阵列、INT8 32.77 TMAC/s | §2.2 A/B 口带宽与阵列流式需求一致（128 元素/拍） |
| §7 MODE PF/DC | §5.3 decode GEMV 权重流按 16 lane×8 行拆解 |
| §8 KV 每 (layer,kv_head) slab、8K 容量、GQA stride 广播 | §6 KV 区、§7.2 驻留、§8.2 KV.LOAD 载入 |
| 数值 §10（hidden 4096/32 q_head/8 kv_head/128 dim） | §5 容量论证全部采用 |

**跨切片标注（待主会话/P3 收敛）**：

1. **KV.GATHER 广播 footprint 与 KV.LOAD 冻结**：`broadcast=1` 在 SRAM 写 4 份副本。K 副本限 KV staging K 区、V 副本限 V 区
   （K/V 分区），故 GATHER tile 上限 = $512\ \text{KiB}/(4 \times 256\ \text{B}) = 512$ token，
   与 05-kv-cache 原 GATHER `count ≤ 2048` 冲突（已同步 KvProtocol，其文件已改 2048→512）。
2. **KV slab 容量参数化**：8K（默认，`SLAB_SHIFT=21`）/ 4K（BF16 模式，`SLAB_SHIFT=20`）/ 16K（8K 边界解码，
   `SLAB_SHIFT=22`，4 MiB slab）为 load-time 参数（域 `{20,21,22}`），
   是 BF16@4K 能落在 16 GiB 内的必要条件（§7.3）。
3. **读/写拆分 3:1（900/300 GB/s）** 与 sustained 80%（720/240 GB/s）为 roofline 统一口径（P1/P6）。

**Backlog（不进 v0）**：分页 KV、KV INT8/INT4 量化、HBM 扩容、非 16B 对齐的标量访存、DMA 乱序/优先级 QoS。
