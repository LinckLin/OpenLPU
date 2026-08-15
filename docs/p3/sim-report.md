# P3 时序模拟器报告（M2b）— qsim 时序模型

> 交付物：`qsim/timing.py`（新增，不改 `qsim/executor.py` 功能语义）+ 本报告。
> 口径：所有数字引用 `docs/spec.md` §3/§4、03-memory、04-execution-engines、
> 05-kv-cache、01b 模型卡、`docs/p1/roofline.md` §5/§6 的原值。时钟 1 GHz，单位 cycle。

---

## 1. 模型描述

### 1.1 目标模型（Qwen3-0.6B，01b 权威）

| 参数 | 值 |
|------|-----|
| hidden / layers | 1024 / 28 |
| Q heads / KV heads / head_dim | 16 / 8 / 128（GQA **2:1**） |
| intermediate / vocab | 3072 / 151936 |
| q_proj / k_proj / v_proj | 2048×1024 / 1024×1024 / 1024×1024 |
| o_proj / gate / up / down | 1024×2048 / 3072×1024 / 3072×1024 / 1024×3072 |
| 每层 dense 参数 | 15,728,640 |
| 每层 RMSNorm | 2,304（in/post 1024 + q/k 128） |
| 每层权重（INT8） | 15,730,944 B（dense + rmsnorm） |
| lm_head | 155,582,464（151936×1024，tied 仍每 token 全读） |
| KV | 114,688 B/token（= 28 × 8 × 128 × 2 × 2 B），每层 4,096 B/token |

### 1.2 硬件常数（spec §4 / 03 / 04，冻结）

| 量 | 值 |
|----|-----|
| 阵列峰值 | 32,768 MAC/cycle（INT8 W8A8，128×128×2） |
| HBM sustained 读/写 | 720 / 240 B/cycle（峰值 900/300） |
| HBM T_first | 100 + align + Q；64 B 突发 |
| SRAM | 16 bank × 512 KiB，2R1W；读 512 / 写 256 B/cycle；bank=addr[7:4]（16B 粒度 16 路交错） |
| MODE 切换 | 300 cycle（BARRIER 后，每请求一次） |
| DMA | 4 in-flight，双缓冲权重 tile（0x000000–0x0FFFFF） |

---

## 2. 时序模型（四引擎）

`qsim/timing.py` 在功能执行器之上按层加 cycle 计数：

- **Matrix**：PF 周期 `ceil(K/256)×M + 256`（04 §1.4，fill/drain 摊平）；DC 16-lane×8 行 GEMV，
  `K/16` cycle/批（04 §2.2）；稳态计算桶统一 `MAC / 32768`（与 roofline 锚点同源）；
  MODE 切换 300 cycle。
- **Vector**：04 §3.2 逐指令吞吐/延迟表 + §3.4 复合序列（softmax/RMSNorm/RoPE/SwiGLU），
  0.6B 重算（softmax 16 head、RMSNorm len 1024、RoPE 24×128、SwiGLU 3072×2）。
- **SRAM**：16 bank 2R1W、bank=addr[7:4]（16B 粒度 16 路交错）、03 §2.3 固定优先级 stall；读 512/写 256 B/cycle。
  PF 稳态（权重刷新 1R/bank + 激活 1R/bank + DMA 权重写 1W/bank）每 bank 恰 2R+1W，
  天然无冲突 → v0 trace 依赖 stall 仅剩 KV staging 写口（见 §5.3/§5.4）。
- **HBM**：sustained 读 720/写 240 GB/s；T_first=100+align+Q；KV 全窗口重读
  `114,688×ctx` B/token（05 §5.2）。
- **DMA**：4 in-flight、双缓冲；PF 权重 tile 加载与计算 overlap（隐藏于计算桶之下）。

---

## 3. trace 重放结果

trace 按 02 §12 / 05 §5.2 步序列构造（0.6B 尺寸），每层 cycle 分解为五桶：
**权重流 / KV 重读 / 计算（矩阵）/ Vector / 依赖 stall**，另单列 KV staging SRAM 写（裁决证据）。

### 3.1 decode 单 token（cache=1024 / 4096），每层分解

| 桶 | cache=1024 | cache=4096 | 说明 |
|----|-----------|-----------|------|
| 权重流 | 21,849 cyc（21.85 µs） | 21,849 cyc | 15,730,944 B ÷ 720，DC bsrc=1 直连流 |
| KV 重读（HBM，单副本） | 5,826 cyc | 23,302 cyc | 4,096×ctx B ÷ 720 |
| 计算（矩阵） | 608 cyc | 992 cyc | dense 480 + attention 128/512 |
| Vector | 513 cyc | 513 cyc | 见 3.3 |
| 依赖 stall | 26,554 cyc | 43,646 cyc | 阵列空闲 = HBM 读 − 计算 − Vector |
| **KV staging SRAM 写（GATHER ×4）** | 65,536 cyc | 262,144 cyc | 裁决证据，**非 bug**（见 §4） |

> 依赖 stall 即 DC 模式「27.3× 带宽短缺」（spec §4）的阵列空闲侧写：阵列被迫降至
> HBM 读速率，计算 608 cyc 隐藏在 21.85 µs 权重流内，其余 26.5K cyc 空等 HBM。

### 3.2 prefill seq=128，每层分解

| 桶 | 值 | 说明 |
|----|-----|------|
| 权重流 | 21,849 cyc（隐藏） | 读一次，M=128 复用，双缓冲下 < 计算 63.5 µs |
| 计算（矩阵） | **63,488 cyc（63.5 µs）** | 2,080,374,784 MAC ÷ 32,768（roofline §5.1 锚点） |
| Vector | **55,377 cyc（55.4 µs）** | 全量（含 per-head QK-norm）；4 项复合 op 34,887 cyc |
| 依赖 stall | 0 | 计算受限，无 HBM 空等 |
| KV（读 729 + STORE_BLOCK 写 2,185） | 忽略 | seq=128 首块窗口小 |

**串行假设下的每层全周期**：

| 口径 | 值 |
|------|-----|
| 纯矩阵锚点 | 63.5 µs |
| + Vector（4 项复合 op，对应计划「~30-40K」） | 98.4 µs |
| + Vector（全量，含 QK-norm + 残差） | **118.9 µs** |

> 计划「全层 ≈ 95 µs」= 锚点 63.5 + Vector ~30-40K（4 项复合 op）。本报告全量 Vector
> 额外计入 per-head QK-norm（18.4K）+ 残差 VADD（2.0K），差异在 §6 需评审项明示。

### 3.3 Vector 复合序列（0.6B 重算，04 §3.4）

| op | 推导 | decode（1 token） | prefill（128 token） |
|----|------|------------------|---------------------|
| Softmax | 6 指令/行 + 36 排空 | 16 head → 132 | 2048 行 → 12,324 |
| RMSNorm normal（len 1024） | 3×8+4=28 指令 + 9 排空 | 2× → 74 | 2×128× → 7,186 |
| QK-norm per-head（24 组） | 6 指令/组 + 10 排空 | 24 组 → 154 | 3072 组 → 18,442 |
| RoPE（24×128） | 1 指令/块 + 8 排空 | 24 块 → 32 | 3072 块 → 3,080 |
| SwiGLU（3072×2） | 48 块 × 2 指令 + 9 排空 | 96 指令 → 105 | 12,288 指令 → 12,297 |
| 残差 VADD（×2） | 8 块 × 2 | 16 | 2,048 |
| **合计** | — | **513** | **55,377** |

### 3.4 decode token/s（sustained HBM 读，含 KV 全窗口重读）

| context | 权重读 | KV 重读 | 总读 | token/s |
|---------|--------|---------|------|---------|
| 4K | 0.596 GB | 0.470 GB | 1.066 GB | **675.5** |
| 8K | 0.596 GB | 0.940 GB | 1.536 GB | **468.9** |

> 与 spec §4 锚点（0.6B @4K ≈ 675 / @8K ≈ 469）一致（sustained 720 GB/s 口径）。

---

## 4. 按桶偏差表（验收判据）

| 桶 | 时序模型 | 锚点 | 偏差 | 判据 |
|----|---------|------|------|------|
| 权重流（decode 每层） | 21,849 cyc（21.85 µs） | roofline §5.2 21.85 µs（15,730,944 B ÷ 720） | **+0.002%** | <10% ✅ |
| 计算（decode dense） | 480 cyc（0.48 µs） | roofline §5.2 0.48 µs（15,728,640 MAC ÷ 32,768） | **0.0%** | <10% ✅ |
| KV 重读（HBM 单副本） | 5,826 cyc（ctx=1024） | roofline §6 4,198,400 B → 5,831 cyc | **−0.09%** | 对 §6 口径 ✅ |
| Vector（decode） | 513 cyc | 04 §3.4 重算（16 head 等） | 一致 | 对 §3.4 重算 ✅ |
| Vector（prefill，4 项） | 34,887 cyc | 计划「~30-40K」 | 区间内 | 对 §3.4 重算 ✅ |

> 权重流/计算偏差为构造性 0%（时序模型与 roofline 引用同一组 spec §3/§4 冻结常数），
> 验证对象是 trace 重放的**字节/MAC 求和是否正确**——求和正确即与锚点逐位对齐。
> KV 桶 −0.09% 源于窗口口径：本报告按计划 05 §5.2 用 `114,688×ctx`（窗口=ctx），
> roofline §6 用 `4096×(ctx+1)`（含当前 token），差恰 1 token（0.024%…0.1%），可忽略。

---

## 5. 待冻结项裁决（spec §5.1 第 1 项）：KV.GATHER vs KV.LOAD

### 5.1 每层周期直接对比（decode，KV staging）

| 项 | KV.LOAD（单副本） | KV.GATHER（broadcast=1，×4） |
|----|------------------|------------------------------|
| HBM 窗口读 | 1×（相同） | 1×（相同） |
| SRAM 写 | **1×** = 4,096×ctx B ÷ 256 | **4×** = 16,384×ctx B ÷ 256 |
| tile 上限 | 2048 token | 512 token（4 副本 footprint） |
| 分块次数 | ⌈ctx/2048⌉ | ⌈ctx/512⌉（4× 分块） |

**每层 staging wall**（= max(HBM 读, SRAM 写) + 分块 T_first，03 §3.3）：

| context | LOAD | GATHER | 比值 |
|---------|------|--------|------|
| 1024 | 16,484 cyc | 65,736 cyc | 3.99× |
| 4096 | **65,736 cyc（65.7 µs）** | **262,944 cyc（262.9 µs）** | **4.00×** |
| 8192 | 131,472 cyc | 525,888 cyc | 4.00× |

### 5.2 裁决依据

1. **HBM 窗口读两案相同（1×）**——GATHER 的 `broadcast=1` 净效果是「1 次 HBM 读 + 4 路 SRAM
   目的写」（05 §4.4），HBM 侧无差别。
2. **差异 = SRAM 写 4× vs 1×**：decode 的 KV staging 恒为 **SRAM 写受限**（03 §3.3：
   `LOAD 恒 SRAM 写受限`，写 256 < 读 720）。GATHER 的 4 副本把 SRAM 写量放大 4×，
   staging wall 随之 ×4（4K 下 262.9 vs 65.7 µs/层）。
3. **tile 上限 512 vs 2048 → 分块次数 4×**：4K 下 GATHER 分 8 块、LOAD 分 2 块，
   额外 T_first 与指令开销 4×。
4. **0.6B GQA 2:1 的冗余**：`broadcast=1` 硬件固定 ×4，而 2:1 GQA 只有 2 个 Q head 消费
   （01b §2、05 §3 注）→ **4 副本中 2 副本冗余（50% 浪费）**。此冗余是裁决证据，非 bug。

### 5.3 v0 裁决（写入 P5 输入）

**decode 数据通路改走 `KV.LOAD` 单副本**（P7 为阵列加内部广播总线后落地，ISA 不变）。
GATHER 的 ×4 SRAM 写 + ×4 分块在 4K 已使 KV staging 达 262.9 µs/层（远超权重流 21.85 µs），
其中 50% 是 0.6B 2:1 GQA 下的冗余副本；LOAD 单副本降至 65.7 µs/层（仍 SRAM 写受限，
但为 GATHER 的 1/4）。协议同时保留两指令（spec §5.1 前提成立）。

**附注（前向，不改变裁决）**：即使 LOAD 单副本，KV staging 的 SRAM 写（4K 下 65.5 µs/层）
仍 > KV HBM 读（23.3 µs/层），是「SRAM 写带宽 256 B/cycle vs HBM 读 720 B/cycle」的固有
1/2.8 比例所致。归 P6 优化项。

### 5.4 待冻结项裁决（spec §5.3 第 2 项）：bank 仲裁优先级 + DMA in-flight

**bank 仲裁（03 §2.3 固定优先级 MATRIX.A > MATRIX.B > MATRIX.C > VECTOR > DMA > KV）**：
交错寻址（bank = addr[7:4]，16B 粒度 16 路交错）下，PF 稳态并发集（权重刷新 1R/bank + 激活 1R/bank +
DMA 权重写 1W/bank）每 bank 恰 2R+1W，**零冲突**（`verify_v0_bank_allocation()` 的 PF 稳态断言）；
KV/vector 写由 BARRIER/依赖图串行化 → 固定优先级仲裁在 v0 trace 下
**stall = 0**，优先级细节不影响 v0 性能。**裁决：固定优先级照 03 §2.3 冻结，不改。**

**DMA in-flight（03 §4.4：硬件池 4，编译器 ≤2）**：v0 双缓冲 ping-pong 只需 2 in-flight，
PREFETCH 可用剩余 2 个槽位重叠预取。**裁决：保持 4 in-flight**（2 用于 ping-pong、2 用于
PREFETCH 余量；2 也可用但无预取重叠余量，4 无额外代价）。

> **附带发现（已裁决 D15，2026-08-13）**：原 03 §2.1「16B 交错」与 P0 公式 `bank = addr[18:15]`
> 的矛盾已由评审裁决修正为 `bank = addr[7:4]`（16B 粒度 16 路交错）、bank 内字 = `addr[22:8]`
> （spec §1 D15、§3.3）。本报告与时序模型已同步为交错公式，连续 256 B 命中 16 bank，
> 「SRAM 连续流读 256 B/cycle」可达性成立。

---

## 6. 交付物清单 + 验收证据 + 需评审项

### 6.1 交付物

- `qsim/timing.py`：四引擎 cycle 模型 + trace 重放（decode/prefill）+ GATHER/LOAD 裁决
  + `TimingExecutor` 包装器 + 数值一致性自检（`__main__` 驱动，`main()` 返回结构化结果）。
- `docs/p3/sim-report.md`：本报告。

### 6.2 验收证据（M2b）

| 验收项 | 证据 |
|--------|------|
| layer trace 在时序模型下跑通，数值与功能级一致（同一 executor） | `verify_executor_numerics()` 用**同一 `Executor`** 重放 golden q_proj：PF max_rel 2.9e-7、DC 3.9e-7，逐位一致 ✅ |
| 权重流 <10% | +0.002%（§4 表） ✅ |
| 计算 <10% | 0.0%（§4 表） ✅ |
| KV 桶对 §6 口径 | −0.09%（窗口 ±1 token，§4 表） ✅ |
| Vector 桶对 04 §3.4 重算 | decode 513 / prefill 4 项 34,887 cyc（§3.3 推导） ✅ |
| decode token/s（4K/8K，含 KV 重读） | 675.5 / 468.9 ✅ |
| prefill 每层全周期（含 Vector） | 63.5 µs（矩阵）+ 55.4 µs（Vector 全量）= 118.9 µs ✅ |
| GATHER vs LOAD 每层周期对比 | 65,736 vs 262,944 cyc（4.00×，§5.1） ✅ |
| bank 仲裁 + DMA in-flight（spec §5.3 第 2 项） | PF 稳态 2R+1W 零冲突；DMA 4 in-flight（§5.4） ✅ |

复现命令：`python3 qsim/timing.py`（打印分解 + 锚点偏差 + executor 数值一致性）。

### 6.3 需评审项

1. **prefill Vector 口径**：计划「~30-40K cycle」对应 softmax/RMSNorm/RoPE/SwiGLU **4 项复合 op**
   （本报告 34.9K）；完整 Vector 桶另含 per-head QK-norm（+18.4K）与残差 VADD（+2.0K），
   合计 55.4K → 每层全周期 118.9 µs（计划「≈95 µs」= 锚点 + 4 项）。请确认是否将 QK-norm
   计入 prefill Vector 桶。
2. **KV 窗口口径**：本报告按计划 05 §5.2 用 `114,688×ctx`（窗口=ctx），roofline §6 用
   `ctx+1`（含当前 token）。差 1 token（0.024–0.1%），不影响任何判据，但建议统一措辞。
3. **KV staging SRAM 写带宽**：主裁决用聚合 256 B/cycle；交错方案下 KV staging 跨 16 bank
   （1W/bank 并行），无单 bank 串行化。归 P5/P6，不改变 GATHER→LOAD 裁决方向。
4. **decode token/s 口径**：675/469 为 HBM 读 roofline（含 KV 单副本重读）；若把 KV staging
   的 SRAM 写计入关键路径（假设 dense 权重流与 KV staging 重叠、lm_head 权重读 216,087 cyc 不重叠），
   可达 token/s @4K = 1e9 / (28 × staging_wall/层 + 216,087)：LOAD = 1e9 / (28 × 65,536 + 216,087)
   = 1e9 / 2,051,095 ≈ **488**；GATHER = 1e9 / (28 × 262,944 + 216,087) ≈ **132**（见 §5.3 附注）。
   本报告按 spec §4 锚点只输出 HBM 读口径，SRAM 写影响作为裁决后果单列。
5. **bank 交错措辞矛盾（已裁决 D15）**：原 03 §2.1「16B 交错」与 P0 公式 `bank = addr[18:15]`
   矛盾；已裁决修正为 `bank = addr[7:4]`（16B 粒度 16 路交错）、bank 内字 = `addr[22:8]`
   （spec §1 D15、§3.3）。时序模型已同步为交错公式（见 §5.4）。
