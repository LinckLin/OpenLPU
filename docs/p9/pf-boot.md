# P9b：PF 引导定义冻结提案（qsim 侧 PF + KV 导入板卡）

> 状态：冻结提案（本文件构成 `porting.md` §5.5 冻结项 5 的预裁决）。
> 依据：`plans/m8-wait-plan.md` §3 P9b、`plans/p8-p10-plan.md` §5.2（「PF 引导」二选一）。
> 只写 `docs/p9/`，不改 `rtl/`/`fpga/`/`qsim/` 器件逻辑。
> 结论：**PF 由 qsim 侧执行，KV + 权重导入板卡内存**（选项 A，板载真跑排除）；导入通道 =
> **ZCU104 PS DDR 共享**（PS 预写、QCore 经 AXI HP 口直读，零额外通道，非 MIG）；**4K 起步**。

## 0. 结论摘要（TL;DR）

| 项 | 冻结结论 |
|----|----------|
| PF 执行位置 | **qsim 侧**（复用 qrun 宿主：tokenizer/采样/embedding 注入/权重装载）；QCore 板卡只跑 decode |
| KV 生成 | qsim PF 产出整上下文 KV cache，**离线导出**（与 qsim/executor 同程序、同精度口径） |
| 导入内容 | 权重（INT8，0.6 GB）+ KV cache（0.44 GiB@4K / 0.875 GiB@8K） |
| 导入通道 | **ZCU104 PS DDR 共享**：PS 侧预写权重+KV 入 PS DDR，QCore 经 **AXI HP 口**（PS DDR 控制器）直读 |
| 通道性质 | **零额外通道**：权重/KV 与 QCore 运行时数据同驻 PS DDR，PS DDR 控制器为硬化 IP（**非 MIG**） |
| 备选通道 | PS-DMA、PCIe、SD 卡（见 §3，均降级为备选） |
| 演示上下文 | **4K 起步**（KV 0.44 GiB，总导入 0.44–0.64 s；8K 为后续） |
| 与周期模型 | 导入通道带宽 = P9 立项 §5.3「DDR 带宽重标定」的输入；`T_FIRST` 口径不变（不双计） |

## 1. 冻结定义（选项 A：qsim 侧 PF + KV 导入）

```
  qsim 侧（宿主，复用 qrun）
  ┌─────────────────────────────┐
  │ tokenizer / 采样 / embedding │
  │ PF 执行（0.6B INT8 全上下文） │──> 产出 KV cache（bf16，8 head × 2 × 128 × 2B/token/layer）
  └──────────────┬──────────────┘
                 │ 权重(0.6GB INT8) + KV(0.44/0.875 GiB) 批量导出
                 ▼
  ZCU104 PS DDR（2 GB，DDR4-2400 64-bit，硬化 DDRC）
  ┌─────────────────────────────┐
  │ PS（Cortex-A53）预写权重+KV  │◀── SD / 网络 / 宿主传输（PS 侧，非 QCore）
  │ QCore 运行时数据（KV 窗口）   │
  └──────────────┬──────────────┘
                 │ QCore 经 AXI HP 口（128-bit @150/300 MHz）直读
                 ▼
  QCore（PL：command_processor + ddr_if + dma_kv_stage）
       decode 逐 token 生成（权重流 + KV 窗口 GATHER）
```

- **PF 不在板上跑**：0.6B 全上下文 PF 在 qsim 侧（宿主）执行，复用 qrun 的宿主闭环
  （`plans/p8-p10-plan.md` §5.2），产出 KV cache 后**批量导入板卡**。
- **QCore 板卡只跑 decode**：权重（PF/DC 共用）+ KV 窗口按层 GATHER 进片上 SRAM（缩编后），
  复用冻结周期模型 `KV.GATHER = T_FIRST + sram_write_cycles(...)`。

## 2. 导入通道：ZCU104 PS DDR 共享（非 MIG）

| 抽象 | 映射 | 说明 |
|------|------|------|
| 权重/KV 存储 | **PS DDR**（ZCU104 2 GB DDR4，硬化 PS DDR 控制器 DDRC） | 与 QCore 运行时 KV 窗口同驻，零额外通道 |
| 预写路径 | **PS 侧**（Cortex-A53）把权重+KV 写入 PS DDR | 数据源：SD/网络/宿主传输（见 §3 备选），写带宽 = PS DDR 带宽，非 QCore 瓶颈 |
| QCore 读路径 | **AXI HP 口**（PL→PS DDRC 直读） | ZU7EV 4 个 HP 口，每个 128-bit @150/300 MHz；`ddr_if` 的 AXI4 窄主机口接 HP 口 |
| 非 MIG | PS DDR 控制器为**硬化 IP 直连**（DDRC），**不是** PL 侧 MIG | 省去 MIG 配置/时钟域/IP 许可；QCore 的 DDR 窗口 = PS DDR 的一段地址空间 |

- **为什么非 MIG**：`porting.md` §4 首选方案的 DDR4 MIG（PL 侧）用于**板卡独立 DDR**；本提案改走
  **PS DDR 共享**——PS DDR 控制器是 Zynq MPSoC 的硬化外设，PL 经 HP 口直接读写，无需 MIG，
  也无需额外 DDR 通道（这正是「零额外通道」的含义）。板卡独立 DDR（MIG）退为备选（见 §7 风险）。

## 3. 备选通道（均降级，记入风险）

| 通道 | 带宽量级 | 为何降级 |
|------|----------|----------|
| **UART 寄存器流** | ~11.5 KB/s（115200 baud） | 0.6 GB 需 ~14 小时，**不可行**（`m8-wait-plan.md` 已排除） |
| PS-DMA | ~PS DDR 带宽（DMA 引擎搬移） | 与 HP 直读等价但多一层 PS 软件；作为 PS 预写的实现细节备选 |
| PCIe | 数 GB/s | ZCU104 无原生 PCIe 根，需外加；宿主闭环复杂度高，后续升级项 |
| SD 卡 | 数十 MB/s（读） | 仅作权重/KV 的**持久化载体**（PS 启动时读入 PS DDR），非运行时通道 |

## 4. 数据规模（0.6B INT8，Qwen3-0.6B）

来源：`rtl/ref/qsim_baseline/timing.py` `ModelCfg`（冻结模型卡，权威口径）。

| 项 | 公式 | 数值 |
|----|------|------|
| 权重/layer | `weight_int8_per_layer` | 15,730,944 B |
| 权重合计 | `weight_int8_per_layer × 28 + lm_head_params` | **596,048,896 B = 0.596 GB = 0.555 GiB**（口径「0.6 GB」） |
| KV/ token/ layer | `8 head × 2 × 128 × 2 B` | 4,096 B = 4 KiB |
| KV/ token（28 layer） | 4,096 × 28 | 114,688 B = 112 KiB |
| KV @ 4K | 114,688 × 4,096 | **469,762,048 B = 0.470 GB = 0.4375 GiB ≈ 0.44 GiB** |
| KV @ 8K | 114,688 × 8,192 | **939,524,096 B = 0.940 GB = 0.875 GiB** |

## 5. 带宽预算表（可执行）

通道带宽 `B`（GB/s）为**参数**；ZCU104 首选的 QCore 读路径 = AXI HP 口：

- HP 口 **128-bit @ 150 MHz**（FPD 典型）→ `B = 2.4 GB/s`；
- HP 口 **128-bit @ 300 MHz**（-2 速度等级上限）→ `B = 4.8 GB/s`；
- 参考上限：PS DDR4-2400 64-bit 峰值 19.2 GB/s（DDRC 硬化带宽，非单 HP 口可达）。

导入时间 `T = 载荷(GB) / B(GB/s)`：

| 载荷 | 大小 | @2.4 GB/s | @4.8 GB/s |
|------|------|-----------|-----------|
| 权重 | 0.596 GB | 0.248 s | 0.124 s |
| KV @ 4K | 0.470 GB | 0.196 s | 0.098 s |
| KV @ 8K | 0.940 GB | 0.392 s | 0.196 s |
| **合计 @ 4K** | 1.066 GB | **0.444 s** | **0.222 s** |
| **合计 @ 8K** | 1.536 GB | **0.640 s** | **0.320 s** |

- **4K 起步**：4K 总导入 0.44–0.64 s（含权重），远低于一次 decode 会话的开销；8K 导入翻倍仍 < 1 s，
  故演示从 4K 起步、8K 为后续。PS 侧预写时间（数据源带宽）单列于 §3，不计入 QCore 导入时间。
- **可持续带宽 ≠ 峰值**：表中 `B` 为单 HP 口**名义**带宽；真实可持续带宽受 PS DDRC 仲裁/刷新/效率影响，
  在 P9 立项 §5.3「DDR 带宽重标定」用板卡实测回填 `HBM_READ_BPC/HBM_WRITE_BPC`（本表只给预算量级，不虚构数字）。

### 5.1 可执行校验（复制即跑）

```python
# 带宽预算可执行校验：与 §4/§5 表逐项一致
WEIGHT = 596_048_896                    # 0.6B INT8 权重（B）
KV_PER_TOK_LAYER = 4096                 # 8x2x128x2 B
KV = lambda ctx: KV_PER_TOK_LAYER * 28 * ctx
def T(payload_B, B_GBs):
    return payload_B / (B_GBs * 1e9)     # -> 秒
for ctx, name in ((4096, "4K"), (8192, "8K")):
    kv = KV(ctx)
    tot = WEIGHT + kv
    print(f"KV@{name}: {kv/2**30:.4f} GiB  total: {tot/1e9:.3f} GB")
    for B in (2.4, 4.8):
        print(f"  @{B} GB/s -> {T(tot, B)*1000:.0f} ms")
```

预期输出（与本表一致）：KV@4K 0.4375 GiB / total 1.066 GB → 444 ms / 222 ms；KV@8K 0.875 GiB /
total 1.536 GB → 640 ms / 320 ms。

## 6. 与冻结周期模型 / dma_kv_stage 的关系

1. **T_FIRST 口径不变**：QCore 读 KV 窗口仍按 `KV.GATHER = T_FIRST + sram_write_cycles(...)` 计费；
   导入通道的**一次导入**是板卡上电一次性行为，不计入逐 token decode 周期模型。
2. **DDR 带宽重标定（§5.3）以本表为输入**：HP 口名义带宽 → 实测可持续带宽 → 回填
   `HBM_READ_BPC/HBM_WRITE_BPC`（qsim timing.py FPGA 口径），**不动 RTL 器件逻辑**。
3. **dma_kv_stage 吸收读延迟（§5.6 预裁决）**：decode 期间 QCore 从 PS DDR 读 KV/权重窗口，走
   `fpga/dma_kv_stage.sv`（挂接 ddr_if 引擎字节口内侧，多 outstanding + 重排序缓冲，在已计费
   `T_FIRST` 时点内交付，不双计）——见 `porting.md` §5.6 预裁决注记。

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| PS DDR 与 QCore 运行时争带宽（导入后 decode 期间同驻） | 导入是一次性上电行为，decode 开始前完成；§5.3 重标定按实测分摊 |
| 单 HP 口 2.4 GB/s 不足（未来大上下文） | 升 300 MHz（4.8 GB/s）或多 HP 口；仍不足则回退板卡独立 DDR4 MIG（`porting.md` §4） |
| PS DDR 容量 2 GB vs 8K KV 0.875 GiB + 权重 0.555 GiB + 运行时 | 8K 合计 ~1.43 GiB < 2 GB ✅；但需预留 qbin/激活/工作区，立项时按 §5.3 冻结容量预算 |
| 权重/KV 导入的宿主侧数据源未冻结 | SD/网络/宿主传输在 P9 立项冻结（本提案只冻结「PS DDR 共享 + HP 直读」通道） |
| 8K 起步过度 | 4K 起步（§5），8K 为后续里程碑，风险后置 |

## 8. 复现命令

```bash
# 带宽预算可执行校验（§5.1）
python3 - <<'PY'
WEIGHT = 596_048_896
KV = lambda ctx: 4096 * 28 * ctx
def T(b, B): return b / (B * 1e9)
for ctx, n in ((4096, "4K"), (8192, "8K")):
    kv, tot = KV(ctx), WEIGHT + KV(ctx)
    print(n, f"KV={kv/2**30:.4f}GiB", f"total={tot/1e9:.3f}GB",
          " ".join(f"@{B}={T(tot,B)*1000:.0f}ms" for B in (2.4, 4.8)))
PY
```
