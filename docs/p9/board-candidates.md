# P9 前置：板卡候选核验与采购清单

> 依据：`plans/p8-p10-plan.md` §5.1（候选清单前置、立即启动采购）。
> 核验口径：**AMD 官方产品页为准**；本文件仅写板卡候选结论，不改动 rtl/、asic/、fpga/。
> 状态：候选规格核验完成，采购建议可执行。核验日期 2026-08-15。

## 0. 结论摘要（TL;DR）

| 项 | 结论 |
|----|------|
| 首选 | **ZCU104（XCZU7EV）**，AMD 官方 $1,678、到货周期 8 周；DDR4 2 GB（1.86 GiB）≥1.5 GiB 下限 ✅ |
| 首选备替（同级） | Alinx AXU7EV（同为 XCZU7EV，器件级 URAM/BRAM/DSP 与 ZCU104 一致；板级 DDR 需购前核验） |
| 备选 | **Kria KV260（K26/ZU5EV 级）**，AMD 官方 $249、到货周期 26 周（分销商多有现货）；SRAM 上限需注明 |
| 订正（两处容量数字） | ① ZU7EV「UltraRAM ≥4 MiB」**错** → 实际 27 Mb ≈ 3.38 MiB（<4 MiB）；② KV260「无 UltraRAM / SRAM 上限 ~0.6 MiB」**错** → 实际有 64 块 URAM（18 Mb ≈ 2.25 MiB），SRAM 上限 ≈ 2.9 MiB |

## 1. 候选规格核验（AMD 官方页）

### 1.1 首选：ZCU104（ZU7EV）

来源：AMD Zynq UltraScale+ MPSoC ZCU104 Evaluation Kit 官方产品页
<https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu104.html>

| 项 | 官方值 | 换算/判定 |
|----|--------|-----------|
| 器件 | XCZU7EV-2FFVC1156（Zynq UltraScale+ MPSoC EV，含视频编解码） | 首选目标器件 |
| 价格 | **$1,678.00**（MSRP） | 首选成本 |
| 料号 | EK-U1-ZCU104-G | 下单用 |
| 到货周期 | **8 周**（AMD Lead Time） | 快于 KV260 |
| System Logic Cells | 504 K | |
| DSP Slices | 1,728 | |
| **Memory（片上）** | **38 Mb**（= BRAM 11.0 Mb + UltraRAM 27.0 Mb） | 官方页直接给出 38 Mb |
| UltraRAM | **27.0 Mb** = 96 块 × 288 Kb = 3,456 KiB ≈ **3.38 MiB** | ← 订正①：**<4 MiB** |
| Block RAM | 11.0 Mb = 312 块 × 36 Kb = 1,404 KiB ≈ 1.37 MiB | |
| 板上 DDR4 | **2 GB（64-bit，Component）**；另带 DDR4 SODIMM 连接器（64-bit） | 2 GB = 1.86 GiB ≥ 1.5 GiB ✅ |
| 工具许可 | 附 **Vivado BASIC 一年免费许可券**；MIG（DDR 控制器 IP）No-Charge | 移植成本最低 |
| 供货渠道 | Avnet / DigiKey / Mouser / Newark / Farnell（AMD 授权分销） | 见 §4 链接 |

> UltraRAM 块 288 Kb、BRAM 块 36 Kb 为 UltraScale+ 架构标准（UG1075/DS925）。
> 38 Mb 总数与「BRAM 11 Mb + UltraRAM 27 Mb」自洽；器件级拆分以 AMD Zynq UltraScale+ MPSoC
> 产品选择表（DS891 概览 / UG1075）为准。

### 1.2 首选备替（同级）：Alinx AXU7EV

来源：Alinx 官方（<https://www.alinx.com>，ZU7EV 系列）。**板级清单受站内反爬限制未逐行抓取，
下列板级项（DDR 容量、板载外设、单价）需购前以 Alinx 官方规格书核验。**

| 项 | 值 | 说明 |
|----|----|------|
| 器件 | XCZU7EV（与 ZCU104 同款） | 器件级 URAM/BRAM/DSP/Logic 与 ZCU104 **完全一致** |
| UltraRAM / BRAM | 27.0 Mb / 11.0 Mb（同 §1.1） | 与 ZCU104 相同 |
| 板上 DDR4 | 典型 2–4 GB（**购前核验**） | 同 ZU7EV，无容量争议 |
| 价格区间 | 约 $1,100–1,700（**购前核验**，通常低于 ZCU104） | 预算敏感时优选 |
| 供货渠道 | Alinx 官方商城 / 淘宝・速卖通旗舰店 | 国内发货，周期短 |

> 结论：Alinx AXU7EV 与 ZCU104 在「可容纳缩编 SRAM」这一关键维度等价（同为 38 Mb 片上内存），
> 可作为成本替代；但缺 Vivado 许可券与 AMD 官方开发套件生态，P9 立项时按预算二选一。

### 1.3 备选：Kria KV260（K26 SoM = ZU5EV 级）

来源：AMD Kria KV260 Vision AI Starter Kit 官方产品页
<https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html>

| 项 | 官方值 | 换算/判定 |
|----|--------|-----------|
| 形态 | K26 SoM + 载体板 + 散热（整包出货，SK-KV260-G） | 无需另购 SoM |
| 价格 | **$249.00**（MSRP）；电源适配器另售 $25 | 电源单卖，采购需一并 |
| 到货周期 | **26 周**（AMD Lead Time）；分销商（DigiKey/Mouser）常有现货 | 下单前确认库存 |
| System Logic Cells | 256 K | |
| DSP Slices | 1.2 K（1,248） | |
| Block RAM Blocks | **144**（×36 Kb = 5,184 Kb ≈ 5.1 Mb ≈ 0.63 MiB） | 计划「BRAM 5.1 Mb」✅ |
| **UltraRAM Blocks** | **64**（×288 Kb = 18,432 Kb = 18 Mb ≈ **2.25 MiB**） | ← 订正②：**非「无 UltraRAM」** |
| DDR Memory | **4 GB**（4×512M×16，non-ECC） | 计划「4 GB」✅ |
| 片上 SRAM 合计 | BRAM 0.63 MiB + URAM 2.25 MiB ≈ **2.88 MiB** | ← 订正②：上限 ≈ 2.9 MiB，非 0.6 MiB |
| 供货渠道 | Avnet / DigiKey / EBV / Excelpoint / Mouser / Silica | 见 §4 链接 |

## 2. 订正记录（计划中两处容量数字）

| # | 计划原文（§5.1） | 官方实际 | 订正 |
|---|------------------|----------|------|
| ① | 首选 ZU7EV「UltraRAM **≥4 MiB** 可容纳缩编 SRAM」 | ZU7EV UltraRAM = **27 Mb ≈ 3.38 MiB**（96 块 × 288 Kb） | 「≥4 MiB」→ **≈3.38 MiB**（<4 MiB）；片上内存 = URAM 3.38 + BRAM 1.37 ≈ **4.75 MiB** |
| ② | 备选 KV260「**无 UltraRAM**、BRAM 5.1 Mb → SRAM 上限 **~0.6 MiB**」 | KV260/K26（ZU5EV 级）**有 64 块 URAM**（18 Mb ≈ 2.25 MiB） | 「无 URAM / 上限 0.6 MiB」→ **URAM 2.25 MiB + BRAM 0.63 MiB ≈ 2.9 MiB** |

> 订正后果（对 P9 的影响）：
> - 首选片上 SRAM 预算 ≈ **4.75 MiB**（URAM 3.38 + BRAM 1.37），较计划「≥4 MiB」更宽松但**仍 < 8 MiB**，缩编比例 ≈ 59%。
> - 备选片上 SRAM 预算 ≈ **2.88 MiB**，较计划「0.6 MiB」显著放宽（约 4.8×），缩编比例 ≈ 36%，「需更大缩编」的风险相应下降但仍存在。

## 3. 缩编 SRAM 预算（8 MiB → 板卡片上）

QCore 全量 Scratchpad SRAM = **8 MiB**（spec §4，16 bank × 512 KiB；D3/D15）。

| 板卡 | UltraRAM | BRAM | 片上合计 | 占 8 MiB | 备注 |
|------|----------|------|----------|----------|------|
| ZCU104 / Alinx（ZU7EV） | 3.38 MiB | 1.37 MiB | **≈4.75 MiB** | 59% | 首选；纯 URAM 容纳 3.38 MiB |
| KV260（ZU5EV 级） | 2.25 MiB | 0.63 MiB | **≈2.88 MiB** | 36% | 备选；缩编更激进 |

- **DDR 下限**：首选 2 GB = 1.86 GiB ≥ 1.5 GiB ✅；备选 4 GB ✅。
- **DDR 带宽**（P9 立项冻结，此处仅记板卡重标定口径）：ZCU104 DDR4 64-bit、KV260 DDR4 64-bit，
  均以板卡 DDR 控制器 IP（MIG）重标定带宽/突发，D3 已声明「FPGA 以 DDR 替代 HBM」。

## 4. 采购建议清单（可执行）

| 优先级 | 板卡 | 料号 | 官方价（MSRP） | 首选渠道（链接） | 到货周期估计 |
|--------|------|------|----------------|------------------|--------------|
| **首选** | AMD ZCU104 | EK-U1-ZCU104-G | **$1,678** | [AMD 官方](https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu104.html) · [DigiKey](https://www.digikey.com/en/products/detail/xilinx-inc/EK-U1-ZCU104-G/9380242) · [Mouser](https://www.mouser.com/ProductDetail/Xilinx/EK-U1-ZCU104-G?qs=unwgFEO1A6sDpUW4RdXfqg%3D%3D) · [Avnet](https://www.avnet.com/americas/product/xilinx/ek-u1-zcu104-g/evolve-36524543/) · [Farnell](https://uk.farnell.com/xilinx/ek-u1-zcu104-g/eval-board-cortex-a53-cortex-r5/dp/3225208) | **8 周**（AMD Lead Time；分销商库存可能更快） |
| 首选备替 | Alinx AXU7EV | （购前核验） | ≈$1,100–1,700 | [Alinx 官方](https://www.alinx.com)（ZU7EV 系列） | 国内发货 1–4 周（购前确认） |
| **备选** | Kria KV260 | SK-KV260-G（+电源适配器 $25） | **$249 + $25** | [AMD 官方](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html) · [DigiKey](http://www.digikey.com/product-detail/en/xilinx-inc/SK-KV260-G/122-SK-KV260-G-ND/13985269) · [Mouser](https://www.mouser.com/ProductDetail/Xilinx/SK-KV260-G?qs=DRkmTr78QATF92lTPoHh8Q%3D%3D) · [Avnet](https://www.avnet.com/shop/us/products/xilinx/sk-kv260-g-3074457345645974711) | AMD 26 周；分销商常有现货（下单前确认库存） |

### 建议执行动作（供 P9 立项）

1. **立即下单 ZCU104**（首选）：8 周到货周期长，与 P8/P10 并行启动；含 Vivado BASIC 一年许可，
   移植工具链零额外成本。
2. **KV260 仅作预算/备选**：$249 便宜但 SRAM 上限 ≈2.9 MiB（缩编 36%）、AMD 直采周期 26 周；
   若选它，须在 P9 立项冻结「更激进缩编 + DDR 带宽重标定」口径，并**一并采购 $25 电源适配器**。
3. **Alinx AXU7EV 作为 ZCU104 的成本替选**：器件级与 ZCU104 等价（38 Mb 片上内存），
   下单前核验板级 DDR 容量与供货渠道。

## 5. 风险与备注

| 风险 | 对策 |
|------|------|
| 首选到货 8 周、备选 AMD 直采 26 周 | 采购已前置并行；分销商库存优先；不可得则 P9 挂起不阻塞 P8/P10（计划 §7） |
| 8 MiB SRAM 无法整量落片 | 均需缩编：首选 ≈59%、备选 ≈36%（§3）；SRAM 缩编口径在 P9 立项冻结 |
| Alinx 板级规格未逐行核验 | 已在 §1.2 显式标注「购前核验」项，未将未核验数字写入结论 |
| KV260 电源单卖 | 采购清单已含 $25 适配器，避免到货无法上电 |

### 引用来源（AMD 官方）

- ZCU104：<https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu104.html>（价格/料号/Lead Time/器件/Logic 504K/Memory 38Mb/DSP 1728/DDR4 2GB）
- KV260：<https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html>（价格/料号/Lead Time/Logic 256K/BRAM 144/UltraRAM 64/DSP 1.2K/DDR 4GB/电源另售）
- 器件资源拆分（URAM 288 Kb/块、BRAM 36 Kb/块，ZU7EV = 96 URAM + 312 BRAM）：AMD Zynq UltraScale+ MPSoC 产品选择表（DS891 概览 / UG1075）。
