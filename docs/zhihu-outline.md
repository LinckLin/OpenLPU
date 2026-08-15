# QCore 系列知乎文章大纲（5 篇）

> 系列定位：一个人如何从零做出一套「下载真实 Qwen → 自己设计的编译器 → 自己设计的机器指令
> → 自己写的 SystemVerilog 处理器真正执行 → 生成正确自然语言」的完整 LLM 推理加速平台。
> 全链路开源（Apache-2.0），体系完整度对标 Tenstorrent，不吹绝对性能、只讲可复现。

---

## 第 1 篇：先立 Flag——我一个人要做一颗能跑大模型的「芯片」

**一句话定位**：为什么 LLM 推理加速器值得个人级重做一遍，以及一套「编译器 + ISA + 运行时 +
RTL」全栈的完整地图。

1. **为什么是「个人级开源加速平台」**：商业 NPU 闭源、SDK 黑盒；个人级全栈开源的稀缺性；
   对标 Tenstorrent TT-Forge 的分层思想（compiler/ISA/runtime/RTL 独立设计）。
2. **目标与边界（先说不做什么）**：不训练、不多卡、不做 70B/MoE/多模态；验证模型 Qwen3-0.6B，
   架构按 8B 设计；证明的是「体系完整度」而非「我会写 GEMM 加速器」。
3. **一张全链路地图**：HF/PyTorch → qforge → Q-MLIR → Q-ISA → qsim → qrun/QMetal →
   QCore RTL → FPGA → ASIC；三级 Golden Reference（PyTorch=qsim=RTL）总验证原则。
4. **里程碑进度与「不粉饰」**：M0–M7、M9 已闭环，M8（FPGA 上板）阻塞于板卡采购；
   1 GHz 未收敛（直接映射 Fmax 59 MHz）、INT8 交叉一致率 2/10 如实记录。
5. **这个系列会讲什么**：五篇的阅读路线 + 每篇「看完能复现什么」。

---

## 第 2 篇：从 Hugging Face 到机器指令——手写一个 LLM 编译器

**一句话定位**：qforge 如何把 Qwen3-0.6B 编译成一个 579 MiB 的 .qbin 容器。

1. **Q-ISA 与 .qbin 容器**：33 条 Tensor Command ISA（128-bit 编码）、00-container 布局
   （magic/version/flags/header/tensors/PF/DC 程序/ENDQ）；为何不选 LLVM backend。
2. **编译前端六步**：config 解析（0.6B 模型卡逐项核验）→ 建图（141 投影）→ safetensors 加载
   → W8A8 量化（对称 per-128-K-group）→ tiling（N≤128 流式）→ 调度；QKV 融合。
3. **量化与误差**：per-128-group 部分和 INT32 逐位 bit-exact、dequant fp32 <1e-6；
   六类投影 × PF/DC 的量化误差实测；激活 scale 折叠与运行时校准的边界。
4. **Q-MLIR 方言与 lowering**：qnn → qisa pass（linear 链 0 条 qnn 残留）、
   与上游 TableGen 的字段冲突处理；为什么「linear = matmul」。

---

## 第 3 篇：用模拟器算清一笔账——qsim 时序模型与 roofline

**一句话定位**：不写 RTL 之前，如何用周期级模型回答「这芯片到底快不快」。

1. **功能级执行器**：只靠 Q-ISA 完整执行 Qwen 的 executor；与 PyTorch 逐 op/逐 token 对齐。
2. **四引擎时序模型**：Matrix/Vector/SRAM/HBM 的周期口径；decode 五桶分解
   （权重流 / KV 重读 / 计算 / Vector / 依赖 stall）；「27.3× 带宽短缺、阵列利用率 3.66%」。
3. **roofline 与 token/s 天花板**：sustained 720 GB/s 口径；0.6B @4K ≈ 675 / @8K ≈ 469；
   KV 全窗口重读是 decode 的隐藏成本（8K 下 = 权重读的 158%）。
4. **KV staging 的裁决**：KV.LOAD vs KV.GATHER（4 副本冗余 50%）逐层周期对比；
   双缓冲 + DMA PREFETCH 的消融（serial → overlap → double_buffer → kv_resident）。
5. **M5 验收数字**：write-roofline 80%（4K 481 tok/s、8K 255 tok/s，PASS）。

---

## 第 4 篇：让 SystemVerilog 真的跑起来——RTL 与三级 golden

**一句话定位**：一颗能通过「PyTorch = qsim = RTL」三向对拍的 QCore，长什么样。

1. **QCore 微架构**：Command Processor + Matrix（128×128 双 MAC，PF/DC 双模）+ Vector（128-lane）
   + DMA + KV 地址生成 + 16 bank × 512 KiB SRAM；冻结周期模型与 Python 基线 1:1。
2. **软浮点与数值语义**：BF16/FP32 逐 op RNE 落盘；RoPE 的 sin/cos LUT + Cody-Waite 归约；
   fp32 为冻结语义（fp64 才是偏差）。
3. **三级 golden 会师**：15 op 实例 + 逐层 hidden 三向 ULP 表；attn_softmax「221 ULP → 0」的
   排查故事（测试台 harness 的 `bd_en` 悬置，非引擎缺陷）——这是最值得读的一节。
4. **co-sim 与性能**：全尺寸 16-tile PF/DC × BF16/INT8 全绿；单层 golden 14/14；
   Verilator 4.038 的接口坑与「整数组非阻塞拷贝编译爆炸」的规避。
5. **端到端边界的诚实声明**：全模型 E2E 为什么归 P9，传递闭包（M4 + M8）如何闭合三级一致。

---

## 第 5 篇：上板、综合、开源——把「玩具」变成一件作品

**一句话定位**：FPGA 原型、ASIC 流程、板卡选型，以及把整个项目开源出来的最后一公里。

1. **板卡无关接口层与 FPGA 原型**：clock_reset/host_if/ddr_if 三模块 + 集成 smoke（4 用例 ALL
   PASS）；「CP 不改、只在 ddr_if 侧吸收延迟」的立项冻结清单；ZCU104 vs KV260 板卡选型与
   片上 SRAM 缩编账（8 MiB → 4.75 MiB ≈ 59%）。
2. **ASIC 流程如实报告**：Yosys 0.44 + SkyWater 130 nm 真综合；8 项 FP 基元 / 两种 MAC 门数；
   面积（SRAM 占 40–64%）；多 corner STA；1 GHz 未收敛的「三重原因」与流水化方向。
3. **数字都不好看，但为什么还值得写**：59 MHz 也照样把 token/s、功耗账算清；「位精确功能 +
   时序模型」与「物理数据通路」的区分；不虚构数字的写作原则。
4. **开源与发布**：Apache-2.0、README 组件索引表、从零复现指南；「下载即复现」的工程洁癖。
5. **下一步与邀请**：缩编 SRAM 参数化、流水化物理层、板卡到位后的 P9 上板；欢迎一起把
   QCore 推到真实 DDR 上跑出 token。
