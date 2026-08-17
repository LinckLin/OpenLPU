# B'-旋转器立项计划（提案 v3，round-2 否决后重写）

> 目标：兑现 B'（INT8-K QK-norm 折叠 + INT4-V）的 1049 tok/s 条件口径。
> round-1/2 否决要点已吸收。round-2 六条：①CD 位分配（18 新位 > 11 reserved）；②sink 行
> 绝对位置（[0,4) 独立基址）；③B-feed 时间必须入模型（余量仅 4.6%）；④k_norm 表驻留
> 决定隐藏成败；⑤LOAD 侧退役的专用测试迁移；⑥PF 模式 co-sim 缺口。

## 1. 架构（沿用 round-2 主路径：流式融合）

KV 以量化形式存窗；dequant + on-the-fly ROPE 融入 attention 的 B 操作数 feed（K 流
INT8→BF16→旋转→QK^T，V 流 INT4→BF16→PV），无 staged BF16 写、无新指令。`kv_quantdequant`
保留 APPEND 侧量化器；LOAD 侧 staged 去量化路径裁剪退役（≈627 口径如实记录后退役）。
- **降级路径（冻结 ISA 内既有机制）**：B-feed 时序无法收敛时回退 = 逐行 VECTOR ROPE
  （已落地无旋转器路径 ≈257 tok/s）或 staged ≈627 口径；ROPE.WINDOW 复合指令在冻结
  ISA（33 条）中不存在，不引用（如确需须另立项走契约修订）。
## 2. 数字链（修正后，全部对标冻结口径）

- **HBM 读（B' 档，W=2048, R=2052）**：每层 = weights 15,730,944 B + KV 2052×1568 B
  （K 1024 INT8 + V 512 INT4 + V-scale 32）= **26,317 cyc/层**；×28 + lm_head 216,087
  = 953K → **1e9/953K = 1049.3** ✓
- **SRAM 读**：K 1024 B/token + V 544 B/token。**k_norm 折叠表（2 KB/层）驻留 B-feed
  本地缓冲**（每层载入 8 cyc，摊销 ≈0）——若按行重读则 K 侧 24,624 cyc 隐藏失效
  （round-2 ④），故本地缓冲为硬件必选项。
- **B-feed 阵列占用（显式入模型，round-2 ③）**：K 旋转 2052×2048/256 = 16,416 +
  V 去量化 2052×1024/256 = 8,208 + dense MAC ≈480 = **≈25,104 cyc/层 ≤ 26,317**，
  余量 1,213 cyc（4.6%）；调度假设 = B-feed 与 dense 权重 HBM 流重叠（阵列时间与 HBM
  并行），模型显式计该假设。**计费口径（round-3 ③）**：K dequant 并入 rotate 系数
  预缩放（scale 与 cos/sin 预乘），按 2 ops/elem 计费；V dequant 按 1 op/elem。
- **指令数**：721,895 不变（旋转随描述符隐式执行）。

## 3. RTL 落地范围

1. **B 操作数 feed 增 dequant+rotate 级**：
   - K：INT8 per-channel 带符号折叠 scale（本地缓冲 k_norm 表）→ BF16 → ROPE（绝对
     位置）→ MAC；V：INT4 per-token scale → BF16 → MAC；
   - sink 行（4 个）**与窗口一致 INT8 折叠存储**（round-3 ①钉死：与冻结参照/门槛
     逐位对应、RTL 最简），同样在线旋转，绝对位置 0..3；
2. **CD/指令位分配（round-2 ①，逐位钉死）**：
   - KV_QUANT + ROTATE_K → **CD[31:30]**（02-isa §6.1 reserved 位改语义，02 §10 非零
     检查豁免此两位）；
   - pos_base[15:0] → **BMM 指令 reserved [20:5]**（复用 0..40959 绝对位置约定）；
   - **双段基址（round-2 ②）**：sink 段与窗口段各发独立 BMM（与现 windowed_kv.py 双
     KV.LOAD 结构一致）——sink 段 pos_base=0，窗口段 pos_base=pos−W+1；
   - 随件修订：02-isa §6.1 reserved→字段语义 + 全文 33 条计数不变（无新指令）。
3. **kv_quantdequant 裁剪**：LOAD 侧 staged 去量化退役（APPEND 侧保留）；scale 元数据
   路径与 B-feed 共用。
4. **时序模型修正**：fold_verify._KVInt4Demands / timing_p6 删 staged 写项、**显式加
   B-feed 项 ≈25,104**（含 K 旋转/V 去量化/dense 与 HBM 重叠假设）；复核 1049.3。

## 4. 验证

- 默认 BF16 回归逐周期不变（新位仅在新描述符下生效）；
- **B' 全链 co-sim（decode，R=2052，含 sink 行绝对位置 0..3 角度断言 + 负通道样本断言）**
  ≤1 ULP + trace；
- **PF 模式 B' attention co-sim 用例（round-2 ⑥）**：PF BMM B 路径（K 流式、batch 映射
  M 维）下 KV_QUANT=1 的 dequant+rotate 正确性（pos_base=0 全序列）；
- **测试迁移（round-2 ⑤）**：run_cosim_bprime.py 两 LOAD 用例改写为 B-feed 路径断言；
  test_bprime_kv.py dequant/roundtrip 用例改写或显式退役并记录 rotator-impl.md；
- qrun 门槛复跑（ΔPPL ≤2% + 交叉 ≥8/10）；qsim 时序模型复跑 1049.3；
- DC 关键路径复测（B-feed 新增级对 Fmax 的影响如实报告）。

## 5. 验收

- 流式融合三端落地 ≤1 ULP；修正后模型（显式 B-feed 项）复现 1049.3；门槛复跑过；
  默认回归 + 迁移后专用测试全绿；
- 报告 rotator-impl.md：双口径收尾（staged ≈627 退役、流式 1049.3 成主口径）、
  口径修正与 round-1/2 六条吸收记录、bprime-impl.md 同步订正。

## 6. 风险

| 风险 | 对策 |
|------|------|
| B-feed dequant+rotate 拉长关键路径（Fmax） | DC 复测；必要时流水 1 级（延迟 +1 cyc，吞吐不变，隐藏于 HBM） |
| 4.6% 余量被流水气泡侵蚀 | 双缓冲窗口行、B-feed 背靠背调度；实测后如实报告（不虚报 1049） |
| 折叠 scale 负通道（3.404%）流式路径遗漏 | co-sim 断言负通道集（与 fold-verify 同一集） |
| D16 单副本广播与每 head 独立角度 | 旋转在广播前按 head 完成，无冲突 |
| PF 模式 K 流与 DC 差异未覆盖 | §4 新增 PF co-sim 用例 |

## 7. 需评审关注点

1. round-2 六条吸收是否到位（逐条对账）；
2. CD[31:30]+指令[20:5] 位分配与 02 §6.1/§10 修订的充分性；
3. B-feed 项 25,104 ≤ 26,317 推导与重叠调度假设的可接受性；
4. 若一致，请声明「评审一致，可执行」。
