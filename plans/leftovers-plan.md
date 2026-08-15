# 早期裁决遗留三件收尾（提案 v2，回应评审第 1 轮 4 条意见）

> v2 变更：①L2 写清单补 qforge/cli.py（现对 --dtype bf16 exit 2）与 qrun/{engine.py,
> __main__.py,m4.py}（--weights-from-hf 穿透、m4.py 模板硬编码同步）；②验证口径固定
> BF16 容器输入（m4.py --only 1 --qbin <bf16.qbin>）+ loader 校验 flags dtype==BF16；
> ③run_all_acceptance 默认模式四证据串保留验收 + --full 用法声明；④L3 措辞引用 §3.4、
> 明示周期口径所计/所不计、量化 ≤0.05%（256 KiB）／<0.03%（512 KiB）、锚点不变。

## 任务

### L1（05 §4.4 Cstride 注，文档）
- 05-kv-cache.md §4.4 表下补一行注：count>1 时编译器须设 Cstride = count×16 字（默认 16 仅
  对 count=1；GATHER 为非默认路径，此注为规格完整性）。

### L2（BF16 权重入容器，代码 + 验证）
1. qforge/build.py + **qforge/cli.py**：BF16 模式产出含 141 张量权重的 tensors 表（BF16
   原样、64B 对齐，scales 段省略；flags bit[1:0]=0）；cli.py 移除 --dtype bf16 的拒绝分支。
2. qrun/weights.py + **qrun/engine.py + qrun/__main__.py**：BF16 路径默认从 qbin tensors 表
   装载；--weights-from-hf 显式选项穿透；**装载器校验容器 flags 的 dtype 编码，INT8 容器
   拒绝 bf16 请求（报错不静默）**。
3. **qrun/m4.py**：write_report 模板硬编码「BF16(safetensors) 权重」「BF16 权重不入容器」
   两处同步改为容器口径（保留四证据串 20/20、8/8、5/5、argmax 一致：**True**）。
4. 验证：`qforge compile --dtype bf16` 产出 → qbin round-trip 位级一致 →
   **`python3 qrun/m4.py --only 1 --qbin <bf16.qbin>` 20/20 复跑（默认走容器；
   safetensors 分支仅在 --weights-from-hf 下达）**；verify_m3 不破坏；
   **run_all_acceptance.sh 默认模式全绿（四证据串保留）**。
5. 文档：docs/p4/qforge.md 增 BF16 容器说明；m4-report 注记路径变更；
   **run_all_acceptance.sh --full 用法声明：全量重跑 m4.py 需 BF16 qbin（m4.py 默认 qbin
   逻辑同步或脚本注明 --full 用法变更，由实现者选择并写入 reproduction.md）**。

### L3（DMA 周期口径文档化，文档）
- docs/spec-src/03-memory.md §3.3 补一行「v0 周期模型简化声明」：DMA 周期口径按 §3.3 公式
  （T_first+T_xfer+T_drain）计费，不计（a）非对齐传输的 head/tail partial burst（§3.4，
  ≤2 burst/传输、对齐流为 0；（b）多 in-flight 传输的带宽重叠；二者量级 ≤0.05%
  （256 KiB tile，2×64 B/262144）／ <0.03%（512 KiB tile），被 80% sustained 余量覆盖，
  **roofline 锚点（720/240 GB/s）不变**；qsim timing 与 RTL co-sim 同口径；重标定另立计划。

## 分工与验收

| 项 | 代理 | 写目录 | 验收 |
|----|------|--------|------|
| L1+L3 | DocLeft | docs/spec-src/05-kv-cache.md、docs/spec-src/03-memory.md | 两注落地且与 §3.4/§4.4 上下文衔接 |
| L2 | ContBf16 | qforge/{build.py,cli.py}、qrun/{weights.py,engine.py,__main__.py,m4.py}、docs/p4/、docs/p5/ | BF16 qbin 自包含 + 20/20 复跑（--qbin 固定）+ flags 校验 + 四串保留 + 回归绿 |

两代理并行、无交叠。

## 风险

| 风险 | 对策 |
|------|------|
| L2 权重入容器增 qbin 体积（BF16 ≈1.2 GB） | 预期内；生成/装载 ~20s/次 |
| 删除 safetensors 直读破坏既有路径 | --weights-from-hf 显式选项保留，默认走容器 |
| --full 重跑 m4.py 数据源断裂 | 默认 qbin 逻辑同步 + reproduction.md 注明用法 |
| L3 注动摇 roofline 锚点 | 措辞量化 ≤0.05%（256 KiB）／<0.03%（512 KiB） 且明示锚点不变 |
