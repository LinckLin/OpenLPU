# 00 — 命令流与 qbin 容器 (v0)

> 本切片由主会话起草。定义编译产物 `.qbin` 的静态布局与命令流执行模型，
> 是 P4 (HF 前端) 与 P3 (模拟器) 的共同接口。

## 1. 命令流执行模型

- Command Processor (CP) 逐条取指执行 ISA 指令流（128-bit 定长）。
- 程序入口先由 `CONFIG` 批量写 AR/C 寄存器（基址、stride、维度、KV 位置）。
- 四个引擎（Matrix/Vector/DMA/KV）各有一条发射队列，CP 按序发射；
  `WAIT <engine>` 阻塞后续发射直到该引擎队列排空；`BARRIER` 等待全部引擎。
- 双模：`MODE PF` / `MODE DC` 是全局开关，切换时需 `BARRIER` 在前（切换代价见
  04-execution-engines）。
- 无程序计数器跳转、无标量分支——命令流是直线序列，控制只经 BARRIER/WAIT。

## 2. `.qbin` 布局 (v0)

所有数值小端序，各 section 64B 对齐。

| 偏移 | 字段 | 大小 | 内容 |
|------|------|------|------|
| 0 | magic | 4B | `"NLPU"` |
| 4 | version | u32 | `1` |
| 8 | flags | u32 | bits[1:0] 默认 dtype 编码（0=BF16，2=INT8，3=INT4）；bit2 双模启用；bits[31:3] reserved |
| 12 | header_size | u32 | header 区总字节（含 JSON 长度前缀） |
| 16 | header | variable | u32 长度前缀 + JSON |
| … | .weights | variable | 量化打包后的权重，64B 对齐 |
| … | .pf_program | variable | prefill ISA 程序（指令序列原始字节） |
| … | .dc_program | variable | decode ISA 程序 |
| … | .end | 4B | `"ENDQ"` 哨兵 + 文件长度校验 |

header JSON 字段：

```json
{
  "model": "Qwen3-8B",
  "cfg": { "hidden": 4096, "layers": 36, "q_heads": 32, "kv_heads": 8,
           "head_dim": 128, "intermediate": 12288, "vocab": 151936,
           "rope_theta": 1000000.0, "rms_eps": 1e-6, "qk_norm": true,
           "max_pos": 40960 },
  "quant": { "mode": "W8A8" | "W4A16" | "BF16", "group": 128, "sym": true },
  "tensors": [ { "name": "layers.0.self_attn.qkv_proj.weight",
                 "shape": [6144, 4096], "dtype": "INT8",
                 "hbm_off": 65536, "bytes": 25165824,
                 "scales_hbm_off": 25231360, "scale_bytes": 393216 } ],
  "pf_entry": <offset of .pf_program>, "dc_entry": <offset of .dc_program>
}
```

- `tensors` 表按 HBM 地址升序；`scales_*` 为 per-128-group 量化 scale（无量化时省略）。
- 权重在 HBM 的最终地址由加载器以 `hbm_off` 为准写入 AR 寄存器；
  qbin 文件内偏移 = `hbm_off`（约定相同，简化 P4/P3）。

## 3. 与 ISA/内存系统的关系

- 权重打包规则（INT4 两两打包、INT8 连续、BF16 原样）见 04 与 P4；
  本容器只规定 64B 对齐与 HBM 偏移表，不规定内存布局字节序之外的打包细节。
- 命令流内 DMA 指令的 HBM 目标地址必须落在 `tensors` 表的
  `[hbm_off, hbm_off+bytes)` 区间或 KV 区，加载器在装载时校验。
- KV cache 区独立于 `.qbin`，由运行时按 05-kv-cache 的 slab 公式在 HBM 高段分配。
