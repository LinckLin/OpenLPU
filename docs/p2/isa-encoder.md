# ISA 编码器用法（compiler/isa/）

`compiler/isa/isa.py` + `compiler/isa/qbin.py`：02-isa 的 128-bit 字段布局 assembler /
disassembler，00-container §2 的 qbin writer / reader。字段布局逐位转录自 02-isa §2/§4–§8，
`OPSPEC` 表即权威（`qsim/test_isa_fields.py` 对 02-isa 做字段级断言）。

## 128-bit 总布局（02-isa §2.1）

```
bit:  127──120 | 119──112 | 111 109 | 108 106 | 105 104 | 103────────────0
field: engine  |  opcode  |  srcA   |  srcB   |  acc    |  engine-specific (104b)
width:  8b     |   8b     |   3b    |   3b    |   2b    |  104b
```

## encode / decode

```python
from compiler.isa import isa as I

w = I.encode_inst("GEMM", srcA=I.DT_INT8, srcB=I.DT_INT8, acc=I.ACC_INT32,
                  ARa=0, ARb=1, ARc=2, M=128, N=128, K=1024, batch=1,
                  CA=0, CB=1, CC=2, CD=3, acc_init=1, bsrc=1, dequant=1,
                  transpose_A=0, transpose_B=1)
bytes16 = I.inst_to_bytes(w)          # 16B 小端
d = I.decode_inst(w)                  # -> 全字段 dict（含 engine_tag_valid 校验）
```

## asm 文本 assembler / disassembler

```
MODE PF
CONFIG AR0 = 0x8000000000100000        # class=1（AR，64b，bit63=1 → HBM）
CONFIG C3 = 0x00200000                 # class=0（C，低 32b）
GEMM ARa=0 ARb=26 ARc=10 M=128 N=128 K=1024 batch=1 CA=0 CB=1 CC=2 CD=6 acc_init=1 bsrc=1 dequant=1 transpose_B=1 srcA=INT8 srcB=INT8 acc=INT32
BARRIER
DMA.STORE SrcAR=2 DstAR=3 RowBytes=8192 NumRows=1 StrideC=5 mode=1 srcA=INT32
```

```python
insts = I.assemble(asm)               # list[Inst]
prog  = I.assemble_bytes(asm)         # raw 128-bit inst byte stream
print(I.disassemble_program(prog))    # 回读反汇编（round-trip 测试覆盖）
```

## qbin writer / reader（00-container §2）

```python
from compiler.isa.qbin import Tensor, write_qbin, read_qbin

t = Tensor(name="model.layers.0.self_attn.q_proj.weight",
           shape=[2048, 1024], dtype="INT8", hbm_off=0x100000,
           data=wq_int8.tobytes(),
           scales_hbm_off=0x800000, scales=cd_bf16.tobytes())
header = write_qbin("l.qbin", "Qwen3-0.6B", cfg, {"mode":"W8A8","group":128,"sym":True},
                    [t], pf_program, dc_program)
qb = read_qbin("l.qbin")              # 解析 magic/version/flags/header JSON/tensors/pf/dc
```

容器布局（小端、各 section 64B 对齐）：`magic "NLPU" | version 1 | flags | header_size |
header(u32 长度前缀+JSON) | .weights(文件偏移==hbm_off) | .pf_program | .dc_program | "ENDQ"+长度校验`。
header JSON：`model / cfg / quant / tensors[](name,shape,dtype,hbm_off,bytes,[scales_*]) / pf_entry / dc_entry`。

> 排障记录：bytearray 越界切片赋值（写远端 offset）会**追加**而非扩展到该 offset ——
> 已改为预分配零缓冲再按绝对偏移填充（qbin.py `write_qbin`）。

## lowering 入口

`compiler/lowering.py`：`lower_linear(x_shape, w_shape, mode, quant)` 产 asm（GEMM PF / GEMV DC，
N≤128 tiling、K 流式、M=seq≤128），`encode_program` 编码为字节，`build_linear_qbin` 一次产出
含 PF+DC 两段程序的最小合法 qbin。
