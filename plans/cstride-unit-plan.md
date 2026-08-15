# Cstride 单位一致性修复（微型提案 v1，待评审）

> 定位：冻结 spec 内部单位矛盾。02-isa §8.4 两处将 Cstride 标为「字节」（L454 表、L459 语义），
> 而 05-kv-cache §4.4 定义为「SRAM 字地址间距」（默认 16 字 = HEAD_DIM×2B/16B）；实现
> （qsim executor、qrun 程序）均按 SRAM 字地址执行（dst_i = dst + i×Cstride，dst 为 19b 字地址）。
> 05 与实现为语义真值，02 的「字节」为笔误。修复 = 规格一致性，非新范围。

## 修复

- 02-isa.md L454：「Q head 广播 stride（字节）」→「Q head 广播 stride（SRAM 字地址间距，与 05 §4.4 一致）」。
- 02-isa.md L459：「Q head 块字节 stride」→「Q head 块 stride（SRAM 字地址间距）」。

## 验收

- grep 02-isa/05-kv-cache 无 Cstride 单位矛盾；实现不变（纯文档）。
