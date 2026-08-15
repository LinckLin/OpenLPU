//===- qisa.td - Q-ISA dialect (1:1 with the 33-instruction Q-ISA) -------===//
//
// QCore dialect 2/3: qisa — the accelerated IR. Each op maps 1:1 to one of the
// frozen 33 Q-ISA instructions (docs/spec-src/02-isa.md §3). Encoding is via
// compiler/isa/isa.py; this dialect is the IR representation consumed by the
// qisa -> Q-ISA asm lowering (compiler/lowering.py).
//
// Groups: SYS 5 (mode/config/barrier/wait/nop), DMA 3 (load/store/prefetch),
// MATRIX 3 (gemm/gemv/bmm), VECTOR 18, KV 4.
//
// Operand references: AR (address register 0..63) and C (config 0..31) are
// encoded as 6-bit / 5-bit indices; the executor resolves them via CONFIG.
//
//===----------------------------------------------------------------------===//
#ifndef QISA_TD
#define QISA_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def QISA_Dialect : Dialect {
  let name = "qisa";
  let cppNamespace = "::mlir::qisa";
  let summary = "Q-ISA dialect: 1:1 with the 33-instruction Tensor Command ISA";
}

class QISA_Op<string mnemonic, list<Trait> traits = []>
    : Op<QISA_Dialect, mnemonic, traits>;

//=== SYS (5) ================================================================//

def QISA_ModeOp : QISA_Op<"mode"> {
  let summary = "MODE: dual-mode switch (PF=00 / DC=01)";
  let arguments = (ins I32Attr:$mode);   // 0=PF, 1=DC
  let assemblyFormat = "$mode attr-dict";
}

def QISA_ConfigOp : QISA_Op<"config"> {
  let summary = "CONFIG: write AR (class=1) or C (class=0) register";
  let arguments = (ins
    I32Attr:$reg,        // 0..63
    I32Attr:$reg_class,  // 0=C, 1=AR (renamed from "class": C++ keyword)
    I64Attr:$imm         // IMM64 (C: low 32b)
  );
  let assemblyFormat = "$reg `,` $reg_class `,` $imm attr-dict";
}

def QISA_BarrierOp : QISA_Op<"barrier"> {
  let summary = "BARRIER: drain all engines";
  let assemblyFormat = "attr-dict";
}

def QISA_WaitOp : QISA_Op<"wait"> {
  let summary = "WAIT: block until masked engine queues drain";
  let arguments = (ins I32Attr:$eng_mask);  // bit0 DMA, bit1 MATRIX, bit2 VECTOR, bit3 KV
  let assemblyFormat = "$eng_mask attr-dict";
}

def QISA_NopOp : QISA_Op<"nop"> {
  let summary = "NOP";
  let assemblyFormat = "attr-dict";
}

//=== DMA (3) ================================================================//

class QISA_DMAOp<string mnemonic> : QISA_Op<mnemonic> {
  let arguments = (ins
    I32Attr:$src_ar, I32Attr:$dst_ar,
    I32Attr:$row_bytes, I32Attr:$num_rows,
    I32Attr:$stride_c, I32Attr:$mode,   // 0=1D, 1=2D
    I32Attr:$src_a                     // transfer dtype (3-bit)
  );
  let assemblyFormat = "$src_ar `,` $dst_ar `,` $row_bytes `,` $num_rows `,` "
      "$stride_c `,` $mode `,` $src_a attr-dict";
}
def QISA_DmaLoadOp : QISA_DMAOp<"dma.load"> {
  let summary = "DMA.LOAD: HBM -> SRAM (1D/2D)";
}
def QISA_DmaStoreOp : QISA_DMAOp<"dma.store"> {
  let summary = "DMA.STORE: SRAM -> HBM (1D/2D)";
}
def QISA_DmaPrefetchOp : QISA_DMAOp<"dma.prefetch"> {
  let summary = "DMA.PREFETCH: HBM -> SRAM non-blocking";
}

//=== MATRIX (3) =============================================================//

class QISA_MatmulOp<string mnemonic> : QISA_Op<mnemonic> {
  let summary = "MATRIX op (shared operand layout, 02-isa §6.1)";
  let arguments = (ins
    I32Attr:$ara, I32Attr:$arb, I32Attr:$arc,
    I32Attr:$m, I32Attr:$n, I32Attr:$k, I32Attr:$batch,
    I32Attr:$ca, I32Attr:$cb, I32Attr:$cc, I32Attr:$cd,
    I32Attr:$acc_init, I32Attr:$bsrc, I32Attr:$dequant,
    I32Attr:$transpose_a, I32Attr:$transpose_b,
    I32Attr:$src_a, I32Attr:$src_b, I32Attr:$acc
  );
  let assemblyFormat = "$ara `,` $arb `,` $arc `,` $m `,` $n `,` $k `,` "
      "$batch `,` $ca `,` $cb `,` $cc `,` $cd `,` $acc_init `,` $bsrc `,` "
      "$dequant `,` $transpose_a `,` $transpose_b `,` $src_a `,` $src_b `,` "
      "$acc attr-dict";
}
def QISA_GemmOp : QISA_MatmulOp<"gemm"> {
  let summary = "GEMM: C[M,N] += A[M,K] x B[K,N]";
}
def QISA_GemvOp : QISA_MatmulOp<"gemv"> {
  let summary = "GEMV: M=1 (decode)";
}
def QISA_BmmOp : QISA_MatmulOp<"bmm"> {
  let summary = "BMM: batched (attention multi-head), batch<=16";
}

//=== VECTOR (18) ============================================================//

// VECTOR shared operand layout: ARa/ARb/ARd, len, CV, imm (02-isa §7.1).
class QISA_VectorOp<string mnemonic> : QISA_Op<mnemonic> {
  let arguments = (ins
    I32Attr:$ara, I32Attr:$arb, I32Attr:$ard,
    I32Attr:$len, I32Attr:$cv, I32Attr:$imm,
    I32Attr:$src_a, I32Attr:$src_b, I32Attr:$acc
  );
  let assemblyFormat = "$ara `,` $arb `,` $ard `,` $len `,` $cv `,` $imm `,` "
      "$src_a `,` $src_b `,` $acc attr-dict";
}
def QISA_VaddOp : QISA_VectorOp<"vadd"> { let summary = "VADD"; }
def QISA_VsubOp : QISA_VectorOp<"vsub"> { let summary = "VSUB"; }
def QISA_VmulOp : QISA_VectorOp<"vmul"> { let summary = "VMUL"; }
def QISA_VdivOp : QISA_VectorOp<"vdiv"> { let summary = "VDIV"; }
def QISA_VrecipOp : QISA_VectorOp<"vrecip"> { let summary = "VRECIP"; }
def QISA_VexpOp : QISA_VectorOp<"vexp"> { let summary = "VEXP"; }
def QISA_VrsqrtOp : QISA_VectorOp<"vrsqrt"> { let summary = "VRSQRT"; }
def QISA_VsiluOp : QISA_VectorOp<"vsilu"> { let summary = "VSILU"; }
def QISA_VmaxOp : QISA_VectorOp<"vmax"> { let summary = "VMAX"; }
def QISA_VmovOp : QISA_VectorOp<"vmov"> { let summary = "VMOV"; }
def QISA_VscaleOp : QISA_VectorOp<"vscale"> { let summary = "VSCALE"; }
def QISA_VmaskOp : QISA_VectorOp<"vmask"> { let summary = "VMASK (causal mask tile)"; }
def QISA_VreduceSumOp : QISA_VectorOp<"vreduce_sum"> { let summary = "VREDUCE_SUM"; }
def QISA_VreduceMaxOp : QISA_VectorOp<"vreduce_max"> { let summary = "VREDUCE_MAX"; }
def QISA_RopeOp : QISA_VectorOp<"rope"> { let summary = "ROPE (single position)"; }
def QISA_RmsnormOp : QISA_VectorOp<"rmsnorm"> { let summary = "RMSNORM"; }
def QISA_QuantOp : QISA_VectorOp<"quant"> { let summary = "QUANT"; }
def QISA_DequantOp : QISA_VectorOp<"dequant"> { let summary = "DEQUANT"; }

//=== KV (4) =================================================================//

def QISA_KvAppendOp : QISA_Op<"kv.append"> {
  let summary = "KV.APPEND: write 1-token K/V at C_KV_POS";
  let arguments = (ins I32Attr:$src_k, I32Attr:$src_v, I32Attr:$layer, I32Attr:$head);
  let assemblyFormat = "$src_k `,` $src_v `,` $layer `,` $head attr-dict";
}

def QISA_KvStoreBlockOp : QISA_Op<"kv.store_block"> {
  let summary = "KV.STORE_BLOCK: write K/V block (prefill)";
  let arguments = (ins
    I32Attr:$src_k, I32Attr:$src_v, I32Attr:$layer, I32Attr:$head,
    I32Attr:$pos_start, I32Attr:$count
  );
  let assemblyFormat = "$src_k `,` $src_v `,` $layer `,` $head `,` $pos_start "
      "`,` $count attr-dict";
}

def QISA_KvLoadOp : QISA_Op<"kv.load"> {
  let summary = "KV.LOAD: read K/V to SRAM";
  let arguments = (ins
    I32Attr:$dst_k, I32Attr:$dst_v, I32Attr:$layer, I32Attr:$head,
    I32Attr:$sel, I32Attr:$pos_start, I32Attr:$count
  );
  let assemblyFormat = "$dst_k `,` $dst_v `,` $layer `,` $head `,` $sel `,` "
      "$pos_start `,` $count attr-dict";
}

def QISA_KvGatherOp : QISA_Op<"kv.gather"> {
  let summary = "KV.GATHER: collect + GQA broadcast";
  let arguments = (ins
    I32Attr:$dst, I32Attr:$dst2, I32Attr:$layer, I32Attr:$head,
    I32Attr:$sel, I32Attr:$broadcast, I32Attr:$pos_start, I32Attr:$count,
    I32Attr:$cstride
  );
  let assemblyFormat = "$dst `,` $dst2 `,` $layer `,` $head `,` $sel `,` "
      "$broadcast `,` $pos_start `,` $count `,` $cstride attr-dict";
}

#endif // QISA_TD
