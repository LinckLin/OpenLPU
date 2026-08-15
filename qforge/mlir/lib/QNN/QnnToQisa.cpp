//===- QnnToQisa.cpp - QNN -> Q-ISA lowering pass ----------------*- C++ -*-===//
//
// Mechanism proof for the linear chain: qnn.matmul -> qisa.gemm (PF) / qisa.gemv
// (DC), qnn.rmsnorm -> qisa.rmsnorm, plus the mode/config/dma/barrier
// scaffolding. The full-model GEMM/GEMV sequence is produced by the Python
// lowering (qforge/lowering.py); this pass demonstrates the same lowering
// mechanism in MLIR. Memory-plan constants and the CD descriptor mirror
// compiler/lowering.py (read-only) / qforge/lowering.py.
//
//===----------------------------------------------------------------------===//

#include "QNN/QnnToQisa.h"

#include "QNN/QnnDialect.h"
#include "QNN/QnnOps.h"
#include "QNN/QisaDialect.h"
#include "QNN/QisaOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::qnn;

namespace {

// -- memory plan / ISA constants (mirror qforge/lowering.py) ---------------
static constexpr uint64_t HBM_BIT = 1ULL << 63;
static constexpr uint64_t WQ_HBM = 0x0010'0000;
static constexpr uint64_t SCALES_HBM = 0x0080'0000;
static constexpr uint64_t INPUT_HBM = 0x0100'0000;
static constexpr uint64_t OUTPUT_HBM = 0x0200'0000;

static constexpr uint32_t AR_X_SRAM = 0, AR_X_HBM = 1, AR_OUT_SRAM = 2,
    AR_OUT_HBM = 3, AR_WQ_HBM = 4, AR_SCALE_SRAM = 5, AR_SCALE_HBM = 6,
    AR_TILE_C = 10, AR_TILE_B = 26;
static constexpr uint32_t C_CA = 0, C_CB = 1, C_CC = 2, C_CD = 6,
    C_DMA_X = 4, C_DMA_OUT = 5;

static constexpr uint32_t DT_BF16 = 0, DT_INT8 = 2, DT_INT32 = 4;
static constexpr uint32_t ACC_INT32 = 0, ACC_FP32 = 1;

static uint64_t align16(uint64_t x) { return (x + 15) / 16 * 16; }
static uint64_t sramWord(uint64_t byte) { return byte / 16; }

struct QnnToQisaPass
    : public PassWrapper<QnnToQisaPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "qnn-to-qisa"; }
  StringRef getDescription() const override {
    return "Lower qnn.matmul/rmsnorm (linear chain) to qisa.gemm/gemv/rmsnorm";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<qnn::QNNDialect, qisa::QISADialect>();
  }


  void runOnOperation() override {
    func::FuncOp f = getOperation();

    // collect linear-chain qnn ops in source order
    SmallVector<Operation *> qnnOps;
    f.walk([&](Operation *op) {
      if (isa<MatmulOp, RMSNormOp>(op))
        qnnOps.push_back(op);
    });
    if (qnnOps.empty())
      return;

    // emit qisa instruction stream at the head of the function body
    OpBuilder b(&f.getBody().front(), f.getBody().front().begin());
    for (Operation *op : qnnOps) {
      if (auto mm = dyn_cast<MatmulOp>(op))
        lowerMatmul(b, mm);
      else if (auto rn = dyn_cast<RMSNormOp>(op))
        lowerRmsnorm(b, rn);
    }

    // erase the qnn ops in reverse order (linear chain: each result is consumed
    // by the following op, so reverse erasure leaves no dangling uses)
    for (Operation *op : llvm::reverse(qnnOps))
      op->erase();
  }

  void lowerMatmul(OpBuilder &b, MatmulOp mm) {
    auto xTy = cast<RankedTensorType>(mm.getX().getType());
    auto wTy = cast<RankedTensorType>(mm.getWeight().getType());
    int64_t M = xTy.getShape()[0];
    int64_t K = xTy.getShape()[1];
    int64_t N = wTy.getShape()[0];
    uint32_t mode = (M == 1) ? 1 : 0;   // 0=PF, 1=DC

    int64_t eszA = 1, eszB = 1, eszOut = 4;
    int64_t xBytes = M * K;
    int64_t outTileBytes = M * 128 * eszOut;
    int64_t G = K / 128;
    int64_t scaleTileBytes = 128 * G * 2;
    int64_t xSram = 0;
    int64_t outTileSram = align16(xBytes);
    int64_t scaleSram = align16(outTileSram + outTileBytes);
    int64_t ntiles = N / 128;

    Location loc = mm.getLoc();
    b.create<qisa::ModeOp>(loc, mode);
    b.create<qisa::ConfigOp>(loc, AR_X_SRAM, 1, sramWord(xSram));
    b.create<qisa::ConfigOp>(loc, AR_X_HBM, 1, HBM_BIT | INPUT_HBM);
    b.create<qisa::ConfigOp>(loc, AR_OUT_SRAM, 1, sramWord(outTileSram));
    b.create<qisa::ConfigOp>(loc, AR_OUT_HBM, 1, HBM_BIT | OUTPUT_HBM);
    b.create<qisa::ConfigOp>(loc, AR_WQ_HBM, 1, HBM_BIT | WQ_HBM);
    b.create<qisa::ConfigOp>(loc, AR_SCALE_SRAM, 1, sramWord(scaleSram));
    b.create<qisa::ConfigOp>(loc, AR_SCALE_HBM, 1, HBM_BIT | SCALES_HBM);
    b.create<qisa::ConfigOp>(loc, AR_TILE_C, 1, sramWord(outTileSram));
    b.create<qisa::ConfigOp>(loc, C_CA, 0, K * eszA);
    b.create<qisa::ConfigOp>(loc, C_CB, 0, K * eszB);
    b.create<qisa::ConfigOp>(loc, C_CC, 0, 128 * eszOut);
    b.create<qisa::ConfigOp>(loc, C_CD, 0, (1 << 20) | (0 << 19) | sramWord(scaleSram));
    b.create<qisa::ConfigOp>(loc, C_DMA_X, 0, K * eszA);
    b.create<qisa::ConfigOp>(loc, C_DMA_OUT, 0, 128 * eszOut);

    b.create<qisa::DmaLoadOp>(loc, AR_X_HBM, AR_X_SRAM, K * eszA, M, C_DMA_X,
                              1, DT_INT8);
    for (int64_t t = 0; t < ntiles; ++t) {
      b.create<qisa::ConfigOp>(loc, AR_TILE_B, 1,
                               HBM_BIT | (WQ_HBM + t * 128 * K * eszB));
      b.create<qisa::ConfigOp>(loc, AR_SCALE_HBM, 1,
                               HBM_BIT | (SCALES_HBM + t * scaleTileBytes));
      b.create<qisa::ConfigOp>(loc, AR_OUT_HBM, 1,
                               HBM_BIT | (OUTPUT_HBM + t * outTileBytes));
      b.create<qisa::DmaLoadOp>(loc, AR_SCALE_HBM, AR_SCALE_SRAM, scaleTileBytes,
                                1, 0, 0, DT_BF16);
      if (mode == 0)
        b.create<qisa::GemmOp>(loc, AR_X_SRAM, AR_TILE_B, AR_TILE_C, M, 128, K,
                               1, C_CA, C_CB, C_CC, C_CD, 1, 1, 1, 0, 1,
                               DT_INT8, DT_INT8, ACC_INT32);
      else
        b.create<qisa::GemvOp>(loc, AR_X_SRAM, AR_TILE_B, AR_TILE_C, M, 128, K,
                               1, C_CA, C_CB, C_CC, C_CD, 1, 1, 1, 0, 1,
                               DT_INT8, DT_INT8, ACC_INT32);
      b.create<qisa::BarrierOp>(loc);
      b.create<qisa::DmaStoreOp>(loc, AR_OUT_SRAM, AR_OUT_HBM, 128 * eszOut, M,
                                 C_DMA_OUT, 1, DT_INT32);
    }
    b.create<qisa::BarrierOp>(loc);
  }

  void lowerRmsnorm(OpBuilder &b, RMSNormOp rn) {
    auto xTy = cast<RankedTensorType>(rn.getX().getType());
    int64_t L = xTy.getShape().back();
    Location loc = rn.getLoc();
    // VECTOR RMSNORM: ARa = x_sram (0), len = L, acc=FP32, src_a=BF16
    b.create<qisa::RmsnormOp>(loc, 0, 0, 0, L, 0, 0, DT_BF16, DT_BF16, ACC_FP32);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::qnn::createQnnToQisaPass() {
  return std::make_unique<QnnToQisaPass>();
}

void mlir::qnn::registerQnnPasses() {
  PassRegistration<QnnToQisaPass>();
}
