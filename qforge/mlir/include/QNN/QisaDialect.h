//===- QisaDialect.h - Q-ISA dialect ----------------------------*- C++ -*-===//
//
// Q-ISA dialect: 1:1 with the 33-instruction Tensor Command ISA (attribute-only
// ops — no SSA operands, matching the straight-line instruction stream).
// TableGen: QisaOps.td (mirrors compiler/mlir/qisa.td, read-only upstream).
//
//===----------------------------------------------------------------------===//

#ifndef QNN_QISADIALECT_H
#define QNN_QISADIALECT_H

#include "mlir/IR/Dialect.h"

#include "QNN/QisaOpsDialect.h.inc"

#endif // QNN_QISADIALECT_H
