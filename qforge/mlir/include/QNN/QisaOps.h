//===- QisaOps.h - Q-ISA dialect ops ----------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef QNN_QISAOPS_H
#define QNN_QISAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#define GET_OP_CLASSES
#include "QNN/QisaOps.h.inc"

#endif // QNN_QISAOPS_H
