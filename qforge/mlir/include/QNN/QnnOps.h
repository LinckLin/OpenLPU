//===- QnnOps.h - QNN layer dialect ops -------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef QNN_QNNOPS_H
#define QNN_QNNOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "QNN/QnnOps.h.inc"

#endif // QNN_QNNOPS_H
