//===- QnnDialect.cpp - QNN layer dialect -----------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "QNN/QnnDialect.h"
#include "QNN/QnnOps.h"

using namespace mlir;

#include "QNN/QnnOpsDialect.cpp.inc"

namespace mlir {
namespace qnn {

void QNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "QNN/QnnOps.cpp.inc"
      >();
}

} // namespace qnn
} // namespace mlir
