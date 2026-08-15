//===- QisaDialect.cpp - Q-ISA dialect --------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "QNN/QisaDialect.h"
#include "QNN/QisaOps.h"

using namespace mlir;

#include "QNN/QisaOpsDialect.cpp.inc"

namespace mlir {
namespace qisa {

void QISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "QNN/QisaOps.cpp.inc"
      >();
}

} // namespace qisa
} // namespace mlir
