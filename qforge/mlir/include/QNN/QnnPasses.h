#ifndef QNN_QNNPASSES_H
#define QNN_QNNPASSES_H

#include "QNN/QnnToQisa.h"

namespace mlir {
namespace qnn {

inline void registerPasses() {
  // Pass is registered via PassRegistration in QnnToQisa.cpp.
}

} // namespace qnn
} // namespace mlir

#endif // QNN_QNNPASSES_H
