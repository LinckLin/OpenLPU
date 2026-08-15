//===- qnn-plugin.cpp - qnn/qisa dialect + pass plugin ----------*- C++ -*-===//
//
// Loadable by mlir-opt:
//   mlir-opt --load-dialect-plugin=<path>/QnnPlugin.so --qnn-to-qisa in.mlir
//
//===----------------------------------------------------------------------===//

#include "QNN/QnnDialect.h"
#include "QNN/QisaDialect.h"
#include "QNN/QnnToQisa.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "QnnQisa", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<mlir::qnn::QNNDialect, mlir::qisa::QISADialect>();
            mlir::qnn::registerQnnPasses();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  // Register the qnn-to-qisa pass with the global pass registry.
  return {MLIR_PLUGIN_API_VERSION, "QnnQisaPasses", LLVM_VERSION_STRING,
          []() { mlir::qnn::registerQnnPasses(); }};
}
