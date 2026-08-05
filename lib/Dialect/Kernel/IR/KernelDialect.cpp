#include "lib/Dialect/Kernel/IR/KernelDialect.h"

#include "lib/Dialect/Kernel/IR/KernelOps.h"

// Generated definitions
#include "lib/Dialect/Kernel/IR/KernelDialect.cpp.inc"

namespace mlir {
namespace heir {
namespace kernel {

void KernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Kernel/IR/KernelOps.cpp.inc"
      >();
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
