#include "lib/Dialect/Orion/IR/OrionDialect.h"

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define OrionOps

#include "lib/Dialect/Orion/IR/OrionOps.h"

// Generated definitions
#include "lib/Dialect/Orion/IR/OrionDialect.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Orion/IR/OrionOps.cpp.inc"

namespace mlir {
namespace heir {
namespace orion {

// ElementwiseByOperandOpInterface: only the input ciphertext (operand 0) is
// mapped elementwise, so a tensor<Nx!ct> input is scalarized to per-ciphertext
// linear transforms by convert-elementwise-to-affine; the diagonal matrix
// (operand 1) is replicated wholesale.
bool LinearTransformOp::operandIsMappable(unsigned operandIndex) {
  return operandIndex == 0;
}

void OrionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Orion/IR/OrionOps.cpp.inc"
      >();
}

}  // namespace orion
}  // namespace heir
}  // namespace mlir
