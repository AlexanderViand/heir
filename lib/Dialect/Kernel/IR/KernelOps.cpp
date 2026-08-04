#include "lib/Dialect/Kernel/IR/KernelOps.h"

// IWYU pragma: begin_keep
#include <bit>

#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"             // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#define GET_OP_CLASSES
#include "lib/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace mlir {
namespace heir {
namespace kernel {

int EvalChebyshevOp::getLevelsToDrop() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) return 0;

  bool hasBackend = false;
  for (NamedAttribute attr : module->getAttrs()) {
    if (isa<UnitAttr>(attr.getValue()) &&
        attr.getName().getValue().starts_with("backend.")) {
      hasBackend = true;
      break;
    }
  }
  if (!hasBackend) return 0;

  FailureOr<CompilationTarget> targetConfig = getTargetConfig(module);
  if (failed(targetConfig)) return 0;

  auto coefficients = getCoefficients();
  if (coefficients.empty()) return 0;

  int baseDepth = 0;
  switch (targetConfig->backendName) {
    case BackendName::Lattigo:
    case BackendName::OpenFHE:
      baseDepth =
          std::bit_width(static_cast<uint64_t>(coefficients.size() - 1));
      break;
    default:
      return 0;
  }

  // Check if linear transformation is needed.
  // If domain is not [-1, 1], we need to scale and shift, which consumes 1
  // level.
  double domainLower = getDomainLowerAttr().getValueAsDouble();
  double domainUpper = getDomainUpperAttr().getValueAsDouble();
  if (std::abs(domainLower - -1.0) > 1e-9 ||
      std::abs(domainUpper - 1.0) > 1e-9) {
    baseDepth += 1;
  }

  return baseDepth;
}

::mlir::OpOperand& EvalChebyshevOp::getOperandToReduce() {
  return getOperation()->getOpOperand(0);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
