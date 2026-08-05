#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_FREEINTERMEDIATES_H_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_FREEINTERMEDIATES_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DECL
#include "lib/Dialect/Cheddar/Transforms/FreeIntermediates.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Cheddar/Transforms/FreeIntermediates.h.inc"

}  // namespace mlir::heir::cheddar

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_FREEINTERMEDIATES_H_
