#ifndef LIB_DIALECT_ORION_IR_ORIONDIALECT_H_
#define LIB_DIALECT_ORION_IR_ORIONDIALECT_H_

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"     // from @llvm-project

namespace mlir::heir::orion {

constexpr const static ::llvm::StringLiteral kImplStyleAttrName =
    "orion.impl_style";
constexpr const static ::llvm::StringLiteral kLevelCostUpperBoundAttrName =
    "orion.level_cost_ub";

}  // namespace mlir::heir::orion

// Generated headers (block clang-format from messing up order)
#include "lib/Dialect/Orion/IR/OrionDialect.h.inc"

#endif  // LIB_DIALECT_ORION_IR_ORIONDIALECT_H_
