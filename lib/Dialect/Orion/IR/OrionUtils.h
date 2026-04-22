#ifndef LIB_DIALECT_ORION_IR_ORIONUTILS_H_
#define LIB_DIALECT_ORION_IR_ORIONUTILS_H_

#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::orion {

inline constexpr llvm::StringLiteral kOpaqueImplStyle = "opaque";
inline constexpr llvm::StringLiteral kDiagonalBasicImplStyle = "diagonal-basic";

inline bool isSupportedLinearTransformImplStyle(StringRef implStyle) {
  return implStyle == kOpaqueImplStyle || implStyle == kDiagonalBasicImplStyle;
}

inline LogicalResult verifyImplStyle(Operation* op, StringRef expectedStyle) {
  auto implStyle = op->getAttrOfType<StringAttr>(kImplStyleAttrName);
  if (!implStyle) {
    return op->emitOpError()
           << "requires Orion implementation style `" << expectedStyle
           << "`, but no `orion.impl_style` annotation is present";
  }
  if (implStyle.getValue() == expectedStyle) {
    return success();
  }
  return op->emitOpError() << "requires Orion implementation style `"
                           << expectedStyle << "`, but got `"
                           << implStyle.getValue() << "`";
}

}  // namespace mlir::heir::orion

#endif  // LIB_DIALECT_ORION_IR_ORIONUTILS_H_
