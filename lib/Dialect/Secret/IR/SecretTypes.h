#ifndef LIB_DIALECT_SECRET_IR_SECRETTYPES_H_
#define LIB_DIALECT_SECRET_IR_SECRETTYPES_H_

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Secret/IR/SecretTypes.h.inc"

namespace mlir {
namespace heir {
namespace secret {

inline Type getTypeOrValueType(Type ty) {
  if (auto secretTy = dyn_cast<secret::SecretType>(ty))
    return secretTy.getValueType();
  return ty;
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_SECRET_IR_SECRETTYPES_H_
