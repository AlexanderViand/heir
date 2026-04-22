#ifndef LIB_DIALECT_CKKSSCALEPOLICY_H_
#define LIB_DIALECT_CKKSSCALEPOLICY_H_

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project

namespace mlir {
namespace heir {

constexpr const static ::llvm::StringLiteral kCKKSScalePolicyAttrName =
    "ckks.scale_policy";
constexpr const static ::llvm::StringLiteral kCKKSReducedErrorAttrName =
    "ckks.reduced_error";
constexpr const static ::llvm::StringLiteral kCKKSNominalScalePolicyValue =
    "nominal";
constexpr const static ::llvm::StringLiteral kCKKSPreciseScalePolicyValue =
    "precise";

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKSSCALEPOLICY_H_
