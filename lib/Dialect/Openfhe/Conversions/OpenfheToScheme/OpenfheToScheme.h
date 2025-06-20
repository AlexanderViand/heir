#ifndef LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_H_
#define LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DECL_OPENFHETOSCHEME
#include "lib/Dialect/Openfhe/Conversions/OpenfheToScheme/OpenfheToScheme.h.inc"

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_H_
