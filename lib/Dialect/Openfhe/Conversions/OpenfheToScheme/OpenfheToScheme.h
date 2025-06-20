#ifndef LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_H_
#define LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir::openfhe {

#define GEN_PASS_DECL
#include "lib/Dialect/Openfhe/Conversions/OpenfheToScheme/OpenfheToScheme.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Openfhe/Conversions/OpenfheToScheme/OpenfheToScheme.h.inc"

}  // namespace mlir::heir::openfhe

#endif  // LIB_DIALECT_OPENFHE_CONVERSIONS_OPENFHETOSCHEME_OPENFHETOSCHEME_H_
