#ifndef LIB_CONVERSION_POLYEXTTOPOLYNOMIAL_POLYEXTTOPOLYNOMIAL_H_
#define LIB_CONVERSION_POLYEXTTOPOLYNOMIAL_POLYEXTTOPOLYNOMIAL_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DECL
#include "lib/Conversion/PolyExtToPolynomial/PolyExtToPolynomial.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/PolyExtToPolynomial/PolyExtToPolynomial.h.inc"

}  // namespace mlir::heir

#endif  // LIB_CONVERSION_POLYEXTTOPOLYNOMIAL_POLYEXTTOPOLYNOMIAL_H_
