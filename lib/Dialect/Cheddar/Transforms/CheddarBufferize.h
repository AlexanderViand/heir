#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CHEDDARBUFFERIZE_H_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CHEDDARBUFFERIZE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_DECL_CHEDDARBUFFERIZE
#include "lib/Dialect/Cheddar/Transforms/Passes.h.inc"

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CHEDDARBUFFERIZE_H_
