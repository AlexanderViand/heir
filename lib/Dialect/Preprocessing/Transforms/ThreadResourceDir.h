#ifndef LIB_DIALECT_PREPROCESSING_TRANSFORMS_THREADRESOURCEDIR_H_
#define LIB_DIALECT_PREPROCESSING_TRANSFORMS_THREADRESOURCEDIR_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DECL_THREADRESOURCEDIR
#include "lib/Dialect/Preprocessing/Transforms/Passes.h.inc"

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_PREPROCESSING_TRANSFORMS_THREADRESOURCEDIR_H_
