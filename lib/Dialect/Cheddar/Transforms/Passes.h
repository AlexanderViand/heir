#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/Transforms/CheddarBufferize.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Cheddar/Transforms/Passes.h.inc"

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_H_
