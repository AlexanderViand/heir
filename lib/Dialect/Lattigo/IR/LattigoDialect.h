#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGODIALECT_H_
#define LIB_DIALECT_LATTIGO_IR_LATTIGODIALECT_H_

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"     // from @llvm-project

// Generated headers (block clang-format from messing up order)
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h.inc"

namespace mlir {
namespace heir {
namespace lattigo {

// Arg attribute identifying the sparse `logSlots` value that a
// `CKKSBootstrappingEvaluatorType` argument should refresh. Absent attr means
// the default (full-slot) bootstrapper. Used to disambiguate between multiple
// bootstrap-evaluator args in the same function when a program needs sparse
// bootstrapping at different slot counts.
constexpr const static ::llvm::StringLiteral kBootstrapLogSlotsAttrName =
    "lattigo.bootstrap_log_slots";

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGODIALECT_H_
