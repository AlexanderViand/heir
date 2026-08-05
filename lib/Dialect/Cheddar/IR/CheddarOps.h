#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace cheddar {

// Side-effect resource for CHEDDAR library state that lives outside the IR's
// value graph: the key store a prepare_* op fills, the library/GPU state a
// create_* op initializes, and the RNG an encrypt draws from. Ops write this
// resource so that (in tensor form, where their operands carry no effects)
// CSE cannot merge two encryptions of the same value and DCE cannot drop a
// keygen op whose threaded result happens to be unused.
struct RuntimeStateResource
    : public ::mlir::SideEffects::Resource::Base<RuntimeStateResource> {
  ::llvm::StringRef getName() const final { return "CheddarRuntimeState"; }
  // Library state (key store, RNG, GPU setup), not pointer-addressable
  // memory.
  bool isAddressable() const final { return false; }
};

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#define GET_OP_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarOps.h.inc"

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDAROPS_H_
