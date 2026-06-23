#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project

// TODO(cheddar-dps): The cheddar dialect was migrated to destination-passing
// style (ops on builtin tensors, e.g. `cheddar.add ins(...) outs(%dest)`), so
// this LWE->cheddar conversion is temporarily DISABLED. The original
// value-style implementation lives in git history (the commit immediately
// before the cheddar DPS migration) and must be rewritten to emit tensor-DPS
// cheddar ops: convert `lwe.ciphertext` -> `tensor<!cheddar.ciphertext>` and
// materialise `bufferization.alloc_tensor` destinations at each rewrite site.
// Until then the pass fails loudly rather than silently miscompiling.

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOCHEDDAR
#include "lib/Dialect/LWE/Conversions/LWEToCheddar/LWEToCheddar.h.inc"

namespace {

struct LWEToCheddar : public impl::LWEToCheddarBase<LWEToCheddar> {
  void runOnOperation() override {
    getOperation()->emitError()
        << "lwe-to-cheddar is temporarily disabled during the cheddar "
           "destination-passing-style migration; the conversion needs to be "
           "rewritten to emit tensor-DPS cheddar ops (see TODO in "
           "LWEToCheddar.cpp)";
    signalPassFailure();
  }
};

}  // namespace

}  // namespace mlir::heir::lwe
