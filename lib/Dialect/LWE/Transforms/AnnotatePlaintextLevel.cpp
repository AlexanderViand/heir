#include "lib/Dialect/LWE/Transforms/AnnotatePlaintextLevel.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETraits.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "llvm/include/llvm/ADT/DenseSet.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DEF_ANNOTATEPLAINTEXTLEVEL
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

namespace {

// Return the highest level among an op's ciphertext operands and results, or
// nullopt if it touches no ciphertext carrying a modulus chain.
std::optional<int64_t> highestCiphertextLevel(Operation* op) {
  std::optional<int64_t> result;
  auto join = [&](Value value) {
    auto ctType =
        dyn_cast<LWECiphertextType>(getElementTypeOrSelf(value.getType()));
    if (!ctType) return;
    std::optional<int64_t> level = getLevel(ctType);
    if (!level) return;
    result = result ? std::max(*result, *level) : *level;
  };
  for (Value operand : op->getOperands()) join(operand);
  for (Value opResult : op->getResults()) join(opResult);
  return result;
}

// Return true if an op's results account for every place a value it consumes
// can end up, so the walk can follow them. Ops with regions route operands into
// block arguments (`scf.for`'s iter_args) and calls route them into a callee,
// neither of which the walk models; a terminator has no results to follow.
bool forwardsThroughResults(Operation* op) {
  return op->getNumResults() > 0 && op->getNumRegions() == 0 &&
         !isa<CallOpInterface>(op) &&
         llvm::all_of(op->getResults(), [](Value result) {
           return isa<LWEPlaintextType>(getElementTypeOrSelf(result.getType()));
         });
}

// Return the level to encode the plaintext at, or nullopt to leave it at the
// top of the modulus chain. Plaintexts are not always consumed directly: the
// encode op may feed a `tensor.from_elements` or similar first, so keep walking
// forward through the ops that forward it until one that consumes it is
// reached.
//
// The two kinds of consumer constrain the level differently:
//
//   - A ct-pt op usually sets a *lower bound*. A backend that combines at the
//     lower operand level can use one encoding at the highest level among all
//     uses. A backend that requires equal operand levels needs a separate
//     encoding for each level.
//
//   - An encryption sets an *exact* level. It has no ciphertext operand to be
//     clamped against: the backend builds the ciphertext at the plaintext's own
//     level. Both `lwe.rlwe_encrypt` and `lwe.trivial_encrypt` work
//     this way.
//
// One encoding can serve both only when the encryption's level also covers
// every ct-pt use (or equals every use for an exact-level backend). When it
// does not, or when two encryptions want different levels, no annotation is
// correct.
FailureOr<std::optional<int64_t>> findUseLevel(RLWEEncodeOp encodeOp,
                                               bool requireExactLevel) {
  std::optional<int64_t> combinedLevel;
  std::optional<int64_t> encryptedLevel;
  bool analysisIncomplete = false;
  SmallVector<Value> worklist = {encodeOp.getOutput()};
  DenseSet<Operation*> visited;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation* user : value.getUsers()) {
      if (!visited.insert(user).second) continue;

      if (isa<RLWEEncryptOp, TrivialEncryptOp>(user)) {
        // A consumer whose ciphertexts carry no modulus chain constrains
        // nothing, and leaves the plaintext to whatever its other uses need.
        std::optional<int64_t> level = highestCiphertextLevel(user);
        if (!level) continue;
        if (encryptedLevel && *encryptedLevel != *level) {
          if (requireExactLevel) return failure();
          analysisIncomplete = true;
          continue;
        }
        encryptedLevel = *level;
        continue;
      }
      if (user->hasTrait<IsCiphertextPlaintextOp>()) {
        // Likewise for a ct-pt op whose ciphertexts carry no modulus chain.
        if (std::optional<int64_t> level = highestCiphertextLevel(user)) {
          if (requireExactLevel && combinedLevel && *combinedLevel != *level) {
            return failure();
          }
          combinedLevel =
              combinedLevel ? std::max(*combinedLevel, *level) : *level;
        }
        continue;
      }
      if (!forwardsThroughResults(user)) {
        analysisIncomplete = true;
        continue;
      }
      for (Value userResult : user->getResults()) {
        worklist.push_back(userResult);
      }
    }
  }

  if (combinedLevel && encryptedLevel &&
      (requireExactLevel ? *combinedLevel != *encryptedLevel
                         : *combinedLevel > *encryptedLevel)) {
    if (requireExactLevel) return failure();
    return std::optional<int64_t>();
  }
  if (analysisIncomplete) return std::optional<int64_t>();
  return encryptedLevel ? encryptedLevel : combinedLevel;
}

struct AnnotatePlaintextLevel
    : impl::AnnotatePlaintextLevelBase<AnnotatePlaintextLevel> {
  using AnnotatePlaintextLevelBase::AnnotatePlaintextLevelBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    bool requireExactLevel = false;
    if (moduleIsCheddar(module) || moduleIsLattigo(module) ||
        moduleIsOpenfhe(module)) {
      FailureOr<CompilationTarget> target = getTargetConfig(module);
      if (failed(target)) {
        signalPassFailure();
        return;
      }
      requireExactLevel = target->requires_matching_ciphertext_plaintext_levels;
    }
    if (exactLevelBackendsOnly && !requireExactLevel) return;

    // BFV does no level management: every ciphertext sits at the bottom of the
    // modulus chain, while the backend's ciphertexts still span the whole chain
    // (the extra modulus is an artifact of the encryption technique). So the
    // chain's `current` is not an encoding level here, and there is nothing to
    // save by lowering one.
    if (moduleIsBFV(module)) {
      module->walk([&](RLWEEncodeOp encodeOp) { encodeOp.removeLevelAttr(); });
      return;
    }

    WalkResult result = module->walk([&](RLWEEncodeOp encodeOp) {
      FailureOr<std::optional<int64_t>> level =
          findUseLevel(encodeOp, requireExactLevel);
      if (failed(level)) {
        encodeOp.emitOpError(
            "is shared by ciphertext operations at different levels, but the "
            "target requires plaintext and ciphertext levels to match");
        return WalkResult::interrupt();
      }
      // Drop a level that no longer has a use to justify it
      if (!*level) {
        encodeOp.removeLevelAttr();
        return WalkResult::advance();
      }
      encodeOp.setLevel(**level);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
