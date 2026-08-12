#include "lib/Dialect/Cheddar/Transforms/BufferizableOpInterfaceImpl.h"

#include <cassert>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir;
using namespace mlir::heir::cheddar;

namespace {

// Generic bufferization model for the cheddar destination-passing-style ops.
// The cheddar payload types are move-only and the C++ API writes into a
// destination, so every payload-producing op is DPS on builtin tensors: tensor
// operands (the ciphertext/plaintext/constant payloads and the float
// message/diagonals buffers) bufferize to memrefs, the op loses its results,
// and each result is replaced by the buffer of its tied DPS init operand.
//
// The stock DPS model supplies the init/result alias relation and write
// semantics. Cheddar overrides its overly conservative "every init reads"
// rule: ordinary results fully overwrite their destination, while the small
// set of genuinely stateful/accumulating ops opt into a read-write init.
template <typename OpTy>
struct CheddarDpsModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          CheddarDpsModel<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (!dstOp.isDpsInit(&opOperand)) return true;
    return OpTy::readsDpsInit();
  }

  bool isNotConflicting(Operation* op, OpOperand* read, OpOperand* write,
                        const bufferization::AnalysisState& state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    // Every scale-snu API represented by a Cheddar DPS op accepts an explicitly
    // identical input and output object. Some kernels operate in place; others
    // stage their input through an internal temporary before replacing the
    // output. In either case, reusing the exact SSA value chosen as the DPS
    // destination is semantically safe.
    //
    // Keep this deliberately narrower than an alias-set query: the lowering
    // must explicitly opt into reuse by passing the same SSA value as an input
    // and the tied init. We do not approve two different values merely because
    // One-Shot believes their future buffers may alias.
    return read->getOwner() == op && write->getOwner() == op &&
           dstOp.isDpsInit(write) && read->get() == write->get();
  }

  LogicalResult verifyAnalysis(
      Operation* op, const bufferization::AnalysisState& state) const {
    if (!OpTy::readsDpsInit()) return success();
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    for (OpOperand& init : op->getOpOperands()) {
      if (!dstOp.isDpsInit(&init)) continue;
      if (isa<TensorType>(init.get().getType()) && !state.isInPlace(init))
        return op->emitOpError(
            "move-only read-write destination must bufferize in-place");
    }
    return success();
  }

  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    // Replace each tensor operand with its buffer; pass non-tensor operands
    // (context/encoder/ui/eval-key/evk-map) through unchanged.
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (OpOperand& operand : op->getOpOperands()) {
      Value v = operand.get();
      if (isa<TensorType>(v.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, v, options, state);
        if (failed(buffer)) return failure();
        newOperands.push_back(*buffer);
      } else {
        newOperands.push_back(v);
      }
    }

    // Rebuild the op on memrefs: same name/properties/attrs/operand order, no
    // results (the DPS init buffers carry the values). All operations attached
    // below are regionless and successorless.
    rewriter.setInsertionPoint(op);
    assert(op->getNumRegions() == 0 && op->getNumSuccessors() == 0);
    rewriter.insert(Operation::create(
        op->getLoc(), op->getName(), /*resultTypes=*/TypeRange{}, newOperands,
        op->getAttrDictionary(), op->getPropertiesStorage(),
        /*successors=*/BlockRange{}, /*numRegions=*/0));

    // Each result is tied to a DPS init; its bufferized value is that init's
    // buffer (already in newOperands at the init's operand index).
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    SmallVector<Value> replacements;
    replacements.reserve(op->getNumResults());
    for (OpResult res : op->getResults()) {
      OpOperand* init = dstOp.getTiedOpOperand(res);
      replacements.push_back(newOperands[init->getOperandNumber()]);
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

template <typename OpTy>
void attachIfDps(MLIRContext* ctx) {
  if constexpr (OpTy::template hasTrait<DestinationStyleOpInterface::Trait>()) {
    OpTy::template attachInterface<CheddarDpsModel<OpTy>>(*ctx);
  }
}

template <typename... OpTys>
void attachAllDpsOps(MLIRContext* ctx) {
  (attachIfDps<OpTys>(ctx), ...);
}

}  // namespace

void mlir::heir::cheddar::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, CheddarDialect* dialect) {
    attachAllDpsOps<
#define GET_OP_LIST
#include "lib/Dialect/Cheddar/IR/CheddarOps.cpp.inc"
        >(ctx);
  });
}
