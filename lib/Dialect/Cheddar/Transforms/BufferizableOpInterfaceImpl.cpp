#include "lib/Dialect/Cheddar/Transforms/BufferizableOpInterfaceImpl.h"

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"  // from @llvm-project
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
// read/write/aliasing come from DstBufferizableOpInterfaceExternalModel:
// every operand reads, only DPS inits write, and an init aliases (Equivalent)
// its tied result. We implement only `bufferize()`.
template <typename OpTy>
struct CheddarDpsModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          CheddarDpsModel<OpTy>, OpTy> {
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

    // Rebuild the op on memrefs: same name/attrs/operand order, no results
    // (the DPS init buffers carry the values).
    rewriter.setInsertionPoint(op);
    OperationState opState(op->getLoc(), op->getName(), newOperands,
                           /*resultTypes=*/TypeRange{}, op->getAttrs());
    Operation* newOp = rewriter.create(opState);
    (void)newOp;

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

template <typename... OpTys>
void attachAll(MLIRContext* ctx) {
  (OpTys::template attachInterface<CheddarDpsModel<OpTys>>(*ctx), ...);
}

}  // namespace

void mlir::heir::cheddar::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, CheddarDialect* dialect) {
    attachAll<EncodeOp, EncodeConstantOp, DecodeOp, EncryptOp, DecryptOp, AddOp,
              SubOp, MultOp, AddPlainOp, SubPlainOp, MultPlainOp, AddConstOp,
              MultConstOp, NegOp, RescaleOp, LevelDownOp, RelinearizeOp,
              RelinearizeRescaleOp, HMultOp, HRotOp, HRotAddOp, HConjOp,
              HConjAddOp, MadUnsafeOp, BootOp, LinearTransformOp, EvalPolyOp>(
        ctx);
  });
}
