#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"

#include <cassert>
#include <cstdint>
#include <functional>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "ScaleAnalysis"

namespace mlir {
namespace heir {

static bool isScaleBalancingConsumer(Operation* op) {
  return isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
             arith::MulIOp, arith::MulFOp>(op);
}

//===----------------------------------------------------------------------===//
// ScaleModel
//===----------------------------------------------------------------------===//

FailureOr<APInt> BGVScaleModel::evalMulScale(const bgv::LocalParam& param,
                                             const APInt& lhs,
                                             const APInt& rhs) {
  const auto* schemeParam = param.getSchemeParam();
  APInt t(64, static_cast<uint64_t>(schemeParam->getPlaintextModulus()));
  return modularMultiplication(lhs.urem(t), rhs.urem(t), t);
}

FailureOr<APInt> BGVScaleModel::evalMulScaleBackward(
    const bgv::LocalParam& param, const APInt& result, const APInt& lhs) {
  const auto* schemeParam = param.getSchemeParam();
  APInt t(64, static_cast<uint64_t>(schemeParam->getPlaintextModulus()));
  APInt lhsInv = multiplicativeInverse(lhs.urem(t), t);
  if (lhsInv.isZero()) return failure();
  return modularMultiplication(result.urem(t), lhsInv, t);
}

FailureOr<APInt> BGVScaleModel::evalModReduceScale(
    const bgv::LocalParam& inputParam, const APInt& scale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  APInt t(64, static_cast<uint64_t>(schemeParam->getPlaintextModulus()));
  auto qi = schemeParam->getQi();
  auto level = inputParam.getCurrentLevel();
  APInt qInvT =
      multiplicativeInverse(APInt(64, qi[level] % t.getZExtValue()), t);
  if (qInvT.isZero()) return failure();
  return modularMultiplication(scale.urem(t), qInvT, t);
}

FailureOr<APInt> BGVScaleModel::evalModReduceScaleBackward(
    const bgv::LocalParam& inputParam, const APInt& resultScale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  APInt t(64, static_cast<uint64_t>(schemeParam->getPlaintextModulus()));
  auto qi = schemeParam->getQi();
  auto level = inputParam.getCurrentLevel();
  return modularMultiplication(resultScale.urem(t),
                               APInt(64, qi[level] % t.getZExtValue()), t);
}

FailureOr<APInt> BGVScaleModel::evalLevelReduceScale(
    const bgv::LocalParam& inputParam, const APInt& scale,
    int64_t levelToDrop) {
  return scale;
}

FailureOr<APInt> BGVScaleModel::evalLevelReduceScaleBackward(
    const bgv::LocalParam& inputParam, const APInt& resultScale,
    int64_t levelToDrop) {
  return resultScale;
}

FailureOr<APInt> CKKSScaleModel::evalMulScale(const ckks::LocalParam& param,
                                              const APInt& lhs,
                                              const APInt& rhs) {
  return multiplyUnsignedAPIntExact(lhs, rhs);
}

FailureOr<APInt> CKKSScaleModel::evalMulScaleBackward(
    const ckks::LocalParam& param, const APInt& result, const APInt& lhs) {
  return divideUnsignedAPIntExact(result, lhs);
}

FailureOr<APInt> CKKSScaleModel::evalModReduceScale(
    const ckks::LocalParam& inputParam, const APInt& scale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  APInt nominalScale =
      getNominalPowerOfTwoScaleFromLog2(schemeParam->getLogDefaultScale());
  return divideUnsignedAPIntExact(scale, nominalScale);
}

FailureOr<APInt> CKKSScaleModel::evalModReduceScaleBackward(
    const ckks::LocalParam& inputParam, const APInt& resultScale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  APInt nominalScale =
      getNominalPowerOfTwoScaleFromLog2(schemeParam->getLogDefaultScale());
  return multiplyUnsignedAPIntExact(resultScale, nominalScale);
}

FailureOr<APInt> CKKSScaleModel::evalLevelReduceScale(
    const ckks::LocalParam& inputParam, const APInt& scale,
    int64_t levelToDrop) {
  return scale;
}

FailureOr<APInt> CKKSScaleModel::evalLevelReduceScaleBackward(
    const ckks::LocalParam& inputParam, const APInt& resultScale,
    int64_t levelToDrop) {
  return resultScale;
}

FailureOr<APInt> CKKSPreciseScaleModel::evalMulScale(
    const ckks::LocalParam& param, const APInt& lhs, const APInt& rhs) {
  return multiplyUnsignedAPIntExact(lhs, rhs);
}

FailureOr<APInt> CKKSPreciseScaleModel::evalMulScaleBackward(
    const ckks::LocalParam& param, const APInt& result, const APInt& lhs) {
  return divideUnsignedAPIntExact(result, lhs);
}

FailureOr<APInt> CKKSPreciseScaleModel::evalModReduceScale(
    const ckks::LocalParam& inputParam, const APInt& scale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  int level = inputParam.getCurrentLevel();
  if (level < 0 || level >= static_cast<int>(schemeParam->getQi().size())) {
    return failure();
  }
  APInt droppedPrime(64, static_cast<uint64_t>(schemeParam->getQi()[level]));
  return divideUnsignedAPIntNearest(scale, droppedPrime);
}

FailureOr<APInt> CKKSPreciseScaleModel::evalModReduceScaleBackward(
    const ckks::LocalParam& inputParam, const APInt& resultScale) {
  const auto* schemeParam = inputParam.getSchemeParam();
  int level = inputParam.getCurrentLevel();
  if (level < 0 || level >= static_cast<int>(schemeParam->getQi().size())) {
    return failure();
  }
  APInt droppedPrime(64, static_cast<uint64_t>(schemeParam->getQi()[level]));
  return multiplyUnsignedAPIntExact(resultScale, droppedPrime);
}

FailureOr<APInt> CKKSPreciseScaleModel::evalLevelReduceScale(
    const ckks::LocalParam& inputParam, const APInt& scale,
    int64_t levelToDrop) {
  return scale;
}

FailureOr<APInt> CKKSPreciseScaleModel::evalLevelReduceScaleBackward(
    const ckks::LocalParam& inputParam, const APInt& resultScale,
    int64_t levelToDrop) {
  return resultScale;
}

//===----------------------------------------------------------------------===//
// ScaleAnalysis (Forward)
//===----------------------------------------------------------------------===//

template <typename ScaleModelT>
LogicalResult ScaleAnalysis<ScaleModelT>::visitOperation(
    Operation* op, ArrayRef<const ScaleLattice*> operands,
    ArrayRef<ScaleLattice*> results) {
  LogicalResult status = success();
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value).getInt();
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, const ScaleState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Propagate " << state << " to " << value << "\n");
    }
    propagateIfChanged(lattice, changed);
  };

  auto getSecretOrInittedOperands =
      [&](Operation* op, SmallVectorImpl<OpOperand*>& secretOperands) {
        for (auto& opOperand : op->getOpOperands()) {
          bool isSecret = this->isSecretInternal(op, opOperand.get());
          bool isMgmtDefined =
              isa_and_nonnull<mgmt::InitOp>(opOperand.get().getDefiningOp());
          if (isSecret || isMgmtDefined) {
            // Treat mgmt.init operands like secret values for scale
            // propagation so ct-pt materialization sees the plaintext scale.
            secretOperands.push_back(&opOperand);
          }
        }
      };

  auto getOperandScales = [&](Operation* op, SmallVectorImpl<APInt>& scales) {
    SmallVector<OpOperand*> secretOperands;
    getSecretOrInittedOperands(op, secretOperands);

    for (auto* operand : secretOperands) {
      auto operandState = getLatticeElement(operand->get())->getValue();
      if (!operandState.isInitialized() || operandState.hasConflict()) {
        continue;
      }
      scales.push_back(operandState.getScale());
    }
    if (scales.size() > 1) {
      if (!APInt::isSameValue(scales[0], scales[1], /*SignedCompare=*/false)) {
        LLVM_DEBUG(llvm::dbgs() << "Different scales: " << scales[0] << ", "
                                << scales[1] << " for " << *op << "\n");
      }
    }
  };

  llvm::TypeSwitch<Operation&>(*op)
      .template Case<arith::MulIOp, arith::MulFOp>([&](auto mulOp) {
        SmallVector<APInt> scales;
        getOperandScales(mulOp, scales);
        // there must be at least one secret operand that has scale
        if (scales.empty()) {
          return;
        }
        auto scaleLhs = scales[0];
        auto scaleRhs = scaleLhs;
        // default to the same scale for both operand
        if (scales.size() > 1) {
          scaleRhs = scales[1];
        }

        // propagate scale to result
        auto result = ScaleModelT::evalMulScale(
            getLocalParam(mulOp.getResult()), scaleLhs, scaleRhs);
        if (failed(result)) {
          op->emitOpError("failed to infer exact output scale");
          status = failure();
          return;
        }
        propagate(mulOp.getResult(), ScaleState(*result));
      })
      .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        SmallVector<APInt> scales;
        getOperandScales(modReduceOp, scales);
        // there must be at least one secret operand that has scale
        if (scales.empty()) {
          return;
        }

        // propagate scale to result
        auto scale = scales[0];
        // get level of the operand.
        auto newScale = ScaleModelT::evalModReduceScale(
            getLocalParam(modReduceOp.getInput()), scale);
        if (failed(newScale)) {
          op->emitOpError("failed to infer exact output scale");
          status = failure();
          return;
        }
        propagate(modReduceOp.getResult(), ScaleState(*newScale));
      })
      .template Case<mgmt::LevelReduceOp>([&](auto levelReduceOp) {
        SmallVector<APInt> scales;
        getOperandScales(levelReduceOp, scales);
        if (scales.empty()) {
          return;
        }

        auto newScale = ScaleModelT::evalLevelReduceScale(
            getLocalParam(levelReduceOp.getInput()), scales[0],
            levelReduceOp.getLevelToDrop());
        if (failed(newScale)) {
          op->emitOpError("failed to infer exact output scale");
          status = failure();
          return;
        }
        propagate(levelReduceOp.getResult(), ScaleState(*newScale));
      })
      .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
        // adjust_scale is materialized later. When a single downstream
        // scale-balancing user already has a known peer scale, use that as
        // the target scale for this abstract result.
        if (!adjustScaleOp->hasOneUse()) {
          return;
        }
        Operation* consumer = *adjustScaleOp->user_begin();
        if (!isScaleBalancingConsumer(consumer)) {
          return;
        }

        std::optional<APInt> peerScale;
        for (Value operand : consumer->getOperands()) {
          if (operand == adjustScaleOp.getResult()) {
            continue;
          }
          auto operandState = getLatticeElement(operand)->getValue();
          if (!operandState.isInitialized() || operandState.hasConflict()) {
            continue;
          }
          if (!peerScale.has_value()) {
            peerScale = operandState.getScale();
            continue;
          }
          if (!APInt::isSameValue(*peerScale, operandState.getScale(),
                                  /*SignedCompare=*/false)) {
            return;
          }
        }
        if (peerScale.has_value()) {
          propagate(adjustScaleOp.getResult(), ScaleState(*peerScale));
        }
        return;
      })
      .template Case<mgmt::InitOp>([&](auto initOp) {
        auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(initOp.getResult());
        // if there is scale annotation, use it
        if (mgmtAttr && !mgmt::getScaleAsAPInt(mgmtAttr).isZero()) {
          propagate(initOp.getResult(),
                    ScaleState(mgmt::getScaleAsAPInt(mgmtAttr)));
        }
      })
      .template Case<mgmt::BootstrapOp>([&](auto bootstrapOp) {
        // inputScale is either Delta or Delta^2 depending on the analysis
        // initialization.
        propagate(bootstrapOp.getResult(), ScaleState(inputScale));
      })
      .Default([&](auto& op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        this->getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        SmallVector<APInt> scales;
        getOperandScales(&op, scales);
        if (scales.empty()) {
          return;
        }
        if (scales.size() > 1 && !APInt::isSameValue(scales[0], scales[1],
                                                     /*SignedCompare=*/false)) {
          op.emitOpError("exact scale mismatch between operands");
          status = failure();
          return;
        }

        // just propagate the scale
        for (auto result : secretResults) {
          propagate(result, ScaleState(scales[0]));
        }
      });
  return status;
}

template <typename ScaleModelT>
void ScaleAnalysis<ScaleModelT>::visitExternalCall(
    CallOpInterface call, ArrayRef<const ScaleLattice*> argumentLattices,
    ArrayRef<ScaleLattice*> resultLattices) {
  auto callback = std::bind(&ScaleAnalysis::propagateIfChangedWrapper, this,
                            std::placeholders::_1, std::placeholders::_2);
  ::mlir::heir::visitExternalCall<ScaleState, ScaleLattice>(
      call, argumentLattices, resultLattices, callback);
}

// instantiation
template class ScaleAnalysis<BGVScaleModel>;
template class ScaleAnalysis<CKKSScaleModel>;
template class ScaleAnalysis<CKKSPreciseScaleModel>;

//===----------------------------------------------------------------------===//
// ScaleAnalysis (Backward)
//===----------------------------------------------------------------------===//

template <typename ScaleModelT>
LogicalResult ScaleAnalysisBackward<ScaleModelT>::visitOperation(
    Operation* op, ArrayRef<ScaleLattice*> operands,
    ArrayRef<const ScaleLattice*> results) {
  LogicalResult status = success();
  auto getLocalParam = [&](Value value) {
    auto level = getLevelFromMgmtAttr(value).getInt();
    auto dimension = getDimensionFromMgmtAttr(value);
    return LocalParamType(&schemeParam, level, dimension);
  };

  auto propagate = [&](Value value, const ScaleState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    if (changed == ChangeResult::Change) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Back Propagate " << state << " to " << value << "\n");
    }
    propagateIfChanged(lattice, changed);
  };

  auto getSecretOrInittedOperands =
      [&](Operation* op, SmallVectorImpl<OpOperand*>& secretOperands) {
        LLVM_DEBUG(
            { llvm::dbgs() << "secretness of operands for " << *op << ":\n"; });
        for (auto& opOperand : op->getOpOperands()) {
          bool isSecret = this->isSecretInternal(op, opOperand.get());
          bool isMgmtDefined =
              isa_and_nonnull<mgmt::InitOp>(opOperand.get().getDefiningOp());
          LLVM_DEBUG({
            llvm::dbgs() << " " << opOperand.getOperandNumber()
                         << ": isSecret=" << isSecret
                         << ", isMgmtDefined=" << isMgmtDefined << "\n";
          });
          if (isSecret || isMgmtDefined) {
            // Treat it as if it were secret for the purpose of scale
            // propagation
            secretOperands.push_back(&opOperand);
          }
        }
      };

  auto getOperandScales =
      [&](Operation* op, SmallVectorImpl<int64_t>& operandWithoutScaleIndices,
          SmallVectorImpl<APInt>& scales) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Operand scales for " << op->getName() << ": ");
        SmallVector<OpOperand*> secretOperands;
        getSecretOrInittedOperands(op, secretOperands);

        for (auto* operand : secretOperands) {
          auto operandState = getLatticeElement(operand->get())->getValue();
          if (!operandState.isInitialized() || operandState.hasConflict()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "o" << operand->getOperandNumber() << "(uninit), ");
            operandWithoutScaleIndices.push_back(operand->getOperandNumber());
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << "o" << operand->getOperandNumber() << "("
                                  << operandState.getScale() << "), ");
          scales.push_back(operandState.getScale());
        }
        if (scales.size() > 1) {
          if (!APInt::isSameValue(scales[0], scales[1],
                                  /*SignedCompare=*/false)) {
            LLVM_DEBUG(llvm::dbgs() << "Different scales: " << scales[0] << ", "
                                    << scales[1] << " for " << *op << "\n");
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "\n");
      };

  auto getResultScales = [&](Operation* op, SmallVectorImpl<APInt>& scales) {
    LLVM_DEBUG(llvm::dbgs() << "Result scales for " << op->getName() << ": ");
    SmallVector<OpResult> secretResults;
    this->getSecretResults(op, secretResults);

    for (auto result : secretResults) {
      auto resultState = getLatticeElement(result)->getValue();
      if (!resultState.isInitialized() || resultState.hasConflict()) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "r" << cast<OpResult>(result).getResultNumber()
                              << "(" << resultState.getScale() << "), ");
      scales.push_back(resultState.getScale());
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  };

  LDBG() << "Backward analysis visiting: " << *op;
  llvm::TypeSwitch<Operation&>(*op)
      .template Case<arith::MulIOp, arith::MulFOp>([&](auto mulOp) {
        SmallVector<APInt> resultScales;
        getResultScales(mulOp, resultScales);
        // there must be at least one secret result that has scale
        if (resultScales.empty()) {
          return;
        }
        SmallVector<int64_t> operandWithoutScaleIndices;
        SmallVector<APInt> operandScales;
        getOperandScales(mulOp, operandWithoutScaleIndices, operandScales);
        // there must be at least one secret operand that has scale
        if (operandScales.empty()) {
          mulOp->emitError("No secret operand has scale");
          status = failure();
          return;
        }
        // two operands have scale, succeed.
        if (operandScales.size() > 1) {
          return;
        }
        auto presentScale = operandScales[0];

        // propagate scale to other operand; this is guarded
        // by the loop for a weird reason: the secretness of the
        // non-scale-holding operand might not be initialized yet, depending
        // on the order in which the analyses run.
        for (auto otherIndex : operandWithoutScaleIndices) {
          auto scaleOther = ScaleModelT::evalMulScaleBackward(
              getLocalParam(mulOp.getResult()), resultScales[0], presentScale);
          if (failed(scaleOther)) {
            mulOp->emitError("failed to back-propagate exact operand scale");
            status = failure();
            return;
          }
          propagate(mulOp->getOperand(otherIndex), ScaleState(*scaleOther));
        }
      })
      .template Case<mgmt::ModReduceOp>([&](auto modReduceOp) {
        SmallVector<APInt> resultScales;
        getResultScales(modReduceOp, resultScales);
        // there must be at least one secret result that has scale
        if (resultScales.empty()) {
          return;
        }
        SmallVector<int64_t> operandWithoutScaleIndices;
        SmallVector<APInt> scales;
        getOperandScales(modReduceOp, operandWithoutScaleIndices, scales);
        // if all operands have scale, succeed.
        if (!scales.empty()) {
          return;
        }

        // propagate scale to operand
        auto resultScale = resultScales[0];
        // get level of the operand.
        auto newScale = ScaleModelT::evalModReduceScaleBackward(
            getLocalParam(modReduceOp.getInput()), resultScale);
        if (failed(newScale)) {
          modReduceOp->emitError(
              "failed to back-propagate exact operand scale");
          status = failure();
          return;
        }
        propagate(modReduceOp.getInput(), ScaleState(*newScale));
      })
      .template Case<mgmt::LevelReduceOp>([&](auto levelReduceOp) {
        SmallVector<APInt> resultScales;
        getResultScales(levelReduceOp, resultScales);
        if (resultScales.empty()) {
          return;
        }
        SmallVector<int64_t> operandWithoutScaleIndices;
        SmallVector<APInt> scales;
        getOperandScales(levelReduceOp, operandWithoutScaleIndices, scales);
        if (!scales.empty()) {
          return;
        }

        auto newScale = ScaleModelT::evalLevelReduceScaleBackward(
            getLocalParam(levelReduceOp.getInput()), resultScales[0],
            levelReduceOp.getLevelToDrop());
        if (failed(newScale)) {
          levelReduceOp->emitError(
              "failed to back-propagate exact operand scale");
          status = failure();
          return;
        }
        propagate(levelReduceOp.getInput(), ScaleState(*newScale));
      })
      .template Case<mgmt::AdjustScaleOp>([&](auto adjustScaleOp) {
        // Do not back propagate through adjust scale op
        return;
      })
      .template Case<mgmt::BootstrapOp>([&](auto bootstrapOp) {
        // Bootstrap resets the ciphertext to the configured input scale. The
        // output scale does not constrain the pre-bootstrap operand scale.
        return;
      })
      .Default([&](auto& op) {
        // condition on result secretness
        SmallVector<OpResult> secretResults;
        this->getSecretResults(&op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        SmallVector<APInt> scales;
        getResultScales(&op, scales);
        if (scales.empty()) {
          return;
        }

        // propagate the scale to all operands
        // including plaintext (non-secret)
        for (auto operand : op.getOperands()) {
          propagate(operand, ScaleState(scales[0]));
        }
      });
  return status;
}

// instantiation
template class ScaleAnalysisBackward<BGVScaleModel>;
template class ScaleAnalysisBackward<CKKSScaleModel>;
template class ScaleAnalysisBackward<CKKSPreciseScaleModel>;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

APInt getScale(Value value, DataFlowSolver* solver) {
  auto* lattice = solver->lookupState<ScaleLattice>(value);
  if (!lattice) {
    assert(false && "ScaleLattice not found");
    return APInt(64, 0);
  }
  if (!lattice->getValue().isInitialized() ||
      lattice->getValue().hasConflict()) {
    assert(false && "ScaleLattice not initialized");
    return APInt(64, 0);
  }
  return lattice->getValue().getScale();
}

APInt getScaleFromMgmtAttr(Value value) {
  auto mgmtAttr = mgmt::findMgmtAttrAssociatedWith(value);
  if (!mgmtAttr) {
    assert(false && "MgmtAttr not found");
    return APInt(64, 0);
  }
  return mgmt::getScaleAsAPInt(mgmtAttr);
}

void annotateScale(Operation* top, DataFlowSolver* solver) {
  walkValues(top, [&](Value value) {
    if (mgmt::shouldHaveMgmtAttribute(value, solver)) {
      auto* lattice = solver->lookupState<ScaleLattice>(value);
      if (!lattice || lattice->getValue().hasConflict() ||
          !lattice->getValue().isInitialized()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping scale annotation for unresolved value " << value
                   << "\n");
        removeAttributeAssociatedWith(value, kArgScaleAttrName);
        return;
      }
      auto scale = lattice->getValue().getScale();
      setAttributeAssociatedWith(
          value, kArgScaleAttrName,
          getSignlessIntegerAttr(top->getContext(), scale));
    }
  });
}

}  // namespace heir
}  // namespace mlir
