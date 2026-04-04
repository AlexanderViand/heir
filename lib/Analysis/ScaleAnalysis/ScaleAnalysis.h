#ifndef LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
#define LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_

#include <cassert>
#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Parameters/BGV/Params.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/APInt.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

class ScaleState {
 public:
  ScaleState() : scale(std::nullopt), conflict(false) {}
  explicit ScaleState(APInt scale)
      : scale(canonicalizeUnsignedAPInt(scale)), conflict(false) {}
  explicit ScaleState(int64_t scale)
      : ScaleState(APInt(64, static_cast<uint64_t>(scale))) {}
  explicit ScaleState(bool conflict)
      : scale(std::nullopt), conflict(conflict) {}

  const APInt& getScale() const {
    assert(isInitialized() && !hasConflict());
    return scale.value();
  }

  bool operator==(const ScaleState& rhs) const {
    if (conflict != rhs.conflict) return false;
    if (scale.has_value() != rhs.scale.has_value()) return false;
    if (!scale.has_value()) return true;
    return APInt::isSameValue(*scale, *rhs.scale, /*SignedCompare=*/false);
  }

  bool isInitialized() const { return scale.has_value(); }
  bool hasConflict() const { return conflict; }

  static ScaleState join(const ScaleState& lhs, const ScaleState& rhs) {
    if (lhs.hasConflict() || rhs.hasConflict()) return ScaleState(true);
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    if (!APInt::isSameValue(*lhs.scale, *rhs.scale, /*SignedCompare=*/false)) {
      return ScaleState(true);
    }
    return lhs;
  }

  static ScaleState meet(const ScaleState& lhs, const ScaleState& rhs) {
    if (lhs.hasConflict() || rhs.hasConflict()) return ScaleState(true);
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    if (!APInt::isSameValue(*lhs.scale, *rhs.scale, /*SignedCompare=*/false)) {
      return ScaleState(true);
    }
    return lhs;
  }

  void print(llvm::raw_ostream& os) const {
    if (hasConflict()) {
      os << "ScaleState(conflict)";
      return;
    }
    if (isInitialized()) {
      os << "ScaleState(" << scale.value() << ")";
    } else {
      os << "ScaleState(uninitialized)";
    }
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const ScaleState& state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<APInt> scale;
  bool conflict;
};

class ScaleLattice : public dataflow::Lattice<ScaleState> {
 public:
  using Lattice::Lattice;
};

struct BGVScaleModel {
  using SchemeParam = bgv::SchemeParam;
  using LocalParam = bgv::LocalParam;

  static FailureOr<APInt> evalMulScale(const LocalParam& param,
                                       const APInt& lhs, const APInt& rhs);
  static FailureOr<APInt> evalMulScaleBackward(const LocalParam& param,
                                               const APInt& result,
                                               const APInt& lhs);
  static FailureOr<APInt> evalModReduceScale(const LocalParam& inputParam,
                                             const APInt& scale);
  static FailureOr<APInt> evalModReduceScaleBackward(
      const LocalParam& inputParam, const APInt& resultScale);
  static FailureOr<APInt> evalLevelReduceScale(const LocalParam& inputParam,
                                               const APInt& scale,
                                               int64_t levelToDrop);
  static FailureOr<APInt> evalLevelReduceScaleBackward(
      const LocalParam& inputParam, const APInt& resultScale,
      int64_t levelToDrop);
};

struct CKKSScaleModel {
  using SchemeParam = ckks::SchemeParam;
  using LocalParam = ckks::LocalParam;

  static FailureOr<APInt> evalMulScale(const LocalParam& param,
                                       const APInt& lhs, const APInt& rhs);
  static FailureOr<APInt> evalMulScaleBackward(const LocalParam& param,
                                               const APInt& result,
                                               const APInt& lhs);
  static FailureOr<APInt> evalModReduceScale(const LocalParam& inputParam,
                                             const APInt& scale);
  static FailureOr<APInt> evalModReduceScaleBackward(
      const LocalParam& inputParam, const APInt& resultScale);
  static FailureOr<APInt> evalLevelReduceScale(const LocalParam& inputParam,
                                               const APInt& scale,
                                               int64_t levelToDrop);
  static FailureOr<APInt> evalLevelReduceScaleBackward(
      const LocalParam& inputParam, const APInt& resultScale,
      int64_t levelToDrop);
};

struct CKKSPreciseScaleModel {
  using SchemeParam = ckks::SchemeParam;
  using LocalParam = ckks::LocalParam;

  static FailureOr<APInt> evalMulScale(const LocalParam& param,
                                       const APInt& lhs, const APInt& rhs);
  static FailureOr<APInt> evalMulScaleBackward(const LocalParam& param,
                                               const APInt& result,
                                               const APInt& lhs);
  static FailureOr<APInt> evalModReduceScale(const LocalParam& inputParam,
                                             const APInt& scale);
  static FailureOr<APInt> evalModReduceScaleBackward(
      const LocalParam& inputParam, const APInt& resultScale);
  static FailureOr<APInt> evalLevelReduceScale(const LocalParam& inputParam,
                                               const APInt& scale,
                                               int64_t levelToDrop);
  static FailureOr<APInt> evalLevelReduceScaleBackward(
      const LocalParam& inputParam, const APInt& resultScale,
      int64_t levelToDrop);
};

/// Forward Analyse the scale of each secret Value
///
/// This forward analysis roots from user input as `inputScale`,
/// and after each HE operation, the scale will be updated.
/// For ct-pt or cross-level operation, we will assume the scale of the
/// undetermined hand side to be the same as the determined one.
/// This forms the level-specific scaling factor constraint.
/// See also the "Ciphertext management" section in the document.
///
/// The analysis will stop propagation for AdjustScaleOp, as the scale
/// of it should be determined together by the forward pass (from input
/// to its operand) and the backward pass (from a determined ciphertext to
/// its result).
///
/// This analysis is expected to determine (almost) all the scales of
/// the secret Value, or ciphertext in the program.
/// The level of plaintext Value, or the opaque result of AdjustLevelOp
/// should be determined by the Backward Analysis below.
template <typename ScaleModelT>
class ScaleAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>,
      public SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysis<ScaleModelT>>;

  using SchemeParamType = typename ScaleModelT::SchemeParam;
  using LocalParamType = typename ScaleModelT::LocalParam;

  ScaleAnalysis(DataFlowSolver& solver, const SchemeParamType& schemeParam,
                const APInt& inputScale)
      : dataflow::SparseForwardDataFlowAnalysis<ScaleLattice>(solver),
        schemeParam(schemeParam),
        inputScale(inputScale) {}

  void setToEntryState(ScaleLattice* lattice) override {
    if (isa<secret::SecretType>(lattice->getAnchor().getType())) {
      propagateIfChanged(lattice, lattice->join(ScaleState(inputScale)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(ScaleState()));
  }

  LogicalResult visitOperation(Operation* op,
                               ArrayRef<const ScaleLattice*> operands,
                               ArrayRef<ScaleLattice*> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const ScaleLattice*> argumentLattices,
                         ArrayRef<ScaleLattice*> resultLattices) override;

  void propagateIfChangedWrapper(AnalysisState* state, ChangeResult changed) {
    propagateIfChanged(state, changed);
  }

 private:
  const SchemeParamType schemeParam;
  APInt inputScale;
};

/// Backward Analyse the scale of plaintext Value / opaque result of
/// AdjustLevelOp
///
/// This analysis should be run after the (forward) ScaleAnalysis
/// where the scale of (almost) all the secret Value is determined.
///
/// A special example is ct2 = mul(ct0, rs(adjust_scale(ct1))), where the scale
/// of ct0, ct1, ct2 is determined by the forward pass, rs is rescaling. Then
/// the scale of adjust_scale(ct1) should be determined by the backward pass
/// via backpropagation from ct2 to rs then to adjust_scale.
template <typename ScaleModelT>
class ScaleAnalysisBackward
    : public dataflow::SparseBackwardDataFlowAnalysis<ScaleLattice>,
      public SecretnessAnalysisDependent<ScaleAnalysisBackward<ScaleModelT>> {
 public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  friend class SecretnessAnalysisDependent<ScaleAnalysisBackward<ScaleModelT>>;

  using SchemeParamType = typename ScaleModelT::SchemeParam;
  using LocalParamType = typename ScaleModelT::LocalParam;

  ScaleAnalysisBackward(DataFlowSolver& solver,
                        SymbolTableCollection& symbolTable,
                        const SchemeParamType& schemeParam)
      : dataflow::SparseBackwardDataFlowAnalysis<ScaleLattice>(solver,
                                                               symbolTable),
        schemeParam(schemeParam) {}

  void setToExitState(ScaleLattice* lattice) override {
    propagateIfChanged(lattice, lattice->join(ScaleState()));
  }

  LogicalResult visitOperation(Operation* op, ArrayRef<ScaleLattice*> operands,
                               ArrayRef<const ScaleLattice*> results) override;

  // dummy impl
  void visitBranchOperand(OpOperand& operand) override {}
  void visitCallOperand(OpOperand& operand) override {}
  void visitNonControlFlowArguments(
      RegionSuccessor& successor, ArrayRef<BlockArgument> arguments) override {}

 private:
  const SchemeParamType schemeParam;
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

APInt getScale(Value value, DataFlowSolver* solver);

constexpr StringRef kArgScaleAttrName = "mgmt.scale";

void annotateScale(Operation* top, DataFlowSolver* solver);

APInt getScaleFromMgmtAttr(Value value);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_SCALEANALYSIS_SCALEANALYSIS_H_
