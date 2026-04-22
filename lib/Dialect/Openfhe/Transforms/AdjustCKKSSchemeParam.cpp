#include "lib/Dialect/Openfhe/Transforms/AdjustCKKSSchemeParam.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/CKKSScalePolicy.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Openfhe/Transforms/ScalingTechniqueUtils.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Dialect/Orion/IR/OrionUtils.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/StringRef.h"               // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"         // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/TypeID.h"              // from @llvm-project
#include "openfhe.h"                                       // from @openfhe

namespace mlir {
namespace heir {
namespace openfhe {

#define DEBUG_TYPE "openfhe-adjust-ckks-scheme-param"

namespace {

Operation* getOwningOperation(Value value) {
  if (auto result = dyn_cast<OpResult>(value)) {
    return result.getOwner();
  }
  auto blockArgument = cast<BlockArgument>(value);
  return blockArgument.getOwner()->getParentOp();
}

FailureOr<int64_t> inferNoiseScaleDegreeFromEncoding(Operation* op,
                                                     Attribute encoding,
                                                     uint64_t logDefaultScale) {
  auto inverseCanonicalEncoding =
      dyn_cast<lwe::InverseCanonicalEncodingAttr>(encoding);
  if (!inverseCanonicalEncoding) {
    op->emitOpError()
        << "OpenFHE CKKS depth adjustment requires inverse canonical "
           "encoding";
    return failure();
  }

  APInt scale = lwe::getScalingFactorFromEncodingAttr(inverseCanonicalEncoding);
  if (scale.isZero()) {
    return 0;
  }
  if (scale == 1) {
    return 0;
  }
  if (!scale.isPowerOf2()) {
    SmallString<32> scaleString;
    scale.toStringUnsigned(scaleString);
    op->emitOpError()
        << "OpenFHE CKKS depth adjustment requires a power-of-two encoding "
           "scale, but found "
        << scaleString.c_str();
    return failure();
  }
  if (logDefaultScale == 0) {
    op->emitOpError()
        << "OpenFHE CKKS depth adjustment requires a non-zero default scale";
    return failure();
  }

  unsigned logScale = scale.logBase2();
  if (logScale % logDefaultScale != 0) {
    op->emitOpError()
        << "OpenFHE CKKS depth adjustment requires the encoding scale to be "
           "an exact power of the default scale 2^"
        << logDefaultScale << ", but found 2^" << logScale;
    return failure();
  }
  return static_cast<int64_t>(logScale / logDefaultScale);
}

FailureOr<int64_t> inferNoiseScaleDegreeFromType(Operation* op, Type type,
                                                 uint64_t logDefaultScale) {
  Type elemTy = getElementTypeOrSelf(type);
  if (auto ctTy = dyn_cast<lwe::LWECiphertextType>(elemTy)) {
    return inferNoiseScaleDegreeFromEncoding(
        op, ctTy.getPlaintextSpace().getEncoding(), logDefaultScale);
  }
  if (auto ptTy = dyn_cast<lwe::LWEPlaintextType>(elemTy)) {
    return inferNoiseScaleDegreeFromEncoding(
        op, ptTy.getPlaintextSpace().getEncoding(), logDefaultScale);
  }
  return 0;
}

class OpenfheNoiseScaleDegreeState {
 public:
  OpenfheNoiseScaleDegreeState() = default;
  explicit OpenfheNoiseScaleDegreeState(int64_t degree) : degree(degree) {}

  bool isInitialized() const { return degree.has_value(); }
  int64_t getDegree() const {
    assert(isInitialized());
    return *degree;
  }

  bool operator==(const OpenfheNoiseScaleDegreeState& rhs) const = default;

  static OpenfheNoiseScaleDegreeState join(
      const OpenfheNoiseScaleDegreeState& lhs,
      const OpenfheNoiseScaleDegreeState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    return OpenfheNoiseScaleDegreeState(
        std::max(lhs.getDegree(), rhs.getDegree()));
  }

  void print(llvm::raw_ostream& os) const {
    if (isInitialized()) {
      os << "OpenfheNoiseScaleDegreeState(" << *degree << ")";
      return;
    }
    os << "OpenfheNoiseScaleDegreeState(uninitialized)";
  }

 private:
  std::optional<int64_t> degree;
};

class OpenfheNoiseScaleDegreeLattice
    : public dataflow::Lattice<OpenfheNoiseScaleDegreeState> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenfheNoiseScaleDegreeLattice)
  using Lattice::Lattice;
};

class OpenfheEffectiveHeirLevelState {
 public:
  OpenfheEffectiveHeirLevelState() = default;
  explicit OpenfheEffectiveHeirLevelState(int64_t level) : level(level) {}

  bool isInitialized() const { return level.has_value(); }
  int64_t getLevel() const {
    assert(isInitialized());
    return *level;
  }

  bool operator==(const OpenfheEffectiveHeirLevelState& rhs) const = default;

  static OpenfheEffectiveHeirLevelState join(
      const OpenfheEffectiveHeirLevelState& lhs,
      const OpenfheEffectiveHeirLevelState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    return OpenfheEffectiveHeirLevelState(
        std::min(lhs.getLevel(), rhs.getLevel()));
  }

  void print(llvm::raw_ostream& os) const {
    if (isInitialized()) {
      os << "OpenfheEffectiveHeirLevelState(" << *level << ")";
      return;
    }
    os << "OpenfheEffectiveHeirLevelState(uninitialized)";
  }

 private:
  std::optional<int64_t> level;
};

class OpenfheEffectiveHeirLevelLattice
    : public dataflow::Lattice<OpenfheEffectiveHeirLevelState> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenfheEffectiveHeirLevelLattice)
  using Lattice::Lattice;
};

std::optional<int64_t> getTypeHeirLevel(Type type) {
  auto ctType = dyn_cast<lwe::LWECiphertextType>(getElementTypeOrSelf(type));
  if (!ctType) {
    return std::nullopt;
  }
  auto modulusChain = ctType.getModulusChain();
  if (!modulusChain) {
    return std::nullopt;
  }
  return static_cast<int64_t>(modulusChain.getCurrent());
}

FailureOr<int64_t> deriveOpenfheNoiseScaleDegree(
    Operation* op, ArrayRef<int64_t> secretOperandDegrees,
    ArrayRef<Type> plaintextOperandTypes, uint64_t logDefaultScale) {
  // OpenFHE's supported CKKS scaling modes differ in their exact scale
  // schedules, but they share the same noise-scale-degree recurrence: ct-ct
  // multiplication adds degrees, rescale/modreduce removes one degree, and
  // certain structured ops such as native linear transform inject one
  // additional scheduled plaintext scale. This pass only models that common
  // chain-length requirement.
  auto maxSecretOperandDegree = [&]() -> int64_t {
    int64_t result = 0;
    for (int64_t degree : secretOperandDegrees) {
      result = std::max(result, degree);
    }
    return result;
  };

  if (isa<ckks::AddOp, ckks::SubOp, ckks::AddPlainOp, ckks::SubPlainOp,
          ckks::RotateOp, ckks::RelinearizeOp, ckks::NegateOp>(op)) {
    return maxSecretOperandDegree();
  }

  if (auto opMul = dyn_cast<ckks::MulOp>(op)) {
    if (secretOperandDegrees.size() != 2) {
      return opMul.emitOpError()
             << "expected exactly two secret operands for ckks.mul";
    }
    return secretOperandDegrees[0] + secretOperandDegrees[1];
  }

  if (auto opMulPlain = dyn_cast<ckks::MulPlainOp>(op)) {
    if (secretOperandDegrees.size() != 1) {
      return opMulPlain.emitOpError()
             << "expected exactly one secret operand for ckks.mul_plain";
    }
    if (plaintextOperandTypes.size() != 1) {
      return opMulPlain.emitOpError()
             << "expected exactly one plaintext operand for ckks.mul_plain";
    }
    FailureOr<int64_t> plaintextDegree = inferNoiseScaleDegreeFromType(
        opMulPlain, plaintextOperandTypes.front(), logDefaultScale);
    if (failed(plaintextDegree)) {
      return failure();
    }
    return secretOperandDegrees[0] + *plaintextDegree;
  }

  if (auto opRescale = dyn_cast<ckks::RescaleOp>(op)) {
    if (secretOperandDegrees.size() != 1) {
      return opRescale.emitOpError()
             << "expected exactly one secret operand for ckks.rescale";
    }
    return std::max<int64_t>(1, secretOperandDegrees[0] - 1);
  }

  if (auto opLevelReduce = dyn_cast<ckks::LevelReduceOp>(op)) {
    if (secretOperandDegrees.size() != 1) {
      return opLevelReduce.emitOpError()
             << "expected exactly one secret operand for ckks.level_reduce";
    }
    return std::max<int64_t>(
        1, secretOperandDegrees[0] - opLevelReduce.getLevelToDrop());
  }

  if (isa<ckks::BootstrapOp>(op)) {
    return 1;
  }

  if (auto linearTransform = dyn_cast<orion::LinearTransformOp>(op)) {
    auto implStyle =
        linearTransform->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
    if (!implStyle) {
      return linearTransform.emitOpError()
             << "requires `orion.impl_style` before OpenFHE CKKS depth "
                "adjustment";
    }
    if (implStyle.getValue() == orion::kOpaqueImplStyle) {
      if (secretOperandDegrees.size() != 1) {
        return linearTransform.emitOpError()
               << "expected exactly one secret operand for opaque "
                  "orion.linear_transform";
      }
      return secretOperandDegrees[0] + 1;
    }
    if (implStyle.getValue() == orion::kDiagonalBasicImplStyle) {
      return linearTransform.emitOpError()
             << "OpenFHE CKKS depth adjustment expects explicit "
                "`diagonal-basic` linear transforms to be lowered by "
                "`--lower-orion` before backend lowering";
    }
    return linearTransform.emitOpError()
           << "unsupported Orion implementation style `" << implStyle.getValue()
           << "`";
  }

  if (auto chebyshev = dyn_cast<orion::ChebyshevOp>(op)) {
    if (failed(orion::verifyImplStyle(chebyshev, orion::kOpaqueImplStyle))) {
      return failure();
    }
    if (secretOperandDegrees.size() != 1) {
      return chebyshev.emitOpError()
             << "expected exactly one secret operand for opaque "
                "orion.chebyshev";
    }
    auto levelCostAttr = chebyshev->getAttrOfType<IntegerAttr>(
        orion::kLevelCostUpperBoundAttrName);
    if (!levelCostAttr) {
      return chebyshev.emitOpError()
             << "requires `orion.level_cost_ub` before OpenFHE CKKS depth "
                "adjustment";
    }
    return secretOperandDegrees[0] +
           static_cast<int64_t>(levelCostAttr.getInt());
  }

  return maxSecretOperandDegree();
}

FailureOr<int64_t> deriveOpenfheEffectiveHeirLevel(
    Operation* op, ArrayRef<int64_t> secretOperandLevels) {
  auto minSecretOperandLevel = [&]() -> FailureOr<int64_t> {
    if (secretOperandLevels.empty()) {
      return op->emitOpError()
             << "requires at least one secret operand to derive the "
                "OpenFHE-effective level";
    }
    return *std::min_element(secretOperandLevels.begin(),
                             secretOperandLevels.end());
  };

  if (isa<ckks::AddOp, ckks::SubOp, ckks::AddPlainOp, ckks::SubPlainOp,
          ckks::RotateOp, ckks::RelinearizeOp, ckks::NegateOp, ckks::MulOp,
          ckks::MulPlainOp>(op)) {
    return minSecretOperandLevel();
  }

  if (auto opRescale = dyn_cast<ckks::RescaleOp>(op)) {
    FailureOr<int64_t> level = minSecretOperandLevel();
    if (failed(level)) {
      return failure();
    }
    return *level - 1;
  }

  if (auto opLevelReduce = dyn_cast<ckks::LevelReduceOp>(op)) {
    FailureOr<int64_t> level = minSecretOperandLevel();
    if (failed(level)) {
      return failure();
    }
    return *level - static_cast<int64_t>(opLevelReduce.getLevelToDrop());
  }

  if (auto bootstrap = dyn_cast<ckks::BootstrapOp>(op)) {
    if (auto targetLevel = bootstrap.getTargetLevel()) {
      return static_cast<int64_t>(*targetLevel);
    }
  }

  if (auto linearTransform = dyn_cast<orion::LinearTransformOp>(op)) {
    auto implStyle =
        linearTransform->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
    if (!implStyle) {
      return linearTransform.emitOpError()
             << "requires `orion.impl_style` before OpenFHE level "
                "adjustment";
    }
    if (implStyle.getValue() == orion::kOpaqueImplStyle) {
      FailureOr<int64_t> inputLevel = minSecretOperandLevel();
      if (failed(inputLevel)) {
        return failure();
      }
      int64_t plaintextLevel = linearTransform.getOrionLevelAttr().getInt();
      if (plaintextLevel < 0 || plaintextLevel > *inputLevel) {
        return linearTransform.emitOpError()
               << "expected `orion_level` to be between 0 and the input "
                  "ciphertext level "
               << *inputLevel << ", but got " << plaintextLevel;
      }
      return *inputLevel;
    }
    if (implStyle.getValue() == orion::kDiagonalBasicImplStyle) {
      return linearTransform.emitOpError()
             << "OpenFHE level adjustment expects explicit "
                "`diagonal-basic` linear transforms to be lowered by "
                "`--lower-orion` before backend lowering";
    }
    return linearTransform.emitOpError()
           << "unsupported Orion implementation style `" << implStyle.getValue()
           << "`";
  }

  if (auto chebyshev = dyn_cast<orion::ChebyshevOp>(op)) {
    if (failed(orion::verifyImplStyle(chebyshev, orion::kOpaqueImplStyle))) {
      return failure();
    }
    FailureOr<int64_t> inputLevel = minSecretOperandLevel();
    if (failed(inputLevel)) {
      return failure();
    }
    auto levelCostAttr = chebyshev->getAttrOfType<IntegerAttr>(
        orion::kLevelCostUpperBoundAttrName);
    if (!levelCostAttr) {
      return chebyshev.emitOpError()
             << "requires `orion.level_cost_ub` before OpenFHE level "
                "adjustment";
    }
    return *inputLevel - static_cast<int64_t>(levelCostAttr.getInt());
  }

  if (auto resultLevel = getTypeHeirLevel(op->getResult(0).getType())) {
    return *resultLevel;
  }

  return minSecretOperandLevel();
}

bool isCiphertextType(Type type) {
  return isa<lwe::LWECiphertextType>(getElementTypeOrSelf(type));
}

bool isPlaintextType(Type type) {
  return isa<lwe::LWEPlaintextType>(getElementTypeOrSelf(type));
}

void getCiphertextResults(Operation* op, SmallVectorImpl<OpResult>& results) {
  for (OpResult result : op->getResults()) {
    if (isCiphertextType(result.getType())) {
      results.push_back(result);
    }
  }
}

void getCiphertextOperands(Operation* op,
                           SmallVectorImpl<OpOperand*>& operands) {
  for (OpOperand& operand : op->getOpOperands()) {
    if (isCiphertextType(operand.get().getType())) {
      operands.push_back(&operand);
    }
  }
}

void getPlaintextOperands(Operation* op,
                          SmallVectorImpl<OpOperand*>& operands) {
  for (OpOperand& operand : op->getOpOperands()) {
    if (isPlaintextType(operand.get().getType())) {
      operands.push_back(&operand);
    }
  }
}

class OpenfheNoiseScaleDegreeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          OpenfheNoiseScaleDegreeLattice> {
 public:
  OpenfheNoiseScaleDegreeAnalysis(DataFlowSolver& solver,
                                  uint64_t logDefaultScale)
      : SparseForwardDataFlowAnalysis(solver),
        logDefaultScale(logDefaultScale) {}

  void setToEntryState(OpenfheNoiseScaleDegreeLattice* lattice) override {
    Value anchor = lattice->getAnchor();
    Type type = anchor.getType();
    if (!isCiphertextType(type) && !isPlaintextType(type)) {
      return;
    }
    FailureOr<int64_t> degree = inferNoiseScaleDegreeFromType(
        getOwningOperation(anchor), type, logDefaultScale);
    if (failed(degree)) {
      return;
    }
    propagateIfChanged(lattice,
                       lattice->join(OpenfheNoiseScaleDegreeState(*degree)));
  }

  LogicalResult visitOperation(
      Operation* op, ArrayRef<const OpenfheNoiseScaleDegreeLattice*> operands,
      ArrayRef<OpenfheNoiseScaleDegreeLattice*> results) override {
    SmallVector<OpResult> ciphertextResults;
    getCiphertextResults(op, ciphertextResults);
    if (ciphertextResults.empty()) {
      return success();
    }

    SmallVector<OpOperand*> ciphertextOperands;
    getCiphertextOperands(op, ciphertextOperands);
    SmallVector<int64_t> ciphertextOperandDegrees;
    for (OpOperand* operand : ciphertextOperands) {
      auto* lattice = getLatticeElement(operand->get());
      if (!lattice || !lattice->getValue().isInitialized()) {
        continue;
      }
      ciphertextOperandDegrees.push_back(lattice->getValue().getDegree());
    }

    SmallVector<OpOperand*> plaintextOperands;
    getPlaintextOperands(op, plaintextOperands);
    SmallVector<Type> plaintextOperandTypes;
    for (OpOperand* operand : plaintextOperands) {
      plaintextOperandTypes.push_back(operand->get().getType());
    }

    FailureOr<int64_t> resultDegree = deriveOpenfheNoiseScaleDegree(
        op, ciphertextOperandDegrees, plaintextOperandTypes, logDefaultScale);
    if (failed(resultDegree)) {
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "OpenFHE degree " << *resultDegree << " for op `"
                   << op->getName() << "`\n";
    });

    for (OpResult result : ciphertextResults) {
      auto* lattice = getLatticeElement(result);
      propagateIfChanged(
          lattice, lattice->join(OpenfheNoiseScaleDegreeState(*resultDegree)));
    }
    return success();
  }

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const OpenfheNoiseScaleDegreeLattice*> argumentLattices,
      ArrayRef<OpenfheNoiseScaleDegreeLattice*> resultLattices) override {
    auto callback = [&](AnalysisState* state, ChangeResult changed) {
      propagateIfChanged(state, changed);
    };
    ::mlir::heir::visitExternalCall<OpenfheNoiseScaleDegreeState,
                                    OpenfheNoiseScaleDegreeLattice>(
        call, argumentLattices, resultLattices, callback);
  }

 private:
  uint64_t logDefaultScale;
};

class OpenfheEffectiveHeirLevelAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          OpenfheEffectiveHeirLevelLattice> {
 public:
  OpenfheEffectiveHeirLevelAnalysis(DataFlowSolver& solver)
      : SparseForwardDataFlowAnalysis(solver) {}

  void setToEntryState(OpenfheEffectiveHeirLevelLattice* lattice) override {
    if (auto level = getTypeHeirLevel(lattice->getAnchor().getType())) {
      propagateIfChanged(lattice,
                         lattice->join(OpenfheEffectiveHeirLevelState(*level)));
    }
  }

  LogicalResult visitOperation(
      Operation* op, ArrayRef<const OpenfheEffectiveHeirLevelLattice*> operands,
      ArrayRef<OpenfheEffectiveHeirLevelLattice*> results) override {
    SmallVector<OpResult> ciphertextResults;
    getCiphertextResults(op, ciphertextResults);
    if (ciphertextResults.empty()) {
      return success();
    }

    SmallVector<OpOperand*> ciphertextOperands;
    getCiphertextOperands(op, ciphertextOperands);
    SmallVector<int64_t> ciphertextOperandLevels;
    for (OpOperand* operand : ciphertextOperands) {
      auto* lattice = getLatticeElement(operand->get());
      if (!lattice || !lattice->getValue().isInitialized()) {
        continue;
      }
      ciphertextOperandLevels.push_back(lattice->getValue().getLevel());
    }

    FailureOr<int64_t> resultLevel =
        deriveOpenfheEffectiveHeirLevel(op, ciphertextOperandLevels);
    if (failed(resultLevel)) {
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "OpenFHE effective level " << *resultLevel << " for op `"
                   << op->getName() << "`\n";
    });

    for (OpResult result : ciphertextResults) {
      auto* lattice = getLatticeElement(result);
      propagateIfChanged(
          lattice, lattice->join(OpenfheEffectiveHeirLevelState(*resultLevel)));
    }
    return success();
  }

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const OpenfheEffectiveHeirLevelLattice*> argumentLattices,
      ArrayRef<OpenfheEffectiveHeirLevelLattice*> resultLattices) override {
    auto callback = [&](AnalysisState* state, ChangeResult changed) {
      propagateIfChanged(state, changed);
    };
    ::mlir::heir::visitExternalCall<OpenfheEffectiveHeirLevelState,
                                    OpenfheEffectiveHeirLevelLattice>(
        call, argumentLattices, resultLattices, callback);
  }
};

int64_t getRequiredOpenfheChainLength(Operation* top, DataFlowSolver& solver,
                                      int64_t maxHeirLevel) {
  int64_t requiredChainLength = 0;
  walkValues(top, [&](Value value) {
    auto* degreeLattice =
        solver.lookupState<OpenfheNoiseScaleDegreeLattice>(value);
    auto* levelLattice =
        solver.lookupState<OpenfheEffectiveHeirLevelLattice>(value);
    if (!degreeLattice || !levelLattice) {
      return;
    }
    auto degreeState = degreeLattice->getValue();
    auto levelState = levelLattice->getValue();
    if (!degreeState.isInitialized() || !levelState.isInitialized()) {
      return;
    }
    int64_t budgetLevel = levelState.getLevel();
    if (auto linearTransform =
            dyn_cast_or_null<orion::LinearTransformOp>(value.getDefiningOp())) {
      auto implStyle =
          linearTransform->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
      if (implStyle && implStyle.getValue() == orion::kOpaqueImplStyle) {
        budgetLevel =
            static_cast<int64_t>(linearTransform.getOrionLevelAttr().getInt());
      }
    }
    int64_t needed = degreeState.getDegree() + (maxHeirLevel - budgetLevel);
    LLVM_DEBUG({
      llvm::dbgs() << "value " << value << " level=" << levelState.getLevel()
                   << " budgetLevel=" << budgetLevel
                   << " degree=" << degreeState.getDegree()
                   << " requires chain length " << needed << "\n";
    });
    requiredChainLength = std::max(requiredChainLength, needed);
  });
  return requiredChainLength;
}

bool usesBuiltInAuxPlaintextModulus(StringRef scalingTechnique) {
  return scalingTechnique == kScalingTechniqueFlexibleAutoExt;
}

bool containsOpaqueLinearTransform(ModuleOp module) {
  bool found = false;
  module.walk([&](orion::LinearTransformOp linearTransform) {
    auto implStyle =
        linearTransform->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
    if (implStyle && implStyle.getValue() == orion::kOpaqueImplStyle) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

Value getCiphertextOperand(Operation* op) {
  for (Value operand : op->getOperands()) {
    if (isCiphertextType(operand.getType())) {
      return operand;
    }
  }
  return Value();
}

Value getPlaintextOperand(Operation* op) {
  for (Value operand : op->getOperands()) {
    if (isPlaintextType(operand.getType())) {
      return operand;
    }
  }
  return Value();
}

LogicalResult setEncodeRequirement(lwe::RLWEEncodeOp encodeOp,
                                   int64_t requiredNoiseScaleDegree,
                                   int64_t requiredHeirLevel) {
  encodeOp->setAttr(
      "level", IntegerAttr::get(IntegerType::get(encodeOp.getContext(), 64),
                                requiredHeirLevel));
  encodeOp->setAttr(
      kNoiseScaleDegreeAttrName,
      IntegerAttr::get(IntegerType::get(encodeOp.getContext(), 64),
                       requiredNoiseScaleDegree));
  return success();
}

LogicalResult resolveAddPlainEncodes(ModuleOp module, DataFlowSolver& solver,
                                     uint64_t logDefaultScale) {
  (void)logDefaultScale;
  bool analysisFailed = false;
  struct EncodeRewrite {
    Operation* user;
    unsigned operandNumber;
    lwe::RLWEEncodeOp encodeOp;
    int64_t requiredNoiseScaleDegree;
    int64_t requiredHeirLevel;
  };
  SmallVector<EncodeRewrite> rewrites;
  module.walk(
      [&](Operation* op) {
        if (!isa<ckks::AddPlainOp, ckks::SubPlainOp>(op)) {
          return;
        }

        Value ciphertext = getCiphertextOperand(op);
        Value plaintext = getPlaintextOperand(op);
        if (!ciphertext || !plaintext) {
          return;
        }

        auto encodeOp = plaintext.getDefiningOp<lwe::RLWEEncodeOp>();
        if (!encodeOp) {
          return;
        }

        auto* degreeLattice =
            solver.lookupState<OpenfheNoiseScaleDegreeLattice>(ciphertext);
        if (!degreeLattice || !degreeLattice->getValue().isInitialized()) {
          op->emitOpError()
              << "requires resolved OpenFHE noise-scale degree for plaintext "
                 "add/sub conversion";
          analysisFailed = true;
          return;
        }
        auto* levelLattice =
            solver.lookupState<OpenfheEffectiveHeirLevelLattice>(ciphertext);
        if (!levelLattice || !levelLattice->getValue().isInitialized()) {
          op->emitOpError()
              << "requires ciphertext operands with explicit HEIR levels for "
                 "OpenFHE plaintext add/sub conversion";
          analysisFailed = true;
          return;
        }

        for (OpOperand& operand : op->getOpOperands()) {
          if (operand.get() != plaintext) {
            continue;
          }
          rewrites.push_back(
              EncodeRewrite{op, operand.getOperandNumber(), encodeOp,
                            degreeLattice->getValue().getDegree(),
                            levelLattice->getValue().getLevel()});
          break;
        }
      });

  if (analysisFailed) {
    return failure();
  }

  DenseMap<Operation*, std::pair<int64_t, int64_t>> assignedRequirements;
  for (auto rewrite : rewrites) {
    Operation* user = rewrite.user;
    unsigned operandNumber = rewrite.operandNumber;
    lwe::RLWEEncodeOp encodeOp = rewrite.encodeOp;
    int64_t requiredNoiseScaleDegree = rewrite.requiredNoiseScaleDegree;
    int64_t requiredHeirLevel = rewrite.requiredHeirLevel;
    if (!encodeOp.getOutput().hasOneUse()) {
      OpBuilder builder(user);
      auto clonedEncode = lwe::RLWEEncodeOp::create(
          builder, encodeOp.getLoc(), encodeOp.getOutput().getType(),
          encodeOp.getInput(), encodeOp.getEncoding(), encodeOp.getRing(),
          builder.getI64IntegerAttr(requiredHeirLevel));
      clonedEncode->setAttr(
          kNoiseScaleDegreeAttrName,
          builder.getI64IntegerAttr(requiredNoiseScaleDegree));
      user->setOperand(operandNumber, clonedEncode.getOutput());
      continue;
    }

    auto it = assignedRequirements.find(encodeOp.getOperation());
    if (it == assignedRequirements.end()) {
      if (failed(setEncodeRequirement(encodeOp, requiredNoiseScaleDegree,
                                      requiredHeirLevel))) {
        return failure();
      }
      assignedRequirements.try_emplace(
          encodeOp.getOperation(),
          std::make_pair(requiredNoiseScaleDegree, requiredHeirLevel));
      continue;
    }

    if (it->second ==
        std::make_pair(requiredNoiseScaleDegree, requiredHeirLevel)) {
      continue;
    }

    encodeOp.emitOpError()
        << "single-use plaintext encode received conflicting OpenFHE "
           "requirements";
    return failure();
  }

  return success();
}

LogicalResult resolveOpaqueLinearTransformNativeLevels(ModuleOp module,
                                                       DataFlowSolver& solver) {
  auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
      ckks::CKKSDialect::kSchemeParamAttrName);
  if (!schemeParamAttr) {
    module.emitOpError()
        << "requires ckks.schemeParam to resolve OpenFHE native linear "
           "transform levels";
    return failure();
  }

  StringRef scalingTechnique = resolveScalingTechnique(
      module->getAttrOfType<StringAttr>(kScalingTechniqueAttrName)
          ? module->getAttrOfType<StringAttr>(kScalingTechniqueAttrName)
                .getValue()
          : StringRef());
  bool builtinAuxModulus = usesBuiltInAuxPlaintextModulus(scalingTechnique);

  Builder builder(module.getContext());
  bool analysisFailed = false;
  module.walk([&](orion::LinearTransformOp linearTransform) {
    auto implStyle =
        linearTransform->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
    if (!implStyle) {
      linearTransform.emitOpError()
          << "requires `orion.impl_style` before OpenFHE native linear "
             "transform level resolution";
      analysisFailed = true;
      return WalkResult::interrupt();
    }
    if (implStyle.getValue() != orion::kOpaqueImplStyle) {
      return WalkResult::advance();
    }

    auto* inputLevelLattice =
        solver.lookupState<OpenfheEffectiveHeirLevelLattice>(
            linearTransform.getInput());
    if (!inputLevelLattice || !inputLevelLattice->getValue().isInitialized()) {
      linearTransform.emitOpError()
          << "requires a resolved ciphertext input level for OpenFHE native "
             "linear transform lowering";
      analysisFailed = true;
      return WalkResult::interrupt();
    }

    int64_t inputHeirLevel = inputLevelLattice->getValue().getLevel();
    if (inputHeirLevel < 0 ||
        inputHeirLevel >= static_cast<int64_t>(schemeParamAttr.getQ().size())) {
      linearTransform.emitOpError()
          << "expected the OpenFHE native linear transform input level to be "
             "in [0, "
          << (static_cast<int64_t>(schemeParamAttr.getQ().size()) - 1)
          << "], but found " << inputHeirLevel;
      analysisFailed = true;
      return WalkResult::interrupt();
    }

    // OpenFHE native linear transforms encode auxiliary plaintext diagonals at
    // the remaining-Q level expected by EvalLinearTransformPrecompute.
    // Non-EXT scaling techniques require HEIR to reserve the extra Q modulus
    // explicitly, so the native plaintext level matches the HEIR ciphertext
    // level after retargeting. FLEXIBLEAUTOEXT gets the extra auxiliary
    // modulus from OpenFHE itself, so its native plaintext level is one higher.
    int64_t nativePlaintextLevel =
        inputHeirLevel + static_cast<int64_t>(builtinAuxModulus);
    linearTransform->setAttr(kNativePlaintextLevelAttrName,
                             builder.getI64IntegerAttr(nativePlaintextLevel));
    return WalkResult::advance();
  });

  if (analysisFailed) {
    return failure();
  }
  return success();
}

bool inferReducedErrorSetting(ModuleOp module, StringRef scalingTechnique) {
  if (auto reducedErrorAttr =
          module->getAttrOfType<BoolAttr>(kCKKSReducedErrorAttrName)) {
    return reducedErrorAttr.getValue();
  }
  return usesReducedErrorPrimeSelection(scalingTechnique);
}

SmallVector<IntegerAttr> toIntegerAttrs(MLIRContext* ctx,
                                        ArrayRef<int64_t> elements) {
  Builder builder(ctx);
  SmallVector<IntegerAttr> attrs;
  attrs.reserve(elements.size());
  for (int64_t element : elements) {
    attrs.push_back(builder.getI64IntegerAttr(element));
  }
  return attrs;
}

FailureOr<polynomial::IntPolynomialAttr> getCyclotomicPolynomialModulus(
    MLIRContext* ctx, int64_t ringDim) {
  std::vector<polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, ringDim);
  monomials.emplace_back(1, 0);
  FailureOr<polynomial::IntPolynomial> polynomial =
      polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(polynomial)) {
    return failure();
  }
  return polynomial::IntPolynomialAttr::get(ctx, *polynomial);
}

Type retargetRlweType(Type type,
                      polynomial::IntPolynomialAttr polynomialModulus,
                      ArrayRef<IntegerAttr> newQ, int64_t delta) {
  Type elementType = getElementTypeOrSelf(type);
  MLIRContext* ctx = type.getContext();
  auto fullChain =
      lwe::ModulusChainAttr::get(ctx, newQ, static_cast<int>(newQ.size()) - 1);
  auto fullRing = lwe::getRingFromModulusChain(fullChain, polynomialModulus);

  if (auto skType = dyn_cast<lwe::LWESecretKeyType>(elementType)) {
    auto newType = lwe::LWESecretKeyType::get(ctx, skType.getKey(), fullRing);
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.clone(newType);
    }
    return newType;
  }

  if (auto pkType = dyn_cast<lwe::LWEPublicKeyType>(elementType)) {
    auto newType = lwe::LWEPublicKeyType::get(ctx, pkType.getKey(), fullRing);
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.clone(newType);
    }
    return newType;
  }

  if (auto ctType = dyn_cast<lwe::LWECiphertextType>(elementType)) {
    auto oldChain = ctType.getModulusChain();
    if (!oldChain) {
      return type;
    }

    int64_t newCurrent = static_cast<int64_t>(oldChain.getCurrent()) + delta;
    auto newChain =
        lwe::ModulusChainAttr::get(ctx, newQ, static_cast<int>(newCurrent));

    auto oldPlaintextSpace = ctType.getPlaintextSpace();
    auto newPlaintextRing = polynomial::RingAttr::get(
        oldPlaintextSpace.getRing().getCoefficientType(), polynomialModulus);
    auto newPlaintextSpace = lwe::PlaintextSpaceAttr::get(
        ctx, newPlaintextRing, oldPlaintextSpace.getEncoding());

    auto oldCiphertextSpace = ctType.getCiphertextSpace();
    auto newCiphertextRing =
        lwe::getRingFromModulusChain(newChain, polynomialModulus);
    auto newCiphertextSpace = lwe::CiphertextSpaceAttr::get(
        ctx, newCiphertextRing, oldCiphertextSpace.getEncryptionType(),
        oldCiphertextSpace.getSize());
    auto newCiphertextType = lwe::LWECiphertextType::get(
        ctx, newPlaintextSpace, newCiphertextSpace, ctType.getKey(), newChain);

    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.clone(newCiphertextType);
    }
    return newCiphertextType;
  }

  if (auto ptType = dyn_cast<lwe::LWEPlaintextType>(elementType)) {
    auto oldPlaintextSpace = ptType.getPlaintextSpace();
    auto newPlaintextRing = polynomial::RingAttr::get(
        oldPlaintextSpace.getRing().getCoefficientType(), polynomialModulus);
    auto newPlaintextSpace = lwe::PlaintextSpaceAttr::get(
        ctx, newPlaintextRing, oldPlaintextSpace.getEncoding());
    auto newPlaintextType = lwe::LWEPlaintextType::get(ctx, newPlaintextSpace);
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.clone(newPlaintextType);
    }
    return newPlaintextType;
  }

  return type;
}

FailureOr<std::unique_ptr<DataFlowSolver>> runOpenfheSchemeAdjustmentAnalyses(
    ModuleOp module, uint64_t logDefaultScale) {
  auto solver = std::make_unique<DataFlowSolver>();
  dataflow::loadBaselineAnalyses(*solver);
  solver->load<OpenfheNoiseScaleDegreeAnalysis>(logDefaultScale);
  solver->load<OpenfheEffectiveHeirLevelAnalysis>();
  if (failed(solver->initializeAndRun(module))) {
    module.emitOpError()
        << "failed to run OpenFHE CKKS depth adjustment analyses";
    return failure();
  }
  return solver;
}

void retargetModuleSchemeParam(ModuleOp module,
                               polynomial::IntPolynomialAttr polynomialModulus,
                               ArrayRef<IntegerAttr> newQ, int64_t delta) {
  auto shiftLevelAttr = [&](IntegerAttr levelAttr) -> IntegerAttr {
    return IntegerAttr::get(levelAttr.getType(), levelAttr.getInt() + delta);
  };

  auto shiftBlockArguments = [&](Block& block) {
    for (BlockArgument arg : block.getArguments()) {
      arg.setType(
          retargetRlweType(arg.getType(), polynomialModulus, newQ, delta));
    }
  };

  shiftBlockArguments(module.getBodyRegion().front());
  module.walk([&](Operation* op) {
    for (Region& region : op->getRegions()) {
      for (Block& block : region) {
        shiftBlockArguments(block);
      }
    }

    for (OpResult result : op->getResults()) {
      result.setType(
          retargetRlweType(result.getType(), polynomialModulus, newQ, delta));
    }

    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      SmallVector<Type> shiftedArgumentTypes;
      shiftedArgumentTypes.reserve(funcOp.getNumArguments());
      for (BlockArgument argument : funcOp.getArguments()) {
        shiftedArgumentTypes.push_back(argument.getType());
      }

      SmallVector<Type> shiftedResultTypes;
      shiftedResultTypes.reserve(funcOp.getNumResults());
      for (Type resultType : funcOp.getResultTypes()) {
        shiftedResultTypes.push_back(
            retargetRlweType(resultType, polynomialModulus, newQ, delta));
      }
      funcOp.setFunctionType(FunctionType::get(
          module.getContext(), shiftedArgumentTypes, shiftedResultTypes));
      return;
    }

    if (auto encodeOp = dyn_cast<lwe::RLWEEncodeOp>(op)) {
      if (auto levelAttr = encodeOp.getLevelAttr()) {
        encodeOp->setAttr("level", shiftLevelAttr(levelAttr));
      }
      return;
    }

    if (auto linearTransform = dyn_cast<orion::LinearTransformOp>(op)) {
      linearTransform->setAttr(
          "orion_level", shiftLevelAttr(linearTransform.getOrionLevelAttr()));
      return;
    }

    if (auto rescaleOp = dyn_cast<ckks::RescaleOp>(op)) {
      auto outputType = dyn_cast<lwe::LWECiphertextType>(
          getElementTypeOrSelf(rescaleOp.getOutput().getType()));
      if (!outputType) {
        return;
      }
      rescaleOp->setAttr("to_ring", outputType.getCiphertextSpace().getRing());
      return;
    }

    if (auto bootstrapOp = dyn_cast<ckks::BootstrapOp>(op)) {
      if (auto targetLevel = bootstrapOp.getTargetLevel()) {
        bootstrapOp->setAttr(
            "targetLevel",
            IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                             *targetLevel + delta));
      }
    }
  });
}

FailureOr<lbcrypto::ScalingTechnique> translateScalingTechnique(
    ModuleOp module, StringRef scalingTechnique) {
  if (scalingTechnique == kScalingTechniqueFixedManual) {
    return lbcrypto::FIXEDMANUAL;
  }
  if (scalingTechnique == kScalingTechniqueFixedAuto) {
    return lbcrypto::FIXEDAUTO;
  }
  if (scalingTechnique == kScalingTechniqueFlexibleAuto) {
    return lbcrypto::FLEXIBLEAUTO;
  }
  if (scalingTechnique == kScalingTechniqueFlexibleAutoExt) {
    return lbcrypto::FLEXIBLEAUTOEXT;
  }
  if (scalingTechnique == kScalingTechniqueCompositeAuto) {
    return lbcrypto::COMPOSITESCALINGAUTO;
  }
  if (scalingTechnique == kScalingTechniqueCompositeManual) {
    return lbcrypto::COMPOSITESCALINGMANUAL;
  }
  if (scalingTechnique == kScalingTechniqueNoRescale) {
    return lbcrypto::NORESCALE;
  }
  module.emitOpError() << "unsupported OpenFHE scaling technique `"
                       << scalingTechnique << "`";
  return failure();
}

struct OpenfheConcreteSchemeParam {
  int64_t ringDim;
  SmallVector<int64_t> q;
  SmallVector<int64_t> p;
};

FailureOr<OpenfheConcreteSchemeParam> materializeOpenfheSchemeParam(
    ModuleOp module, int64_t preferredRingDim, int64_t requestedSlotCount,
    int firstModBits, int scalingModBits, int64_t requestedChainLength,
    StringRef scalingTechnique) {
  FailureOr<lbcrypto::ScalingTechnique> openfheScalingTechnique =
      translateScalingTechnique(module, scalingTechnique);
  if (failed(openfheScalingTechnique)) {
    return failure();
  }

  auto tryBuild =
      [&](int64_t ringDim) -> FailureOr<OpenfheConcreteSchemeParam> {
    try {
      lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
      parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
      parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
      parameters.SetScalingTechnique(*openfheScalingTechnique);
      parameters.SetFirstModSize(static_cast<uint32_t>(firstModBits));
      parameters.SetScalingModSize(static_cast<uint32_t>(scalingModBits));
      parameters.SetMultiplicativeDepth(
          static_cast<uint32_t>(requestedChainLength - 1));
      if (requestedSlotCount > 0) {
        parameters.SetBatchSize(static_cast<uint32_t>(requestedSlotCount));
      }
      if (ringDim > 0) {
        parameters.SetRingDim(static_cast<uint32_t>(ringDim));
      }

      auto cryptoContext = lbcrypto::GenCryptoContext(parameters);
      auto cryptoParams =
          std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(
              cryptoContext->GetCryptoParameters());
      if (!cryptoParams) {
        module.emitOpError()
            << "failed to read generated OpenFHE CKKS parameters";
        return failure();
      }

      OpenfheConcreteSchemeParam result;
      result.ringDim = static_cast<int64_t>(cryptoContext->GetRingDimension());
      for (const auto& modulusParam :
           cryptoParams->GetElementParams()->GetParams()) {
        result.q.push_back(
            static_cast<int64_t>(modulusParam->GetModulus().ConvertToInt()));
      }
      if (auto paramsP = cryptoParams->GetParamsP()) {
        for (const auto& modulusParam : paramsP->GetParams()) {
          result.p.push_back(
              static_cast<int64_t>(modulusParam->GetModulus().ConvertToInt()));
        }
      }
      return result;
    } catch (const std::exception&) {
      return failure();
    }
  };

  if (preferredRingDim > 0) {
    auto concrete = tryBuild(preferredRingDim);
    if (succeeded(concrete)) {
      return *concrete;
    }
  }
  auto concrete = tryBuild(/*ringDim=*/0);
  if (succeeded(concrete)) {
    return *concrete;
  }

  module.emitOpError()
      << "failed to materialize an OpenFHE-compatible CKKS scheme parameter";
  return failure();
}

}  // namespace

#define GEN_PASS_DEF_ADJUSTCKKSSCHEMEPARAM
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

struct AdjustCKKSSchemeParam
    : impl::AdjustCKKSSchemeParamBase<AdjustCKKSSchemeParam> {
  using AdjustCKKSSchemeParamBase::AdjustCKKSSchemeParamBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    if (!moduleIsCKKS(module)) {
      return;
    }

    auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      return;
    }

    StringRef concreteScalingTechnique =
        resolveScalingTechnique(scalingTechnique);

    if (!isSupportedScalingTechnique(concreteScalingTechnique)) {
      module.emitOpError() << "unsupported OpenFHE scaling technique `"
                           << concreteScalingTechnique << "`";
      signalPassFailure();
      return;
    }

    Builder builder(module.getContext());
    module->setAttr(kScalingTechniqueAttrName,
                    builder.getStringAttr(concreteScalingTechnique));

    FailureOr<std::unique_ptr<DataFlowSolver>> solver =
        runOpenfheSchemeAdjustmentAnalyses(
            module, schemeParamAttr.getLogDefaultScale());
    if (failed(solver)) {
      signalPassFailure();
      return;
    }

    int64_t currentChainLength =
        static_cast<int64_t>(schemeParamAttr.getQ().size());
    int64_t maxLevel = currentChainLength - 1;
    int64_t requiredChainLength =
        getRequiredOpenfheChainLength(module, *solver.value(), maxLevel);
    int64_t targetChainLength =
        std::max(requiredChainLength, currentChainLength);
    if (containsOpaqueLinearTransform(module) &&
        !usesBuiltInAuxPlaintextModulus(concreteScalingTechnique)) {
      // OpenFHE native linear transforms encode auxiliary plaintext diagonals
      // one Q modulus above the corresponding HEIR ciphertext level. All
      // current scaling techniques except FLEXIBLEAUTOEXT require HEIR to
      // reserve that extra modulus explicitly in the Q chain.
      targetChainLength = std::max(targetChainLength, currentChainLength + 1);
    }
    LLVM_DEBUG({
      llvm::dbgs() << "OpenFHE scheme param adjustment: current chain length="
                   << currentChainLength
                   << ", required chain length=" << requiredChainLength
                   << ", max HEIR level=" << maxLevel << "\n";
    });

    auto firstQ = static_cast<uint64_t>(schemeParamAttr.getQ()[0]);
    int firstModBits =
        std::numeric_limits<uint64_t>::digits - llvm::countl_zero(firstQ);

    int64_t requestedSlotCount =
        (static_cast<int64_t>(1) << schemeParamAttr.getLogN()) / 2;
    if (auto requestedSlotCountAttr =
            module->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName)) {
      requestedSlotCount = requestedSlotCountAttr.getInt();
    }

    bool reducedError =
        inferReducedErrorSetting(module, concreteScalingTechnique);

    FailureOr<OpenfheConcreteSchemeParam> concreteSchemeParam =
        materializeOpenfheSchemeParam(
            module, static_cast<int64_t>(1) << schemeParamAttr.getLogN(),
            requestedSlotCount, firstModBits,
            schemeParamAttr.getLogDefaultScale(), targetChainLength,
            concreteScalingTechnique);
    if (failed(concreteSchemeParam)) {
      signalPassFailure();
      return;
    }

    auto newQ = toIntegerAttrs(module.getContext(), concreteSchemeParam->q);
    auto newP = toIntegerAttrs(module.getContext(), concreteSchemeParam->p);
    int64_t delta = static_cast<int64_t>(concreteSchemeParam->q.size()) -
                    currentChainLength;
    FailureOr<polynomial::IntPolynomialAttr> polynomialModulus =
        getCyclotomicPolynomialModulus(module.getContext(),
                                       concreteSchemeParam->ringDim);
    if (failed(polynomialModulus)) {
      module.emitOpError()
          << "failed to build the updated cyclotomic polynomial modulus for "
             "OpenFHE CKKS scheme adjustment";
      signalPassFailure();
      return;
    }
    retargetModuleSchemeParam(module, *polynomialModulus, newQ, delta);
    module->setAttr(
        ckks::CKKSDialect::kSchemeParamAttrName,
        ckks::SchemeParamAttr::get(
            module.getContext(), llvm::Log2_64(concreteSchemeParam->ringDim),
            DenseI64ArrayAttr::get(module.getContext(),
                                   ArrayRef(concreteSchemeParam->q)),
            DenseI64ArrayAttr::get(module.getContext(),
                                   ArrayRef(concreteSchemeParam->p)),
            schemeParamAttr.getLogDefaultScale(),
            schemeParamAttr.getEncryptionType(),
            schemeParamAttr.getEncryptionTechnique()));
    module->setAttr(kRequestedSlotCountAttrName,
                    builder.getI64IntegerAttr(requestedSlotCount));
    module->setAttr(
        kActualSlotCountAttrName,
        builder.getI64IntegerAttr(concreteSchemeParam->ringDim / 2));
    module->setAttr(kCKKSReducedErrorAttrName,
                    builder.getBoolAttr(reducedError));
    solver = runOpenfheSchemeAdjustmentAnalyses(
        module, schemeParamAttr.getLogDefaultScale());
    if (failed(solver)) {
      signalPassFailure();
      return;
    }
    if (failed(resolveOpaqueLinearTransformNativeLevels(module,
                                                        *solver.value()))) {
      signalPassFailure();
      return;
    }
    if (failed(resolveAddPlainEncodes(module, *solver.value(),
                                      schemeParamAttr.getLogDefaultScale()))) {
      signalPassFailure();
    }
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
