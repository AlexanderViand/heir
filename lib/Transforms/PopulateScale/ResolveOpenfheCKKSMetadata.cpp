#include "lib/Transforms/PopulateScale/ResolveOpenfheCKKSMetadata.h"

#include <algorithm>
#include <cmath>
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

#define DEBUG_TYPE "resolve-openfhe-ckks-metadata"

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

FailureOr<int64_t> inferPredictiveNoiseScaleDegreeFromType(
    Operation* op, Type type, uint64_t logDefaultScale,
    StringRef scalingTechnique, int64_t maxHeirLevel) {
  FailureOr<int64_t> baseDegree =
      inferNoiseScaleDegreeFromType(op, type, logDefaultScale);
  if (failed(baseDegree)) {
    return failure();
  }

  if (resolveScalingTechnique(scalingTechnique) !=
      kScalingTechniqueFlexibleAutoExt) {
    return *baseDegree;
  }

  auto ctType = dyn_cast<lwe::LWECiphertextType>(getElementTypeOrSelf(type));
  if (!ctType || !ctType.getModulusChain()) {
    return *baseDegree;
  }

  if (static_cast<int64_t>(ctType.getModulusChain().getCurrent()) !=
      maxHeirLevel) {
    return *baseDegree;
  }

  // OpenFHE FLEXIBLEAUTOEXT forces freshly encoded level-0 plaintexts to
  // runtime noise-scale degree 2. Encrypt copies that metadata into fresh
  // ciphertexts, so top-of-chain ciphertext values also start at degree 2.
  return std::max<int64_t>(*baseDegree, 2);
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

class OpenfhePredictiveNoiseScaleDegreeLattice
    : public dataflow::Lattice<OpenfheNoiseScaleDegreeState> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OpenfhePredictiveNoiseScaleDegreeLattice)
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

class OpenfheScalingBitsState {
 public:
  OpenfheScalingBitsState() = default;
  explicit OpenfheScalingBitsState(double bits) : bits(bits) {}

  bool isInitialized() const { return bits.has_value(); }
  double getBits() const {
    assert(isInitialized());
    return *bits;
  }

  bool operator==(const OpenfheScalingBitsState& rhs) const = default;

  static OpenfheScalingBitsState join(const OpenfheScalingBitsState& lhs,
                                      const OpenfheScalingBitsState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    return OpenfheScalingBitsState(std::max(lhs.getBits(), rhs.getBits()));
  }

  void print(llvm::raw_ostream& os) const {
    if (isInitialized()) {
      os << "OpenfheScalingBitsState(" << *bits << ")";
      return;
    }
    os << "OpenfheScalingBitsState(uninitialized)";
  }

 private:
  std::optional<double> bits;
};

class OpenfhePredictiveScalingBitsLattice
    : public dataflow::Lattice<OpenfheScalingBitsState> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OpenfhePredictiveScalingBitsLattice)
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

FailureOr<double> inferOpenfheCiphertextScalingBitsFromType(
    Operation* op, Type type, uint64_t logDefaultScale,
    StringRef scalingTechnique, int64_t maxHeirLevel,
    ArrayRef<double> realScalingBits, ArrayRef<double> bigScalingBits) {
  auto ctType = dyn_cast<lwe::LWECiphertextType>(getElementTypeOrSelf(type));
  if (!ctType || !ctType.getModulusChain()) {
    return failure();
  }

  int64_t heirLevel =
      static_cast<int64_t>(ctType.getModulusChain().getCurrent());
  FailureOr<int64_t> predictiveDegree = inferPredictiveNoiseScaleDegreeFromType(
      op, type, logDefaultScale, scalingTechnique, maxHeirLevel);
  if (failed(predictiveDegree)) {
    return failure();
  }

  int64_t openfheLevel = maxHeirLevel - heirLevel;
  if (openfheLevel < 0 ||
      openfheLevel >= static_cast<int64_t>(realScalingBits.size())) {
    op->emitOpError() << "requires OpenFHE scaling bits for level "
                      << openfheLevel;
    return failure();
  }

  StringRef resolved = resolveScalingTechnique(scalingTechnique);
  if (resolved == kScalingTechniqueFlexibleAutoExt && openfheLevel == 0 &&
      *predictiveDegree >= 2) {
    if (bigScalingBits.empty()) {
      op->emitOpError()
          << "requires OpenFHE FLEXIBLEAUTOEXT big scaling bits at level 0";
      return failure();
    }
    return bigScalingBits.front();
  }

  return realScalingBits[openfheLevel] *
         static_cast<double>(std::max<int64_t>(*predictiveDegree, 1));
}

FailureOr<double> deriveOpenfhePredictiveScalingBits(
    Operation* op, ArrayRef<double> secretOperandScalingBits,
    ArrayRef<int64_t> secretOperandLevels,
    ArrayRef<int64_t> secretOperandDegrees, StringRef scalingTechnique,
    int64_t maxHeirLevel, ArrayRef<double> realScalingBits) {
  auto maxSecretOperandBits = [&]() -> FailureOr<double> {
    if (secretOperandScalingBits.empty()) {
      return op->emitOpError()
             << "requires at least one secret operand to derive the "
                "OpenFHE predictive scaling factor";
    }
    return *std::max_element(secretOperandScalingBits.begin(),
                             secretOperandScalingBits.end());
  };

  if (isa<ckks::AddOp, ckks::SubOp, ckks::AddPlainOp, ckks::SubPlainOp,
          ckks::RotateOp, ckks::RelinearizeOp, ckks::NegateOp>(op)) {
    return maxSecretOperandBits();
  }

  if (auto mulOp = dyn_cast<ckks::MulOp>(op)) {
    if (secretOperandScalingBits.size() != 2) {
      return mulOp.emitOpError()
             << "expected exactly two secret operands for ckks.mul";
    }
    return secretOperandScalingBits[0] + secretOperandScalingBits[1];
  }

  if (auto rescaleOp = dyn_cast<ckks::RescaleOp>(op)) {
    if (secretOperandScalingBits.size() != 1) {
      return rescaleOp.emitOpError()
             << "expected exactly one secret operand for ckks.rescale";
    }
    return secretOperandScalingBits[0];
  }

  if (auto levelReduceOp = dyn_cast<ckks::LevelReduceOp>(op)) {
    if (secretOperandScalingBits.size() != 1) {
      return levelReduceOp.emitOpError()
             << "expected exactly one secret operand for ckks.level_reduce";
    }
    return secretOperandScalingBits[0];
  }

  if (auto linearTransform = dyn_cast<orion::LinearTransformOp>(op)) {
    if (secretOperandScalingBits.size() != 1 ||
        secretOperandLevels.size() != 1 || secretOperandDegrees.size() != 1) {
      return linearTransform.emitOpError()
             << "expected exactly one secret operand for "
                "orion.linear_transform";
    }
    auto implStyle =
        linearTransform->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
    if (!implStyle) {
      return linearTransform.emitOpError()
             << "requires `orion.impl_style` before OpenFHE scaling analysis";
    }
    if (implStyle.getValue() != orion::kOpaqueImplStyle) {
      return linearTransform.emitOpError()
             << "OpenFHE scaling analysis expects opaque Orion linear "
                "transforms before backend lowering";
    }
    int64_t towersToDrop = maxHeirLevel - secretOperandLevels.front();
    if (towersToDrop < 0 ||
        towersToDrop >= static_cast<int64_t>(realScalingBits.size())) {
      return linearTransform.emitOpError()
             << "computed invalid OpenFHE linear transform towersToDrop="
             << towersToDrop;
    }
    return secretOperandScalingBits.front() + realScalingBits[towersToDrop];
  }

  if (auto chebyshev = dyn_cast<orion::ChebyshevOp>(op)) {
    if (failed(orion::verifyImplStyle(chebyshev, orion::kOpaqueImplStyle))) {
      return failure();
    }
    if (secretOperandScalingBits.size() != 1 ||
        secretOperandLevels.size() != 1 || secretOperandDegrees.size() != 1) {
      return chebyshev.emitOpError()
             << "expected exactly one secret operand for orion.chebyshev";
    }
    auto levelCostAttr = chebyshev->getAttrOfType<IntegerAttr>(
        orion::kLevelCostUpperBoundAttrName);
    if (!levelCostAttr) {
      return chebyshev.emitOpError()
             << "requires `orion.level_cost_ub` before OpenFHE scaling "
                "analysis";
    }
    int64_t levelCost = levelCostAttr.getInt();
    int64_t inputOpenFHELevel = maxHeirLevel - secretOperandLevels.front();
    double accumulated = secretOperandScalingBits.front();
    for (int64_t i = 0; i < levelCost; ++i) {
      int64_t stepLevel = inputOpenFHELevel + i;
      if (stepLevel < 0 ||
          stepLevel >= static_cast<int64_t>(realScalingBits.size())) {
        return chebyshev.emitOpError()
               << "computed invalid OpenFHE chebyshev step level=" << stepLevel
               << " at internal step " << i;
      }
      accumulated += realScalingBits[stepLevel];
    }
    return accumulated;
  }

  return maxSecretOperandBits();
}

FailureOr<int64_t> deriveOpenfheNoiseScaleDegree(
    Operation* op, ArrayRef<int64_t> secretOperandDegrees,
    ArrayRef<Type> plaintextOperandTypes, uint64_t logDefaultScale,
    StringRef scalingTechnique) {
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
    if (usesPredictiveLevelState(scalingTechnique) &&
        *std::max_element(secretOperandDegrees.begin(),
                          secretOperandDegrees.end()) == 2) {
      return int64_t{2};
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
    if (!usesExplicitPublicLevelManagement(scalingTechnique)) {
      return secretOperandDegrees[0];
    }
    return std::max<int64_t>(1, secretOperandDegrees[0] - 1);
  }

  if (auto opLevelReduce = dyn_cast<ckks::LevelReduceOp>(op)) {
    if (secretOperandDegrees.size() != 1) {
      return opLevelReduce.emitOpError()
             << "expected exactly one secret operand for ckks.level_reduce";
    }
    if (!usesExplicitPublicLevelManagement(scalingTechnique)) {
      return secretOperandDegrees[0];
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
    Operation* op, ArrayRef<int64_t> secretOperandLevels,
    ArrayRef<int64_t> secretOperandDegrees, StringRef scalingTechnique) {
  // This models the runtime ciphertext meeting level seen by native OpenFHE
  // operations, not the compiler's predictive CKKS management markers. In the
  // predictive OpenFHE modes (`flexible-auto` and
  // `flexible-auto-ext`), explicit `ckks.rescale` / `ckks.level_reduce`
  // markers are compiler structure only; the runtime Level changes only when
  // OpenFHE's own internal adjustment code drops towers.
  auto minSecretOperandLevel = [&]() -> FailureOr<int64_t> {
    if (secretOperandLevels.empty()) {
      return op->emitOpError()
             << "requires at least one secret operand to derive the "
                "OpenFHE-effective level";
    }
    return *std::min_element(secretOperandLevels.begin(),
                             secretOperandLevels.end());
  };

  auto getExplicitPlaintextEncodeHeirLevel = [&]() -> std::optional<int64_t> {
    for (Value operand : op->getOperands()) {
      if (!isa<lwe::LWEPlaintextType>(
              getElementTypeOrSelf(operand.getType()))) {
        continue;
      }
      if (auto encodeOp = operand.getDefiningOp<lwe::RLWEEncodeOp>()) {
        if (auto levelAttr = encodeOp.getLevelAttr()) {
          return levelAttr.getInt();
        }
      }
    }
    return std::nullopt;
  };

  if (isa<ckks::AddPlainOp, ckks::SubPlainOp>(op)) {
    FailureOr<int64_t> level = minSecretOperandLevel();
    if (failed(level)) {
      return failure();
    }
    if (!usesPredictiveLevelState(scalingTechnique)) {
      return *level;
    }
    if (auto plaintextHeirLevel = getExplicitPlaintextEncodeHeirLevel()) {
      return std::min(*level, *plaintextHeirLevel);
    }
    return *level;
  }

  if (isa<ckks::AddOp, ckks::SubOp, ckks::RotateOp, ckks::RelinearizeOp,
          ckks::NegateOp, ckks::MulPlainOp>(op)) {
    return minSecretOperandLevel();
  }

  if (isa<ckks::MulOp>(op)) {
    FailureOr<int64_t> level = minSecretOperandLevel();
    if (failed(level)) {
      return failure();
    }
    if (!usesPredictiveLevelState(scalingTechnique)) {
      return *level;
    }
    if (secretOperandDegrees.empty()) {
      return *level;
    }
    int64_t alignedDegree = *std::max_element(secretOperandDegrees.begin(),
                                              secretOperandDegrees.end());
    if (alignedDegree == 2) {
      return *level - 1;
    }
    return *level;
  }

  if (auto opRescale = dyn_cast<ckks::RescaleOp>(op)) {
    FailureOr<int64_t> level = minSecretOperandLevel();
    if (failed(level)) {
      return failure();
    }
    if (!usesExplicitPublicLevelManagement(scalingTechnique)) {
      return *level;
    }
    return *level - 1;
  }

  if (auto opLevelReduce = dyn_cast<ckks::LevelReduceOp>(op)) {
    FailureOr<int64_t> level = minSecretOperandLevel();
    if (failed(level)) {
      return failure();
    }
    if (!usesExplicitPublicLevelManagement(scalingTechnique)) {
      return *level;
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
        op, ciphertextOperandDegrees, plaintextOperandTypes, logDefaultScale,
        /*scalingTechnique=*/kScalingTechniqueFixedManual);
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

class OpenfhePredictiveNoiseScaleDegreeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          OpenfhePredictiveNoiseScaleDegreeLattice> {
 public:
  OpenfhePredictiveNoiseScaleDegreeAnalysis(DataFlowSolver& solver,
                                            uint64_t logDefaultScale,
                                            StringRef scalingTechnique,
                                            int64_t maxHeirLevel)
      : SparseForwardDataFlowAnalysis(solver),
        logDefaultScale(logDefaultScale),
        scalingTechnique(scalingTechnique.str()),
        maxHeirLevel(maxHeirLevel) {}

  void setToEntryState(
      OpenfhePredictiveNoiseScaleDegreeLattice* lattice) override {
    Value anchor = lattice->getAnchor();
    Type type = anchor.getType();
    if (!isCiphertextType(type) && !isPlaintextType(type)) {
      return;
    }
    FailureOr<int64_t> degree = inferPredictiveNoiseScaleDegreeFromType(
        getOwningOperation(anchor), type, logDefaultScale, scalingTechnique,
        maxHeirLevel);
    if (failed(degree)) {
      return;
    }
    propagateIfChanged(lattice,
                       lattice->join(OpenfheNoiseScaleDegreeState(*degree)));
  }

  LogicalResult visitOperation(
      Operation* op,
      ArrayRef<const OpenfhePredictiveNoiseScaleDegreeLattice*> operands,
      ArrayRef<OpenfhePredictiveNoiseScaleDegreeLattice*> results) override {
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
        op, ciphertextOperandDegrees, plaintextOperandTypes, logDefaultScale,
        scalingTechnique);
    if (failed(resultDegree)) {
      return failure();
    }

    for (OpResult result : ciphertextResults) {
      auto* lattice = getLatticeElement(result);
      propagateIfChanged(
          lattice, lattice->join(OpenfheNoiseScaleDegreeState(*resultDegree)));
    }
    return success();
  }

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const OpenfhePredictiveNoiseScaleDegreeLattice*>
          argumentLattices,
      ArrayRef<OpenfhePredictiveNoiseScaleDegreeLattice*> resultLattices)
      override {
    auto callback = [&](AnalysisState* state, ChangeResult changed) {
      propagateIfChanged(state, changed);
    };
    ::mlir::heir::visitExternalCall<OpenfheNoiseScaleDegreeState,
                                    OpenfhePredictiveNoiseScaleDegreeLattice>(
        call, argumentLattices, resultLattices, callback);
  }

 private:
  uint64_t logDefaultScale;
  std::string scalingTechnique;
  int64_t maxHeirLevel;
};

class OpenfheEffectiveHeirLevelAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          OpenfheEffectiveHeirLevelLattice> {
 public:
  OpenfheEffectiveHeirLevelAnalysis(DataFlowSolver& solver,
                                    StringRef scalingTechnique)
      : SparseForwardDataFlowAnalysis(solver),
        solver(solver),
        scalingTechnique(scalingTechnique.str()) {}

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
    SmallVector<int64_t> ciphertextOperandDegrees;
    for (OpOperand* operand : ciphertextOperands) {
      auto* lattice = getLatticeElement(operand->get());
      if (!lattice || !lattice->getValue().isInitialized()) {
        continue;
      }
      ciphertextOperandLevels.push_back(lattice->getValue().getLevel());
      auto* degreeLattice =
          solver.lookupState<OpenfhePredictiveNoiseScaleDegreeLattice>(
              operand->get());
      if (degreeLattice && degreeLattice->getValue().isInitialized()) {
        ciphertextOperandDegrees.push_back(
            degreeLattice->getValue().getDegree());
      }
    }

    FailureOr<int64_t> resultLevel = deriveOpenfheEffectiveHeirLevel(
        op, ciphertextOperandLevels, ciphertextOperandDegrees,
        scalingTechnique);
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

 private:
  DataFlowSolver& solver;
  std::string scalingTechnique;
};

class OpenfhePredictiveScalingBitsAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          OpenfhePredictiveScalingBitsLattice> {
 public:
  OpenfhePredictiveScalingBitsAnalysis(DataFlowSolver& solver,
                                       uint64_t logDefaultScale,
                                       StringRef scalingTechnique,
                                       int64_t maxHeirLevel,
                                       ArrayRef<double> realScalingBits,
                                       ArrayRef<double> bigScalingBits)
      : SparseForwardDataFlowAnalysis(solver),
        solver(solver),
        logDefaultScale(logDefaultScale),
        scalingTechnique(scalingTechnique.str()),
        maxHeirLevel(maxHeirLevel),
        realScalingBits(realScalingBits.begin(), realScalingBits.end()),
        bigScalingBits(bigScalingBits.begin(), bigScalingBits.end()) {}

  void setToEntryState(OpenfhePredictiveScalingBitsLattice* lattice) override {
    Value anchor = lattice->getAnchor();
    auto blockArgument = dyn_cast<BlockArgument>(anchor);
    if (!blockArgument || !isCiphertextType(anchor.getType())) {
      return;
    }
    FailureOr<double> bits = inferOpenfheCiphertextScalingBitsFromType(
        getOwningOperation(anchor), anchor.getType(), logDefaultScale,
        scalingTechnique, maxHeirLevel, realScalingBits, bigScalingBits);
    if (failed(bits)) {
      return;
    }
    propagateIfChanged(lattice, lattice->join(OpenfheScalingBitsState(*bits)));
  }

  LogicalResult visitOperation(
      Operation* op,
      ArrayRef<const OpenfhePredictiveScalingBitsLattice*> operands,
      ArrayRef<OpenfhePredictiveScalingBitsLattice*> results) override {
    SmallVector<OpResult> ciphertextResults;
    getCiphertextResults(op, ciphertextResults);
    if (ciphertextResults.empty()) {
      return success();
    }

    SmallVector<OpOperand*> ciphertextOperands;
    getCiphertextOperands(op, ciphertextOperands);
    SmallVector<double> ciphertextOperandScalingBits;
    SmallVector<int64_t> ciphertextOperandLevels;
    SmallVector<int64_t> ciphertextOperandDegrees;
    for (OpOperand* operand : ciphertextOperands) {
      auto* scaleLattice = getLatticeElement(operand->get());
      if (scaleLattice && scaleLattice->getValue().isInitialized()) {
        ciphertextOperandScalingBits.push_back(
            scaleLattice->getValue().getBits());
      }
      auto* levelLattice =
          solver.lookupState<OpenfheEffectiveHeirLevelLattice>(operand->get());
      if (levelLattice && levelLattice->getValue().isInitialized()) {
        ciphertextOperandLevels.push_back(levelLattice->getValue().getLevel());
      }
      auto* degreeLattice =
          solver.lookupState<OpenfhePredictiveNoiseScaleDegreeLattice>(
              operand->get());
      if (degreeLattice && degreeLattice->getValue().isInitialized()) {
        ciphertextOperandDegrees.push_back(
            degreeLattice->getValue().getDegree());
      }
    }

    FailureOr<double> resultBits = deriveOpenfhePredictiveScalingBits(
        op, ciphertextOperandScalingBits, ciphertextOperandLevels,
        ciphertextOperandDegrees, scalingTechnique, maxHeirLevel,
        realScalingBits);
    if (failed(resultBits)) {
      return failure();
    }

    for (OpResult result : ciphertextResults) {
      auto* lattice = getLatticeElement(result);
      propagateIfChanged(lattice,
                         lattice->join(OpenfheScalingBitsState(*resultBits)));
    }
    return success();
  }

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const OpenfhePredictiveScalingBitsLattice*> argumentLattices,
      ArrayRef<OpenfhePredictiveScalingBitsLattice*> resultLattices) override {
    auto callback = [&](AnalysisState* state, ChangeResult changed) {
      propagateIfChanged(state, changed);
    };
    ::mlir::heir::visitExternalCall<OpenfheScalingBitsState,
                                    OpenfhePredictiveScalingBitsLattice>(
        call, argumentLattices, resultLattices, callback);
  }

 private:
  DataFlowSolver& solver;
  uint64_t logDefaultScale;
  std::string scalingTechnique;
  int64_t maxHeirLevel;
  SmallVector<double> realScalingBits;
  SmallVector<double> bigScalingBits;
};

int64_t getRequiredOpenfheChainLength(Operation* top, DataFlowSolver& solver,
                                      int64_t maxHeirLevel) {
  int64_t requiredChainLength = 0;
  walkValues(top, [&](Value value) {
    auto* degreeLattice =
        solver.lookupState<OpenfhePredictiveNoiseScaleDegreeLattice>(value);
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
                                   int64_t requiredHeirLevel,
                                   std::optional<double> requiredScalingBits) {
  encodeOp->setAttr(
      "level", IntegerAttr::get(IntegerType::get(encodeOp.getContext(), 64),
                                requiredHeirLevel));
  encodeOp->setAttr(
      kNoiseScaleDegreeAttrName,
      IntegerAttr::get(IntegerType::get(encodeOp.getContext(), 64),
                       requiredNoiseScaleDegree));
  if (requiredScalingBits) {
    encodeOp->setAttr(kScalingFactorBitsAttrName,
                      FloatAttr::get(Float64Type::get(encodeOp.getContext()),
                                     *requiredScalingBits));
  } else {
    encodeOp->removeAttr(kScalingFactorBitsAttrName);
  }
  return success();
}

FailureOr<bool> resolveAddPlainEncodes(ModuleOp module, DataFlowSolver& solver,
                                       uint64_t logDefaultScale,
                                       StringRef scalingTechnique) {
  (void)logDefaultScale;
  auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
      ckks::CKKSDialect::kSchemeParamAttrName);
  if (!schemeParamAttr) {
    module.emitOpError()
        << "requires ckks.schemeParam to resolve OpenFHE plaintext add/sub "
           "encodes";
    return failure();
  }
  bool analysisFailed = false;
  bool changed = false;
  struct EncodeRewrite {
    Operation* user;
    unsigned operandNumber;
    lwe::RLWEEncodeOp encodeOp;
    int64_t requiredNoiseScaleDegree;
    int64_t requiredHeirLevel;
    std::optional<double> requiredScalingBits;
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
            solver.lookupState<OpenfhePredictiveNoiseScaleDegreeLattice>(
                ciphertext);
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

        int64_t requiredNoiseScaleDegree =
            degreeLattice->getValue().getDegree();
        int64_t requiredHeirLevel = levelLattice->getValue().getLevel();
        std::optional<double> requiredScalingBits;
        if (resolveScalingTechnique(scalingTechnique) ==
            kScalingTechniqueFlexibleAutoExt) {
          auto* scalingBitsLattice =
              solver.lookupState<OpenfhePredictiveScalingBitsLattice>(
                  ciphertext);
          if (!scalingBitsLattice ||
              !scalingBitsLattice->getValue().isInitialized()) {
            op->emitOpError()
                << "requires resolved OpenFHE predictive scaling bits for "
                   "FLEXIBLEAUTOEXT plaintext add/sub conversion";
            analysisFailed = true;
            return;
          }
          requiredScalingBits = scalingBitsLattice->getValue().getBits();
        }

        for (OpOperand& operand : op->getOpOperands()) {
          if (operand.get() != plaintext) {
            continue;
          }
          rewrites.push_back(EncodeRewrite{op, operand.getOperandNumber(),
                                           encodeOp, requiredNoiseScaleDegree,
                                           requiredHeirLevel,
                                           requiredScalingBits});
          break;
        }
      });

  if (analysisFailed) {
    return failure();
  }

  struct AssignedEncodeRequirement {
    int64_t noiseScaleDegree;
    int64_t heirLevel;
    std::optional<double> scalingBits;

    bool operator==(const AssignedEncodeRequirement&) const = default;
  };
  DenseMap<Operation*, AssignedEncodeRequirement> assignedRequirements;
  for (auto rewrite : rewrites) {
    Operation* user = rewrite.user;
    unsigned operandNumber = rewrite.operandNumber;
    lwe::RLWEEncodeOp encodeOp = rewrite.encodeOp;
    int64_t requiredNoiseScaleDegree = rewrite.requiredNoiseScaleDegree;
    int64_t requiredHeirLevel = rewrite.requiredHeirLevel;
    std::optional<double> requiredScalingBits = rewrite.requiredScalingBits;
    AssignedEncodeRequirement requirement{
        requiredNoiseScaleDegree, requiredHeirLevel, requiredScalingBits};
    if (!encodeOp.getOutput().hasOneUse()) {
      OpBuilder builder(user);
      auto clonedEncode = lwe::RLWEEncodeOp::create(
          builder, encodeOp.getLoc(), encodeOp.getOutput().getType(),
          encodeOp.getInput(), encodeOp.getEncoding(), encodeOp.getRing(),
          builder.getI64IntegerAttr(requiredHeirLevel));
      if (failed(setEncodeRequirement(clonedEncode, requiredNoiseScaleDegree,
                                      requiredHeirLevel,
                                      requiredScalingBits))) {
        return failure();
      }
      user->setOperand(operandNumber, clonedEncode.getOutput());
      changed = true;
      continue;
    }

    auto it = assignedRequirements.find(encodeOp.getOperation());
    if (it == assignedRequirements.end()) {
      auto oldLevel =
          encodeOp.getLevelAttr()
              ? std::optional<int64_t>(encodeOp.getLevelAttr().getInt())
              : std::nullopt;
      auto oldDegreeAttr =
          encodeOp->getAttrOfType<IntegerAttr>(kNoiseScaleDegreeAttrName);
      auto oldDegree = oldDegreeAttr
                           ? std::optional<int64_t>(oldDegreeAttr.getInt())
                           : std::nullopt;
      auto oldScalingBitsAttr =
          encodeOp->getAttrOfType<FloatAttr>(kScalingFactorBitsAttrName);
      auto oldScalingBits =
          oldScalingBitsAttr
              ? std::optional<double>(oldScalingBitsAttr.getValueAsDouble())
              : std::nullopt;
      if (failed(setEncodeRequirement(encodeOp, requiredNoiseScaleDegree,
                                      requiredHeirLevel,
                                      requiredScalingBits))) {
        return failure();
      }
      if (oldLevel != std::optional<int64_t>(requiredHeirLevel) ||
          oldDegree != std::optional<int64_t>(requiredNoiseScaleDegree) ||
          oldScalingBits != requiredScalingBits) {
        changed = true;
      }
      assignedRequirements.try_emplace(encodeOp.getOperation(), requirement);
      continue;
    }

    if (it->second == requirement) {
      continue;
    }

    encodeOp.emitOpError()
        << "single-use plaintext encode received conflicting OpenFHE "
           "requirements";
    return failure();
  }

  return changed;
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

    // OpenFHE's linear-transform precompute contract is mode-sensitive.
    // HEIR levels track the remaining public Q chain (`current` in the
    // modulus chain attr), but FLEXIBLEAUTOEXT uses a one-step-shifted
    // auxiliary plaintext convention in OpenFHE's own precompute paths.
    StringRef scalingTechnique = {};
    if (auto scalingTechniqueAttr =
            module->getAttrOfType<StringAttr>(kScalingTechniqueAttrName)) {
      scalingTechnique = scalingTechniqueAttr.getValue();
    }
    int64_t nativePlaintextLevel = inputHeirLevel;
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
    ModuleOp module, uint64_t logDefaultScale, StringRef scalingTechnique,
    ArrayRef<double> realScalingBits, ArrayRef<double> bigScalingBits) {
  auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
      ckks::CKKSDialect::kSchemeParamAttrName);
  if (!schemeParamAttr) {
    module.emitOpError()
        << "requires ckks.schemeParam to run OpenFHE metadata analyses";
    return failure();
  }
  int64_t maxHeirLevel =
      static_cast<int64_t>(schemeParamAttr.getQ().size()) - 1;

  auto solver = std::make_unique<DataFlowSolver>();
  dataflow::loadBaselineAnalyses(*solver);
  solver->load<OpenfheNoiseScaleDegreeAnalysis>(logDefaultScale);
  solver->load<OpenfhePredictiveNoiseScaleDegreeAnalysis>(
      logDefaultScale, scalingTechnique, maxHeirLevel);
  solver->load<OpenfheEffectiveHeirLevelAnalysis>(scalingTechnique);
  if (resolveScalingTechnique(scalingTechnique) ==
      kScalingTechniqueFlexibleAutoExt) {
    solver->load<OpenfhePredictiveScalingBitsAnalysis>(
        logDefaultScale, scalingTechnique, maxHeirLevel, realScalingBits,
        bigScalingBits);
  }
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
  if (scalingTechnique == kScalingTechniqueFlexibleAuto) {
    return lbcrypto::FLEXIBLEAUTO;
  }
  if (scalingTechnique == kScalingTechniqueFlexibleAutoExt) {
    return lbcrypto::FLEXIBLEAUTOEXT;
  }
  module.emitOpError() << "unsupported OpenFHE scaling technique `"
                       << scalingTechnique << "`";
  return failure();
}

struct OpenfheConcreteSchemeParam {
  int64_t ringDim;
  SmallVector<int64_t> q;
  SmallVector<int64_t> p;
  SmallVector<double> realScalingBits;
  SmallVector<double> bigScalingBits;
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

  auto requestedMulDepth = [&]() -> FailureOr<uint32_t> {
    int64_t overhead =
        scalingTechnique == kScalingTechniqueFlexibleAutoExt ? 2 : 1;
    if (requestedChainLength < overhead) {
      module.emitOpError() << "requested OpenFHE CKKS chain length "
                           << requestedChainLength
                           << " is too small for scaling technique `"
                           << scalingTechnique << "`";
      return failure();
    }
    return static_cast<uint32_t>(requestedChainLength - overhead);
  }();
  if (failed(requestedMulDepth)) {
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
      parameters.SetMultiplicativeDepth(*requestedMulDepth);
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
      for (size_t i = 0; i < result.q.size(); ++i) {
        result.realScalingBits.push_back(
            std::log2(cryptoParams->GetScalingFactorReal(i)));
        if (i + 1 < result.q.size()) {
          result.bigScalingBits.push_back(
              std::log2(cryptoParams->GetScalingFactorRealBig(i)));
        }
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

LogicalResult resolveOpenfheCKKSMetadata(ModuleOp module,
                                         StringRef scalingTechnique) {
  if (!moduleIsCKKS(module)) {
    return success();
  }

  auto schemeParamAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
      ckks::CKKSDialect::kSchemeParamAttrName);
  if (!schemeParamAttr) {
    return success();
  }

  StringRef concreteScalingTechnique =
      resolveScalingTechnique(scalingTechnique);
  if (!isSupportedScalingTechnique(concreteScalingTechnique)) {
    module.emitOpError() << "unsupported OpenFHE scaling technique `"
                         << concreteScalingTechnique << "`";
    return failure();
  }

  Builder builder(module.getContext());
  module->setAttr(kScalingTechniqueAttrName,
                  builder.getStringAttr(concreteScalingTechnique));

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

  auto retargetToChainLength =
      [&](int64_t targetChainLength) -> FailureOr<bool> {
    auto currentSchemeParam = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!currentSchemeParam) {
      module.emitOpError()
          << "requires ckks.schemeParam while retargeting OpenFHE metadata";
      return failure();
    }
    int64_t currentChainLength =
        static_cast<int64_t>(currentSchemeParam.getQ().size());
    FailureOr<OpenfheConcreteSchemeParam> concreteSchemeParam =
        materializeOpenfheSchemeParam(
            module, static_cast<int64_t>(1) << currentSchemeParam.getLogN(),
            requestedSlotCount, firstModBits,
            currentSchemeParam.getLogDefaultScale(), targetChainLength,
            concreteScalingTechnique);
    if (failed(concreteSchemeParam)) {
      return failure();
    }

    auto newQ = DenseI64ArrayAttr::get(module.getContext(),
                                       ArrayRef(concreteSchemeParam->q));
    auto newP = DenseI64ArrayAttr::get(module.getContext(),
                                       ArrayRef(concreteSchemeParam->p));
    bool sameSchemeParam = currentSchemeParam.getLogN() ==
                               llvm::Log2_64(concreteSchemeParam->ringDim) &&
                           currentSchemeParam.getQ() == newQ &&
                           currentSchemeParam.getP() == newP;
    if (sameSchemeParam) {
      bool sameActualSlotCount = false;
      if (auto actualSlotCountAttr =
              module->getAttrOfType<IntegerAttr>(kActualSlotCountAttrName)) {
        sameActualSlotCount =
            actualSlotCountAttr.getInt() == concreteSchemeParam->ringDim / 2;
      }
      bool sameRequestedSlotCount = false;
      if (auto requestedSlotCountAttr =
              module->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName)) {
        sameRequestedSlotCount =
            requestedSlotCountAttr.getInt() == requestedSlotCount;
      }
      bool sameReducedError = false;
      if (auto reducedErrorAttr =
              module->getAttrOfType<BoolAttr>(kCKKSReducedErrorAttrName)) {
        sameReducedError = reducedErrorAttr.getValue() == reducedError;
      }
      if (sameActualSlotCount && sameRequestedSlotCount && sameReducedError) {
        LLVM_DEBUG(llvm::dbgs()
                   << "OpenFHE scheme param canonical at chain length "
                   << targetChainLength << "\n");
        return false;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "OpenFHE scheme param attr matches but aux attrs differ: "
                 << "actualSlot=" << sameActualSlotCount
                 << " requestedSlot=" << sameRequestedSlotCount
                 << " reducedError=" << sameReducedError << "\n");
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Retargeting OpenFHE scheme param to chain length "
               << targetChainLength << " sameSchemeParam=" << sameSchemeParam
               << " currentLogN=" << currentSchemeParam.getLogN()
               << " concreteRingDim=" << concreteSchemeParam->ringDim
               << " qSize=" << concreteSchemeParam->q.size() << "\n");

    auto newQAttrs =
        toIntegerAttrs(module.getContext(), concreteSchemeParam->q);
    auto newPAttrs =
        toIntegerAttrs(module.getContext(), concreteSchemeParam->p);
    int64_t delta = static_cast<int64_t>(concreteSchemeParam->q.size()) -
                    currentChainLength;
    FailureOr<polynomial::IntPolynomialAttr> polynomialModulus =
        getCyclotomicPolynomialModulus(module.getContext(),
                                       concreteSchemeParam->ringDim);
    if (failed(polynomialModulus)) {
      module.emitOpError()
          << "failed to build the updated cyclotomic polynomial modulus for "
             "OpenFHE CKKS metadata resolution";
      return failure();
    }
    retargetModuleSchemeParam(module, *polynomialModulus, newQAttrs, delta);
    module->setAttr(
        ckks::CKKSDialect::kSchemeParamAttrName,
        ckks::SchemeParamAttr::get(
            module.getContext(), llvm::Log2_64(concreteSchemeParam->ringDim),
            newQ, newP, currentSchemeParam.getLogDefaultScale(),
            currentSchemeParam.getEncryptionType(),
            currentSchemeParam.getEncryptionTechnique()));
    module->setAttr(kRequestedSlotCountAttrName,
                    builder.getI64IntegerAttr(requestedSlotCount));
    module->setAttr(
        kActualSlotCountAttrName,
        builder.getI64IntegerAttr(concreteSchemeParam->ringDim / 2));
    module->setAttr(kCKKSReducedErrorAttrName,
                    builder.getBoolAttr(reducedError));
    return true;
  };

  for (int iteration = 0; iteration < 8; ++iteration) {
    auto currentSchemeParam = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!currentSchemeParam) {
      module.emitOpError()
          << "requires ckks.schemeParam during OpenFHE metadata resolution";
      return failure();
    }
    int64_t currentChainLength =
        static_cast<int64_t>(currentSchemeParam.getQ().size());
    FailureOr<OpenfheConcreteSchemeParam> currentConcreteSchemeParam =
        materializeOpenfheSchemeParam(
            module, static_cast<int64_t>(1) << currentSchemeParam.getLogN(),
            requestedSlotCount, firstModBits,
            currentSchemeParam.getLogDefaultScale(), currentChainLength,
            concreteScalingTechnique);
    if (failed(currentConcreteSchemeParam)) {
      return failure();
    }
    FailureOr<std::unique_ptr<DataFlowSolver>> solver =
        runOpenfheSchemeAdjustmentAnalyses(
            module, currentSchemeParam.getLogDefaultScale(),
            concreteScalingTechnique,
            currentConcreteSchemeParam->realScalingBits,
            currentConcreteSchemeParam->bigScalingBits);
    if (failed(solver)) {
      return failure();
    }

    int64_t maxLevel = currentChainLength - 1;
    int64_t requiredChainLength =
        getRequiredOpenfheChainLength(module, *solver.value(), maxLevel);
    int64_t targetChainLength =
        std::max(requiredChainLength, currentChainLength);
    LLVM_DEBUG({
      llvm::dbgs() << "OpenFHE CKKS metadata resolution iteration " << iteration
                   << ": current chain length=" << currentChainLength
                   << ", required chain length=" << requiredChainLength
                   << ", max HEIR level=" << maxLevel << "\n";
    });

    FailureOr<bool> schemeChanged = retargetToChainLength(targetChainLength);
    if (failed(schemeChanged)) {
      return failure();
    }
    if (*schemeChanged) {
      continue;
    }

    if (failed(resolveOpaqueLinearTransformNativeLevels(module,
                                                        *solver.value()))) {
      return failure();
    }
    FailureOr<bool> encodeChanged = resolveAddPlainEncodes(
        module, *solver.value(), currentSchemeParam.getLogDefaultScale(),
        concreteScalingTechnique);
    if (failed(encodeChanged)) {
      return failure();
    }
    if (!*encodeChanged) {
      return success();
    }
  }

  module.emitOpError()
      << "OpenFHE CKKS metadata resolution failed to converge after 8 "
         "iterations";
  return failure();
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
