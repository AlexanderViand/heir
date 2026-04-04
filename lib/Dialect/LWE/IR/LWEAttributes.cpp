#include "lib/Dialect/LWE/IR/LWEAttributes.h"

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKSScalePolicy.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "lwe-attributes"

namespace mlir {
namespace heir {
namespace lwe {

namespace {

constexpr llvm::StringLiteral kCheddarBackendAttrName = "backend.cheddar";

APInt getNominalCkksRescaleFactor(Operation* op, const APInt& dividedModulus) {
  if (op) {
    if (auto moduleOp = op->getParentOfType<ModuleOp>()) {
      if (auto schemeParamAttr = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
              ckks::CKKSDialect::kSchemeParamAttrName)) {
        return getNominalPowerOfTwoScaleFromLog2(
            schemeParamAttr.getLogDefaultScale());
      }
    }
  }
  // Type-only callers have no enclosing module from which to recover the
  // configured CKKS nominal scale. Preserve the historical approximation based
  // on the dropped modulus in that context.
  return getNominalPowerOfTwoScaleFromLog2(dividedModulus.nearestLogBase2());
}

FailureOr<APInt> inferModulusSwitchOrRescaleOpScalingFactor(
    Attribute xEncoding, APInt dividedModulus, const APInt& plaintextModulus,
    const APInt& nominalScale, bool preciseCkksScalePolicy) {
  APInt xScale = getScalingFactorFromEncodingAttr(xEncoding);
  return llvm::TypeSwitch<Attribute, FailureOr<APInt>>(xEncoding)
      .Case<FullCRTPackingEncodingAttr>([&](auto attr) -> FailureOr<APInt> {
        if (xScale.isZero()) return xScale;
        APInt modulus = canonicalizeUnsignedAPInt(plaintextModulus);
        unsigned width =
            std::max(dividedModulus.getBitWidth(), modulus.getBitWidth());
        APInt wideDividedModulus = dividedModulus.zextOrTrunc(width);
        APInt wideModulus = modulus.zextOrTrunc(width);
        APInt qInvT = multiplicativeInverse(
            wideDividedModulus.urem(wideModulus), wideModulus);
        if (qInvT.isZero()) return failure();
        return modularMultiplication(xScale, qInvT, wideModulus);
      })
      .Case<InverseCanonicalEncodingAttr>([&](auto attr) -> FailureOr<APInt> {
        if (xScale.isZero()) return xScale;
        if (preciseCkksScalePolicy) {
          return divideUnsignedAPIntNearest(xScale, dividedModulus);
        }
        auto newScale = divideUnsignedAPIntExact(xScale, nominalScale);
        if (failed(newScale)) return failure();
        LLVM_DEBUG(llvm::dbgs() << "inferring new scale; dividedModulus="
                                << dividedModulus << ", xScale=" << xScale
                                << ", newScale=" << *newScale << "\n");
        return *newScale;
      })
      .Default([](Attribute) -> FailureOr<APInt> { return failure(); });
}

}  // namespace

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

APInt getScalingFactorFromEncodingAttr(Attribute encoding) {
  return llvm::TypeSwitch<Attribute, APInt>(encoding)
      .Case<FullCRTPackingEncodingAttr>([](auto attr) {
        auto value = attr.getScalingFactor().getValue();
        assert(!value.isNegative() && "expected non-negative scaling factor");
        return canonicalizeUnsignedAPInt(value);
      })
      .Case<InverseCanonicalEncodingAttr>([](auto attr) {
        auto value = attr.getScalingFactor().getValue();
        assert(!value.isNegative() && "expected non-negative scaling factor");
        return canonicalizeUnsignedAPInt(value);
      })
      .Default([](Attribute) { return APInt(64, 0); });
}

APInt inferMulOpScalingFactor(Attribute xEncoding, Attribute yEncoding,
                              const APInt& plaintextModulus) {
  APInt xScale = getScalingFactorFromEncodingAttr(xEncoding);
  APInt yScale = getScalingFactorFromEncodingAttr(yEncoding);
  return llvm::TypeSwitch<Attribute, APInt>(xEncoding)
      .Case<FullCRTPackingEncodingAttr>([&](auto attr) {
        return modularMultiplication(
            xScale, yScale, canonicalizeUnsignedAPInt(plaintextModulus));
      })
      .Case<InverseCanonicalEncodingAttr>(
          [&](auto attr) { return multiplyUnsignedAPIntExact(xScale, yScale); })
      .Default([](Attribute) { return APInt(64, 0); });
}

Attribute getEncodingAttrWithNewScalingFactor(Attribute encoding,
                                              const APInt& newScale) {
  return llvm::TypeSwitch<Attribute, Attribute>(encoding)
      .Case<FullCRTPackingEncodingAttr>([&](auto attr) {
        return FullCRTPackingEncodingAttr::get(encoding.getContext(), newScale);
      })
      .Case<InverseCanonicalEncodingAttr>([&](auto attr) {
        return InverseCanonicalEncodingAttr::get(encoding.getContext(),
                                                 newScale);
      })
      .Default([](Attribute) { return nullptr; });
}

PlaintextSpaceAttr inferMulOpPlaintextSpaceAttr(MLIRContext* ctx,
                                                PlaintextSpaceAttr x,
                                                PlaintextSpaceAttr y) {
  auto xRing = x.getRing();
  auto xEncoding = x.getEncoding();
  auto yEncoding = y.getEncoding();

  APInt plaintextModulus(64, 0);
  if (auto modArithType =
          llvm::dyn_cast<mod_arith::ModArithType>(xRing.getCoefficientType())) {
    plaintextModulus =
        canonicalizeUnsignedAPInt(modArithType.getModulus().getValue());
  }

  auto newScale =
      inferMulOpScalingFactor(xEncoding, yEncoding, plaintextModulus);
  return PlaintextSpaceAttr::get(
      ctx, xRing, getEncodingAttrWithNewScalingFactor(xEncoding, newScale));
}

FailureOr<PlaintextSpaceAttr> inferModulusSwitchOrRescaleOpPlaintextSpaceAttr(
    Operation* op, PlaintextSpaceAttr x, APInt dividedModulus) {
  auto nominalScale = getNominalCkksRescaleFactor(op, dividedModulus);
  bool preciseCkksScalePolicy = false;
  if (op) {
    if (auto moduleOp = op->getParentOfType<ModuleOp>()) {
      bool requestedPrecise = false;
      if (auto policy =
              moduleOp->getAttrOfType<StringAttr>(kCKKSScalePolicyAttrName)) {
        requestedPrecise = policy.getValue() == kCKKSPreciseScalePolicyValue;
      }
      bool isCheddar =
          moduleOp->getAttrOfType<UnitAttr>(kCheddarBackendAttrName) != nullptr;
      preciseCkksScalePolicy = requestedPrecise && !isCheddar;
    }
  }
  auto xRing = x.getRing();
  auto xEncoding = x.getEncoding();

  APInt plaintextModulus(64, 0);
  if (auto modArithType =
          llvm::dyn_cast<mod_arith::ModArithType>(xRing.getCoefficientType())) {
    plaintextModulus =
        canonicalizeUnsignedAPInt(modArithType.getModulus().getValue());
  }

  auto newScale = inferModulusSwitchOrRescaleOpScalingFactor(
      xEncoding, dividedModulus, plaintextModulus, nominalScale,
      preciseCkksScalePolicy);
  if (failed(newScale)) return failure();
  LLVM_DEBUG(llvm::dbgs() << "dividedModulus=" << dividedModulus
                          << " new scale=" << *newScale << "\n");
  return PlaintextSpaceAttr::get(
      op->getContext(), xRing,
      getEncodingAttrWithNewScalingFactor(xEncoding, *newScale));
}

FailureOr<PlaintextSpaceAttr> inferModulusSwitchOrRescaleOpPlaintextSpaceAttr(
    MLIRContext* ctx, PlaintextSpaceAttr x, APInt dividedModulus) {
  auto nominalScale =
      getNominalCkksRescaleFactor(/*op=*/nullptr, dividedModulus);
  auto xRing = x.getRing();
  auto xEncoding = x.getEncoding();

  APInt plaintextModulus(64, 0);
  if (auto modArithType =
          llvm::dyn_cast<mod_arith::ModArithType>(xRing.getCoefficientType())) {
    plaintextModulus =
        canonicalizeUnsignedAPInt(modArithType.getModulus().getValue());
  }

  auto newScale = inferModulusSwitchOrRescaleOpScalingFactor(
      xEncoding, dividedModulus, plaintextModulus, nominalScale,
      /*preciseCkksScalePolicy=*/false);
  if (failed(newScale)) return failure();
  LLVM_DEBUG(llvm::dbgs() << "dividedModulus=" << dividedModulus
                          << " new scale=" << *newScale << "\n");
  return PlaintextSpaceAttr::get(
      ctx, xRing, getEncodingAttrWithNewScalingFactor(xEncoding, *newScale));
}

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level) {
  auto rnsType = cast<rns::RNSType>(ringAttr.getCoefficientType());
  auto newRnsType = rns::RNSType::get(
      rnsType.getContext(), rnsType.getBasisTypes().take_front(level + 1));
  return polynomial::RingAttr::get(newRnsType, ringAttr.getPolynomialModulus());
}

polynomial::RingAttr getRingFromModulusChain(
    ModulusChainAttr chainAttr,
    polynomial::IntPolynomialAttr polynomialModulus) {
  SmallVector<Type> limbTypes = llvm::to_vector(llvm::map_range(
      chainAttr.getElements(), [](mlir::IntegerAttr attr) -> Type {
        return mod_arith::ModArithType::get(attr.getType().getContext(), attr);
      }));
  rns::RNSType rnsType = rns::RNSType::get(
      chainAttr.getContext(),
      ArrayRef<Type>(limbTypes).take_front(chainAttr.getCurrent() + 1));
  return polynomial::RingAttr::get(rnsType, polynomialModulus);
}

//===----------------------------------------------------------------------===//
// Attribute Verification
//===----------------------------------------------------------------------===//

LogicalResult PlaintextSpaceAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::heir::polynomial::RingAttr ring, Attribute encoding) {
  auto verifyNonNegativeScalingFactor =
      [&](Attribute encodingAttr) -> LogicalResult {
    return llvm::TypeSwitch<Attribute, LogicalResult>(encodingAttr)
        .Case<FullCRTPackingEncodingAttr, InverseCanonicalEncodingAttr,
              ConstantCoefficientEncodingAttr, CoefficientEncodingAttr>(
            [&](auto attr) {
              if (attr.getScalingFactor().getValue().isNegative()) {
                emitError() << "scaling_factor must be a non-negative integer";
                return failure();
              }
              return success();
            })
        .Default([](Attribute) { return success(); });
  };

  if (failed(verifyNonNegativeScalingFactor(encoding))) {
    return failure();
  }

  if (mlir::isa<FullCRTPackingEncodingAttr>(encoding)) {
    // For full CRT packing, the ring must be of the form x^n + 1 and the
    // modulus must be 1 mod n.
    auto polyMod = ring.getPolynomialModulus();
    auto poly = polyMod.getPolynomial();
    auto polyTerms = poly.getTerms();
    if (polyTerms.size() != 2) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    const auto& constantTerm = polyTerms[0];
    const auto& constantCoeff = constantTerm.getCoefficient();
    if (!(constantTerm.getExponent().isZero() && constantCoeff.isOne() &&
          polyTerms[1].getCoefficient().isOne())) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    // Check that the modulus is 1 mod n.
    auto modCoeffTy =
        llvm::dyn_cast<mod_arith::ModArithType>(ring.getCoefficientType());
    if (modCoeffTy) {
      APInt modulus = modCoeffTy.getModulus().getValue();
      unsigned n = poly.getDegree();
      if (!modulus.urem(APInt(modulus.getBitWidth(), n)).isOne()) {
        return emitError()
               << "modulus must be 1 mod n for full CRT packing, mod = "
               << modulus.getZExtValue() << " n = " << n << "\n";
      }
    }
  }

  return success();
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
