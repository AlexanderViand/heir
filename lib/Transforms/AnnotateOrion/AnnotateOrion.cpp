#include "lib/Transforms/AnnotateOrion/AnnotateOrion.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/Openfhe/Transforms/ScalingTechniqueUtils.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Dialect/Orion/IR/OrionUtils.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"       // from @llvm-project

namespace mlir {
namespace heir {

namespace {

std::optional<int64_t> getChebyshevLevelCost(orion::ChebyshevOp op,
                                             StringRef implStyle) {
  if (implStyle != orion::kOpaqueImplStyle) {
    return std::nullopt;
  }
  auto coeffs = op.getCoefficients();
  if (coeffs.empty()) {
    return int64_t{0};
  }
  uint64_t degree = coeffs.size() - 1;
  if (degree <= 1) {
    return int64_t{0};
  }
  return static_cast<int64_t>(llvm::Log2_64_Ceil(degree));
}

}  // namespace

#define GEN_PASS_DEF_ANNOTATEORION
#include "lib/Transforms/AnnotateOrion/AnnotateOrion.h.inc"

struct AnnotateOrion : impl::AnnotateOrionBase<AnnotateOrion> {
  using AnnotateOrionBase::AnnotateOrionBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());

    if (!orion::isSupportedLinearTransformImplStyle(linearTransformImplStyle)) {
      module.emitOpError()
          << "unsupported Orion linear_transform implementation style `"
          << linearTransformImplStyle
          << "`, expected `opaque` or `diagonal-basic`";
      signalPassFailure();
      return;
    }

    if (chebyshevImplStyle != orion::kOpaqueImplStyle) {
      module.emitOpError()
          << "unsupported Orion chebyshev implementation style `"
          << chebyshevImplStyle << "`";
      signalPassFailure();
      return;
    }

    Builder builder(module.getContext());
    module.walk([&](orion::LinearTransformOp op) {
      op->setAttr(orion::kImplStyleAttrName,
                  builder.getStringAttr(linearTransformImplStyle));
      op->setAttr(orion::kLevelCostUpperBoundAttrName,
                  builder.getI64IntegerAttr(0));
      // Set OpenFHE native plaintext level from orion_level if not already set.
      if (!op->hasAttr(openfhe::kNativePlaintextLevelAttrName)) {
        op->setAttr(openfhe::kNativePlaintextLevelAttrName,
                    op.getOrionLevelAttr());
      }
    });
    module.walk([&](orion::ChebyshevOp op) {
      op->setAttr(orion::kImplStyleAttrName,
                  builder.getStringAttr(chebyshevImplStyle));
      auto levelCost = getChebyshevLevelCost(op, chebyshevImplStyle);
      if (!levelCost) {
        op.emitOpError() << "failed to determine coarse level cost";
        signalPassFailure();
        return WalkResult::interrupt();
      }
      op->setAttr(orion::kLevelCostUpperBoundAttrName,
                  builder.getI64IntegerAttr(*levelCost));
      return WalkResult::advance();
    });
    // Annotate tensor_ext.diagonal_matvec ops with level cost = 0.
    // The native-API linear transform implementations (OpenFHE's EXT-based
    // helper, Lattigo's lintrans) operate in the extended QP basis and
    // do not consume Q levels. For the diagonal-basic expansion to
    // rotate+multiply+add, the individual ops handle their own levels.
    module.walk([&](tensor_ext::DiagonalMatvecOp op) {
      if (!op->hasAttr(orion::kLevelCostUpperBoundAttrName)) {
        op->setAttr(orion::kLevelCostUpperBoundAttrName,
                    builder.getI64IntegerAttr(0));
      }
    });
    // Annotate polynomial.eval ops (from PolynomialApproximation) with level
    // cost so that management can account for their depth. The level cost
    // follows the same formula as orion.chebyshev: ceil(log2(degree)).
    module.walk([&](polynomial::EvalOp op) {
      if (op->hasAttr(orion::kLevelCostUpperBoundAttrName)) {
        return;
      }
      uint64_t degree = 0;
      if (auto chebAttr = dyn_cast<polynomial::TypedChebyshevPolynomialAttr>(
              op.getPolynomialAttr())) {
        auto coeffs = chebAttr.getValue().getCoefficients();
        degree = coeffs.empty() ? 0 : coeffs.size() - 1;
      } else if (auto floatAttr =
                     dyn_cast<polynomial::TypedFloatPolynomialAttr>(
                         op.getPolynomialAttr())) {
        auto terms = floatAttr.getValue().getPolynomial().getTerms();
        degree = terms.empty() ? 0 : terms.back().getExponent().getSExtValue();
      } else {
        return;
      }
      int64_t levelCost =
          degree <= 1 ? 0 : static_cast<int64_t>(llvm::Log2_64_Ceil(degree));
      op->setAttr(orion::kLevelCostUpperBoundAttrName,
                  builder.getI64IntegerAttr(levelCost));
    });
  }
};

}  // namespace heir
}  // namespace mlir
