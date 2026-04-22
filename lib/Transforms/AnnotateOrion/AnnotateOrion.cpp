#include "lib/Transforms/AnnotateOrion/AnnotateOrion.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"       // from @llvm-project

namespace mlir {
namespace heir {

namespace {

constexpr StringRef kDiagonalBSGSImplStyle = "diagonal-bsgs";
constexpr StringRef kBSGSImplStyle = "bsgs";

std::optional<int64_t> getChebyshevLevelCost(orion::ChebyshevOp op,
                                             StringRef implStyle) {
  if (implStyle != kBSGSImplStyle) {
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

    if (linearTransformImplStyle != kDiagonalBSGSImplStyle) {
      module.emitOpError()
          << "unsupported Orion linear_transform implementation style `"
          << linearTransformImplStyle << "`";
      signalPassFailure();
      return;
    }

    if (chebyshevImplStyle != kBSGSImplStyle) {
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
  }
};

}  // namespace heir
}  // namespace mlir
