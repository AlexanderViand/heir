#include "lib/Transforms/LowerOrion/LowerOrion.h"

#include <cstdint>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Orion/IR/OrionDialect.h"
#include "lib/Dialect/Orion/IR/OrionOps.h"
#include "lib/Dialect/Orion/IR/OrionUtils.h"
#include "lib/Utils/RotationUtils.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir {

namespace {

LogicalResult validateDiagonalBasicLinearTransform(
    orion::LinearTransformOp op) {
  auto ctType = dyn_cast<lwe::LWECiphertextType>(op.getInput().getType());
  if (!ctType) {
    return op.emitOpError() << "expected lwe ciphertext input type";
  }
  auto modulusChain = ctType.getModulusChain();
  if (!modulusChain) {
    return op.emitOpError()
           << "diagonal-basic linear transform requires an input modulus "
              "chain";
  }
  int64_t inputLevel = modulusChain.getCurrent();
  int64_t requestedLevel = op.getOrionLevelAttr().getInt();
  if (requestedLevel < 0 || requestedLevel > inputLevel) {
    return op.emitOpError()
           << "expected `orion_level` to be between 0 and the input "
              "ciphertext level "
           << inputLevel << ", but got " << requestedLevel;
  }

  auto diagonalsType = dyn_cast<RankedTensorType>(op.getDiagonals().getType());
  if (!diagonalsType || diagonalsType.getRank() != 2) {
    return op.emitOpError() << "expected diagonals operand to have rank 2";
  }

  int64_t diagonalCount = diagonalsType.getShape()[0];
  int64_t slots = diagonalsType.getShape()[1];
  int64_t slotsAttr = op.getSlots().getInt();
  if (slots != slotsAttr) {
    return op.emitOpError()
           << "expected diagonals tensor width to match the `slots` "
              "attribute, but got tensor width "
           << slots << " and slots attr " << slotsAttr;
  }
  if (auto diagonalCountAttr =
          op->getAttrOfType<IntegerAttr>("diagonal_count")) {
    if (diagonalCountAttr.getInt() != diagonalCount) {
      return op.emitOpError()
             << "expected diagonals tensor row count to match the "
                "`diagonal_count` attribute, but got tensor row count "
             << diagonalCount << " and diagonal_count attr "
             << diagonalCountAttr.getInt();
    }
  }
  if (auto blockRowAttr = op->getAttrOfType<IntegerAttr>("block_row")) {
    if (blockRowAttr.getInt() != 0) {
      return op.emitOpError()
             << "diagonal-basic linear transform only supports block_row = 0";
    }
  }
  if (auto blockColAttr = op->getAttrOfType<IntegerAttr>("block_col")) {
    if (blockColAttr.getInt() != 0) {
      return op.emitOpError()
             << "diagonal-basic linear transform only supports block_col = 0";
    }
  }

  auto diagonalIndices = op.getDiagonalIndicesAttr().asArrayRef();
  if (static_cast<int64_t>(diagonalIndices.size()) != diagonalCount) {
    return op.emitOpError()
           << "diagonal count does not match diagonal_indices size";
  }

  return success();
}

struct LowerExplicitLinearTransformPattern
    : public OpRewritePattern<orion::LinearTransformOp> {
  using OpRewritePattern<orion::LinearTransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(orion::LinearTransformOp op,
                                PatternRewriter& rewriter) const override {
    auto implStyle = op->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
    if (!implStyle || implStyle.getValue() != orion::kDiagonalBasicImplStyle) {
      return rewriter.notifyMatchFailure(
          op, "linear transform does not request the diagonal-basic style");
    }

    if (failed(validateDiagonalBasicLinearTransform(op))) {
      return failure();
    }
    auto ctType = cast<lwe::LWECiphertextType>(op.getInput().getType());
    int64_t requestedLevel = op.getOrionLevelAttr().getInt();
    auto plaintextSpace = ctType.getPlaintextSpace();
    auto scaleOneEncoding = lwe::getEncodingAttrWithNewScalingFactor(
        plaintextSpace.getEncoding(), APInt(64, 1));
    auto unitScalePlaintextType = lwe::LWEPlaintextType::get(
        rewriter.getContext(),
        lwe::PlaintextSpaceAttr::get(
            rewriter.getContext(), plaintextSpace.getRing(), scaleOneEncoding));

    auto diagonalsType = cast<RankedTensorType>(op.getDiagonals().getType());
    int64_t diagonalCount = diagonalsType.getShape()[0];
    int64_t slots = diagonalsType.getShape()[1];
    auto diagonalIndices = op.getDiagonalIndicesAttr().asArrayRef();

    auto row2dType =
        RankedTensorType::get({1, slots}, diagonalsType.getElementType());
    auto row1dType =
        RankedTensorType::get({slots}, diagonalsType.getElementType());
    SmallVector<ReassociationIndices> reassociation = {{0, 1}};

    Value accumulated;
    for (int64_t i = 0; i < diagonalCount; ++i) {
      SmallVector<OpFoldResult> offsets = {rewriter.getIndexAttr(i),
                                           rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(slots)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};
      auto row2d = tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                  row2dType, op.getDiagonals(),
                                                  offsets, sizes, strides);
      auto row1d = tensor::CollapseShapeOp::create(
          rewriter, op.getLoc(), row1dType, row2d.getResult(), reassociation);
      auto plaintext = lwe::RLWEEncodeOp::create(
          rewriter, op.getLoc(), unitScalePlaintextType, row1d.getResult(),
          scaleOneEncoding, plaintextSpace.getRing(),
          rewriter.getI64IntegerAttr(requestedLevel));

      Value rotated = op.getInput();
      int64_t shift = normalizeRotationSignedMinimal(diagonalIndices[i], slots);
      if (shift != 0) {
        rotated = ckks::RotateOp::create(rewriter, op.getLoc(), rotated,
                                         Value(), rewriter.getIndexAttr(shift))
                      .getResult();
      }

      auto product = ckks::MulPlainOp::create(rewriter, op.getLoc(), rotated,
                                              plaintext.getResult());
      accumulated = accumulated
                        ? ckks::AddOp::create(rewriter, op.getLoc(),
                                              accumulated, product.getResult())
                              .getResult()
                        : product.getResult();
    }

    if (!accumulated) {
      return op.emitOpError()
             << "cannot lower linear transform with no diagonals";
    }

    rewriter.replaceOp(op, accumulated);
    return success();
  }
};

}  // namespace

#define GEN_PASS_DEF_LOWERORION
#include "lib/Transforms/LowerOrion/LowerOrion.h.inc"

struct LowerOrion : impl::LowerOrionBase<LowerOrion> {
  using LowerOrionBase::LowerOrionBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerExplicitLinearTransformPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    WalkResult result = getOperation()->walk([&](orion::LinearTransformOp op) {
      auto implStyle = op->getAttrOfType<StringAttr>(orion::kImplStyleAttrName);
      if (!implStyle ||
          implStyle.getValue() != orion::kDiagonalBasicImplStyle) {
        return WalkResult::advance();
      }
      if (failed(validateDiagonalBasicLinearTransform(op))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      op.emitOpError()
          << "requested explicit diagonal-basic lowering, but the Orion op "
             "survived `--lower-orion`";
      signalPassFailure();
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      return;
    }
  }
};

}  // namespace mlir::heir
