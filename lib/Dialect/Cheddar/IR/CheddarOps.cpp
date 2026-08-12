#include "lib/Dialect/Cheddar/IR/CheddarOps.h"

#include <algorithm>
#include <numeric>

#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/DenseSet.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

namespace {

SmallVector<int64_t> normalizedRotations(DenseI32ArrayAttr diagonals,
                                         int64_t width) {
  SmallVector<int64_t> rotations;
  rotations.reserve(diagonals.size());
  for (int32_t diagonal : diagonals.asArrayRef())
    rotations.push_back(((diagonal % width) + width) % width);
  return rotations;
}

struct LinearTransformRotationPlan {
  llvm::DenseSet<int64_t> babySteps;
  llvm::DenseSet<int64_t> giantSteps;
  int64_t babyStride = 0;
  int64_t giantStride = 0;

  bool supportsMinKs(int64_t bs, int64_t gs) const {
    auto isCompleteProgression = [](const llvm::DenseSet<int64_t>& steps,
                                    int64_t stride) {
      int64_t maximum = 0;
      int64_t nonZeroCount = 0;
      for (int64_t step : steps) {
        if (step == 0) continue;
        maximum = std::max(maximum, step);
        ++nonZeroCount;
      }
      return nonZeroCount > 0 && stride > 0 && nonZeroCount * stride == maximum;
    };
    return bs > 1 && gs > 1 && isCompleteProgression(babySteps, babyStride) &&
           isCompleteProgression(giantSteps, giantStride);
  }
};

LinearTransformRotationPlan linearTransformRotationPlan(
    DenseI32ArrayAttr diagonals, int64_t width, int64_t bs) {
  SmallVector<int64_t> normalized = normalizedRotations(diagonals, width);
  int64_t stride = 0;
  for (int64_t rotation : normalized) stride = std::gcd(stride, rotation);

  LinearTransformRotationPlan plan;
  if (stride == 0) return plan;
  int64_t giantStepStride = stride * bs;
  for (int64_t rotation : normalized) {
    int64_t baby = rotation % giantStepStride;
    int64_t giant = rotation - baby;
    plan.babySteps.insert(baby);
    plan.giantSteps.insert(giant);
    plan.babyStride = std::gcd(plan.babyStride, baby);
    plan.giantStride = std::gcd(plan.giantStride, giant);
  }
  return plan;
}

SmallVector<OpFoldResult> requiredLinearTransformRotations(
    Operation* op, DenseI32ArrayAttr diagonals, int64_t width, int64_t bs,
    int64_t gs, bool minKs) {
  LinearTransformRotationPlan plan =
      linearTransformRotationPlan(diagonals, width, bs);
  llvm::DenseSet<int64_t> required;
  if (minKs && plan.supportsMinKs(bs, gs)) {
    required.insert(plan.babyStride);
    required.insert(plan.giantStride);
  } else {
    for (int64_t baby : plan.babySteps)
      if (baby != 0) required.insert(baby);
    for (int64_t giant : plan.giantSteps)
      if (giant != 0) required.insert(giant);
  }
  SmallVector<OpFoldResult> result;
  result.reserve(required.size());
  SmallVector<int64_t> sorted(required.begin(), required.end());
  llvm::sort(sorted);
  for (int64_t rotation : sorted)
    result.push_back(
        IntegerAttr::get(IndexType::get(op->getContext()), rotation));
  return result;
}

LogicalResult verifyLinearTransformShape(Operation* op, ShapedType diagonals,
                                         DenseI32ArrayAttr indices,
                                         DenseI32ArrayAttr sourceRows,
                                         int64_t bs, int64_t gs, bool minKs) {
  if (diagonals.getRank() != 2)
    return op->emitOpError("diagonals must be a 2D tensor or memref");
  int64_t height = diagonals.getShape()[0];
  int64_t width = diagonals.getShape()[1];
  if (sourceRows) {
    if (sourceRows.size() != indices.size())
      return op->emitOpError(
          "number of source row indices must match number of diagonal indices");
    for (int32_t row : sourceRows.asArrayRef()) {
      if (row < 0 || row >= height)
        return op->emitOpError("source row index ")
               << row << " is out of bounds for " << height << " diagonal rows";
    }
  } else if (height != indices.size()) {
    return op->emitOpError(
        "number of diagonals must match number of diagonal indices");
  }
  if (width <= 0 || (width & (width - 1)) != 0)
    return op->emitOpError("diagonal width must be a positive power of two");
  if (indices.size() < 2)
    return op->emitOpError("requires at least two non-zero diagonals");
  if (bs <= 0 || gs <= 0) return op->emitOpError("bs and gs must be positive");

  int64_t stride = 0;
  int64_t maxRotation = 0;
  llvm::DenseSet<int64_t> seen;
  for (int64_t rotation : normalizedRotations(indices, width)) {
    if (!seen.insert(rotation).second)
      return op->emitOpError("duplicate normalized diagonal index ")
             << rotation;
    stride = std::gcd(stride, rotation);
    maxRotation = std::max(maxRotation, rotation);
  }
  if (stride == 0)
    return op->emitOpError("requires at least one non-zero diagonal index");
  if (maxRotation > (bs * gs - 1) * stride)
    return op->emitOpError("bs/gs cannot represent the maximum diagonal index");
  if (minKs &&
      !linearTransformRotationPlan(indices, width, bs).supportsMinKs(bs, gs))
    return op->emitOpError(
        "min_ks requires complete non-zero baby- and giant-step progressions");
  return success();
}

}  // namespace

bool supportsMinKs(DenseI32ArrayAttr diagonalIndices, int64_t width, int64_t bs,
                   int64_t gs) {
  return linearTransformRotationPlan(diagonalIndices, width, bs)
      .supportsMinKs(bs, gs);
}

::llvm::SmallVector<::mlir::OpFoldResult> HRotOp::getRotationIndices() {
  if (getStaticDistance()) return {getStaticDistanceAttr()};
  return {getDynamicDistance()};
}

LogicalResult HRotOp::verify() {
  return containsExactlyOneOrEmitError(getOperation(), getDynamicDistance(),
                                       getStaticDistance());
}

::llvm::SmallVector<::mlir::OpFoldResult> HRotAddOp::getRotationIndices() {
  return {getDistanceAttr()};
}

::llvm::SmallVector<::mlir::OpFoldResult>
LinearTransformOp::getRotationIndices() {
  auto diagonalsType = cast<ShapedType>(getDiagonals().getType());
  return requiredLinearTransformRotations(
      getOperation(), getDiagonalIndicesAttr(), diagonalsType.getShape()[1],
      getBs().getInt(), getGs().getInt(), getMinKs());
}

::llvm::SmallVector<::mlir::OpFoldResult>
PrepareLinearTransformOp::getRotationIndices() {
  return requiredLinearTransformRotations(
      getOperation(), getDiagonalIndicesAttr(), getWidth().getInt(),
      getBs().getInt(), getGs().getInt(), getMinKs());
}
LogicalResult LinearTransformOp::verify() {
  if (getLevel().getInt() <= 0)
    return emitOpError("requires an input level above zero");
  auto diagonalsType = cast<ShapedType>(getDiagonals().getType());
  return verifyLinearTransformShape(getOperation(), diagonalsType,
                                    getDiagonalIndicesAttr(),
                                    getSourceRowIndicesAttr(), getBs().getInt(),
                                    getGs().getInt(), getMinKs());
}

LogicalResult PrepareLinearTransformOp::verify() {
  if (getLevel().getInt() <= 0)
    return emitOpError("requires an input level above zero");
  auto diagonals = cast<ShapedType>(getDiagonals().getType());
  if (diagonals.getRank() == 2 &&
      diagonals.getShape()[1] != getWidth().getInt())
    return emitOpError("width must match the diagonal width");
  return verifyLinearTransformShape(getOperation(), diagonals,
                                    getDiagonalIndicesAttr(),
                                    getSourceRowIndicesAttr(), getBs().getInt(),
                                    getGs().getInt(), getMinKs());
}

LogicalResult EvalPolyOp::verify() {
  if (getLevelConsumption().getInt() < 2)
    return emitOpError("level consumption must be at least two");
  return success();
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
