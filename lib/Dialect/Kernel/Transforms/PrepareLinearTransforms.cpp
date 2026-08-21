#include "lib/Dialect/Kernel/Transforms/PrepareLinearTransforms.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/Kernel/IR/KernelTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

#define GEN_PASS_DEF_PREPARELINEARTRANSFORMS
#include "lib/Dialect/Kernel/Transforms/Passes.h.inc"

namespace {

// Returns the (possibly tensor-wrapped) LWE ciphertext type of a linear
// transform's input, or null when the input has some other type.
lwe::LWECiphertextType getInputCiphertextType(Type inputType) {
  if (auto ctType = dyn_cast<lwe::LWECiphertextType>(inputType)) return ctType;
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    return dyn_cast<lwe::LWECiphertextType>(tensorType.getElementType());
  }
  return nullptr;
}

// A one-diagonal transform is a rotation followed by an elementwise plaintext
// multiplication.
LogicalResult lowerSingleDiagonalTransform(LinearTransformOp op,
                                           OpBuilder& builder) {
  ArrayRef<int64_t> diagonalIndices = op.getDiagonalIndices();
  if (diagonalIndices.size() != 1) return failure();

  auto inputTensorType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (inputTensorType &&
      (inputTensorType.getRank() != 1 || inputTensorType.getNumElements() != 1))
    return failure();

  auto diagonalsType = dyn_cast<RankedTensorType>(op.getDiagonals().getType());
  if (!diagonalsType || !diagonalsType.hasStaticShape()) return failure();

  lwe::LWECiphertextType inputType =
      getInputCiphertextType(op.getInput().getType());
  lwe::LWECiphertextType outputType =
      getInputCiphertextType(op.getOutput().getType());
  if (!inputType || !outputType || !inputType.getModulusChain())
    return failure();

  int64_t sourceRow = 0;
  if (auto sourceRows = op.getSourceRowIndicesAttr())
    sourceRow = sourceRows.asArrayRef().front();
  int64_t width = diagonalsType.getDimSize(1);
  SmallVector<int64_t> rowShape = inputTensorType
                                      ? SmallVector<int64_t>{1, width}
                                      : SmallVector<int64_t>{width};
  auto rowType =
      RankedTensorType::get(rowShape, diagonalsType.getElementType());
  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(sourceRow),
                                       builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1),
                                     builder.getIndexAttr(width)};
  SmallVector<OpFoldResult> strides(2, builder.getIndexAttr(1));
  Value diagonal = tensor::ExtractSliceOp::create(builder, op.getLoc(), rowType,
                                                  op.getDiagonals(), offsets,
                                                  sizes, strides);

  int64_t rotation = diagonalIndices.front() % width;
  if (rotation < 0) rotation += width;
  Value input = op.getInput();
  if (rotation != 0) {
    input = ckks::RotateOp::create(builder, op.getLoc(), input.getType(), input,
                                   /*dynamic_shift=*/nullptr,
                                   builder.getIndexAttr(rotation));
  }

  auto plaintextElementType = lwe::LWEPlaintextType::get(
      op.getContext(), inputType.getPlaintextSpace());
  Type plaintextType = plaintextElementType;
  if (inputTensorType)
    plaintextType =
        RankedTensorType::get(inputTensorType.getShape(), plaintextElementType);
  auto encoding = inputType.getPlaintextSpace().getEncoding();
  auto encoded = lwe::RLWEEncodeOp::create(
      builder, op.getLoc(), plaintextType, diagonal, encoding,
      inputType.getPlaintextSpace().getRing(),
      builder.getI64IntegerAttr(inputType.getModulusChain().getCurrent()),
      builder.getI64IntegerAttr(
          lwe::getScalingFactorFromEncodingAttr(encoding)));
  auto product = ckks::MulPlainOp::create(builder, op.getLoc(), input,
                                          encoded.getOutput());
  auto rescaled = ckks::RescaleOp::create(
      builder, op.getLoc(), op.getOutput().getType(), product.getOutput(),
      outputType.getCiphertextSpace().getRing());
  rescaled->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  op.getOutput().replaceAllUsesWith(rescaled.getOutput());
  op.erase();
  return success();
}

struct PrepareLinearTransforms
    : impl::PrepareLinearTransformsBase<PrepareLinearTransforms> {
  using PrepareLinearTransformsBase::PrepareLinearTransformsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto target = getTargetConfig(module);
    if (failed(target) || !target->has_prepared_linear_transform) return;
    // Only the CKKS lowerings implement prepare/apply, so splitting anything
    // else would leave ops no backend pattern can convert.
    if (!moduleIsCKKS(module)) return;

    module->walk([&](LinearTransformOp op) {
      lwe::LWECiphertextType ctType =
          getInputCiphertextType(op.getInput().getType());
      if (!ctType) return;
      std::optional<int64_t> level = lwe::getLevel(ctType);
      if (!level.has_value()) return;

      OpBuilder builder(op);
      if (!target->supports_single_diagonal_prepared_linear_transform &&
          op.getDiagonalIndices().size() == 1 &&
          succeeded(lowerSingleDiagonalTransform(op, builder))) {
        return;
      }

      // The slot count the diagonals are encoded for is the ciphertext's
      // *encoded* width, which getEncodedSlotCount derives from the ring's
      // capacity and the module's requested count.
      auto plaintextSpace = ctType.getPlaintextSpace();
      int64_t ringCapacity = plaintextSpace.getRing()
                                 .getPolynomialModulus()
                                 .getPolynomial()
                                 .getDegree();
      if (isa<lwe::InverseCanonicalEncodingAttr>(
              plaintextSpace.getEncoding())) {
        ringCapacity /= 2;
      }
      int64_t slots = getEncodedSlotCount(module, ringCapacity);

      // kernel.linear_transform's bsgs_ratio is a baby-step/giant-step
      // ratio, of which the prepared type records the log2. No attribute
      // means the backend's own default split.
      int64_t logBsgsRatio = 0;
      if (auto ratio = op.getBsgsRatioAttr()) {
        double value = ratio.getValueAsDouble();
        if (value < 1.0) {
          op.emitOpError("bsgs_ratio must be at least 1");
          signalPassFailure();
          return;
        }
        logBsgsRatio = static_cast<int64_t>(std::log2(value));
      }
      auto preparedType = PreparedLinearTransformType::get(
          module.getContext(), *level, slots, logBsgsRatio);
      auto prepare = PrepareLinearTransformOp::create(
          builder, op.getLoc(), preparedType, op.getDiagonals(),
          op.getDiagonalIndicesAttr(), op.getSourceRowIndicesAttr());
      auto apply = ApplyLinearTransformOp::create(
          builder, op.getLoc(), op.getOutput().getType(), op.getInput(),
          prepare.getPrepared(), op.getDiagonalIndicesAttr(),
          builder.getI64IntegerAttr(
              cast<ShapedType>(op.getDiagonals().getType()).getDimSize(1)));
      // The transform op may carry analysis attributes (mgmt levels, debug
      // names); they describe the ciphertext computation, so they move to the
      // apply.
      apply->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      op.getOutput().replaceAllUsesWith(apply.getOutput());
      op.erase();
    });
  }
};

}  // namespace

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
