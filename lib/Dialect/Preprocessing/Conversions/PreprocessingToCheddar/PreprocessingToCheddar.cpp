#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToCheddar/PreprocessingToCheddar.h"

#include <cstdint>
#include <iterator>
#include <optional>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::preprocessing {

#define GEN_PASS_DEF_PREPROCESSINGTOCHEDDAR
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToCheddar/PreprocessingToCheddar.h.inc"

namespace {

// Cheddar represents a single plaintext as a rank-0
// `tensor<!cheddar.plaintext>`. A memref element type cannot be a tensor, so
// the storage memref holds the unwrapped scalar element; store/load bridge via
// tensor.extract/from_elements.
Type unwrapRank0Tensor(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type);
      tensorType && tensorType.getRank() == 0)
    return tensorType.getElementType();
  return type;
}

bool isMoveOnlyPayload(Type type) {
  return isa<cheddar::CiphertextType, cheddar::PlaintextType,
             cheddar::ConstantType, cheddar::EvalKeyType>(
      unwrapRank0Tensor(type));
}

SmallVector<Type> getUniqueElementTypes(PreprocessingStorageType storageType) {
  SmallVector<Type> elementTypes;
  for (Type elementType : storageType.getElementTypes())
    if (!llvm::is_contained(elementTypes, elementType))
      elementTypes.push_back(elementType);
  return elementTypes;
}

FailureOr<int> getElementTypeIndex(PreprocessingStorageType storageType,
                                   Type elementType) {
  SmallVector<Type> elementTypes = getUniqueElementTypes(storageType);
  auto it = llvm::find(elementTypes, elementType);
  if (it == elementTypes.end()) return failure();
  return std::distance(elementTypes.begin(), it);
}

// One preprocessing storage value maps to one flat memref per element type.
class CheddarPreprocessingTypeConverter : public TypeConverter {
 public:
  explicit CheddarPreprocessingTypeConverter(
      const PreprocessingStorageLayoutAnalysis& analysis) {
    addConversion([](Type type) { return type; });
    addConversion([&analysis](PreprocessingStorageType type,
                              SmallVectorImpl<Type>& results) {
      for (Type elementType : getUniqueElementTypes(type)) {
        int64_t totalSize = analysis.getTotalSize(elementType).value_or(0);
        results.push_back(
            MemRefType::get({totalSize}, unwrapRank0Tensor(elementType)));
      }
      return success();
    });
  }
};

struct EmptyOpPattern : public OpConversionPattern<EmptyOp> {
  using OneToNOpAdaptor =
      typename OpConversionPattern<EmptyOp>::OneToNOpAdaptor;
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EmptyOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertType(op.getStorage().getType(),
                                               resultTypes)))
      return failure();
    SmallVector<Value> allocations;
    for (Type resultType : resultTypes)
      allocations.push_back(memref::AllocOp::create(
          rewriter, op.getLoc(), cast<MemRefType>(resultType)));
    rewriter.replaceOpWithMultiple(op, {allocations});
    return success();
  }
};

struct SiteIndex {
  int elementIndex;
  Value linearIndex;
};

FailureOr<SiteIndex> resolveSite(
    Operation* op, PreprocessingStorageType storageType, Type elementType,
    uint32_t siteId, ArrayRef<ValueRange> indexRanges,
    const PreprocessingStorageLayoutAnalysis& analysis,
    ConversionPatternRewriter& rewriter) {
  FailureOr<int> elementIndex = getElementTypeIndex(storageType, elementType);
  if (failed(elementIndex)) return failure();
  SmallVector<Value> flatIndices;
  for (ValueRange range : indexRanges) {
    if (range.size() != 1)
      return op->emitOpError() << "expected exactly one SSA value per index";
    flatIndices.push_back(range.front());
  }
  std::optional<SiteLayout> layout = analysis.getLayout(elementType, siteId);
  if (!layout.has_value())
    return op->emitOpError() << "missing layout for site ID " << siteId;
  FailureOr<Value> index = preprocessing::getLinearIndex(
      rewriter, op->getLoc(), op, layout->offset, flatIndices);
  if (failed(index)) return failure();
  return SiteIndex{*elementIndex, index.value()};
}

Value makeRankZeroSubview(ConversionPatternRewriter& rewriter, Location loc,
                          Value storage, Value index) {
  SmallVector<OpFoldResult> offsets{index};
  SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1)};
  SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
  SmallVector<int64_t> resultShape;
  auto resultType = memref::SubViewOp::inferRankReducedResultType(
      resultShape, cast<MemRefType>(storage.getType()), offsets, sizes,
      strides);
  return memref::SubViewOp::create(rewriter, loc, resultType, storage, offsets,
                                   sizes, strides)
      .getResult();
}

struct StoreOpPattern : public OpConversionPattern<StoreOp> {
  using OneToNOpAdaptor =
      typename OpConversionPattern<StoreOp>::OneToNOpAdaptor;

  StoreOpPattern(const TypeConverter& converter, MLIRContext* context,
                 const PreprocessingStorageLayoutAnalysis& analysis)
      : OpConversionPattern(converter, context), analysis(analysis) {}

  LogicalResult matchAndRewrite(
      StoreOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto storageType =
        cast<PreprocessingStorageType>(op.getStorage().getType());
    FailureOr<SiteIndex> site =
        resolveSite(op, storageType, op.getElementType(), op.getSiteId(),
                    adaptor.getIndices(), analysis, rewriter);
    if (failed(site)) return failure();
    ValueRange storageValues = adaptor.getStorage();
    if (site->elementIndex >= static_cast<int>(storageValues.size()))
      return op->emitOpError() << "storage index out of bounds";

    Value value = adaptor.getValue().front();
    if (isMoveOnlyPayload(op.getElementType())) {
      Value slot = makeRankZeroSubview(rewriter, op.getLoc(),
                                       storageValues[site->elementIndex],
                                       site->linearIndex);
      // This is a destination constraint, not a payload copy. The standard
      // empty-tensor-elimination pass redirects the producer's tensor.empty
      // destination to this slot before One-Shot Bufferize. If it cannot do
      // so, bufferization leaves a memcpy that the EmitC lowering rejects for
      // move-only payloads.
      bufferization::MaterializeInDestinationOp::create(
          rewriter, op.getLoc(), TypeRange{}, value, slot,
          rewriter.getUnitAttr(), rewriter.getUnitAttr());
      rewriter.eraseOp(op);
      return success();
    }
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType());
        tensorType && tensorType.getRank() == 0)
      value =
          tensor::ExtractOp::create(rewriter, op.getLoc(), value, ValueRange{});
    memref::StoreOp::create(rewriter, op.getLoc(), value,
                            storageValues[site->elementIndex],
                            site->linearIndex);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  const PreprocessingStorageLayoutAnalysis& analysis;
};

struct LoadOpPattern : public OpConversionPattern<LoadOp> {
  using OneToNOpAdaptor = typename OpConversionPattern<LoadOp>::OneToNOpAdaptor;

  LoadOpPattern(const TypeConverter& converter, MLIRContext* context,
                const PreprocessingStorageLayoutAnalysis& analysis)
      : OpConversionPattern(converter, context), analysis(analysis) {}

  LogicalResult matchAndRewrite(
      LoadOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto storageType =
        cast<PreprocessingStorageType>(op.getStorage().getType());
    FailureOr<SiteIndex> site =
        resolveSite(op, storageType, op.getElementType(), op.getSiteId(),
                    adaptor.getIndices(), analysis, rewriter);
    if (failed(site)) return failure();
    ValueRange storageValues = adaptor.getStorage();
    if (site->elementIndex >= static_cast<int>(storageValues.size()))
      return op->emitOpError() << "storage index out of bounds";

    if (isMoveOnlyPayload(op.getElementType())) {
      Value slot = makeRankZeroSubview(rewriter, op.getLoc(),
                                       storageValues[site->elementIndex],
                                       site->linearIndex);
      // The storage owns the move-only payload. Cheddar consumers can borrow
      // it directly through their tensor-or-memref input operands; rebuilding
      // a tensor value here would ask One-Shot Bufferize to copy the payload.
      for (OpOperand& use :
           llvm::make_early_inc_range(op.getResult().getUses())) {
        Operation* user = use.getOwner();
        auto dpsUser = dyn_cast<DestinationStyleOpInterface>(user);
        if (!dpsUser ||
            user->getName().getDialectNamespace() !=
                cheddar::CheddarDialect::getDialectNamespace() ||
            dpsUser.isDpsInit(&use)) {
          return op->emitOpError()
                 << "move-only preprocessing loads must be borrowed by a "
                    "Cheddar DPS input operand";
        }
        rewriter.modifyOpInPlace(user, [&] { use.set(slot); });
      }
      rewriter.eraseOp(op);
      return success();
    }

    Value loaded = memref::LoadOp::create(rewriter, op.getLoc(),
                                          storageValues[site->elementIndex],
                                          site->linearIndex);
    if (auto tensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
        tensorType && tensorType.getRank() == 0) {
      auto result = tensor::FromElementsOp::create(
          rewriter, op.getLoc(), tensorType, ValueRange{loaded});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, loaded);
    }
    return success();
  }

 private:
  const PreprocessingStorageLayoutAnalysis& analysis;
};

struct PreprocessingToCheddar
    : impl::PreprocessingToCheddarBase<PreprocessingToCheddar> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!containsDialects<PreprocessingDialect>(module)) return;

    PreprocessingStorageLayoutAnalysis analysis(module);
    if (!analysis.isValid()) return signalPassFailure();
    if (analysis.getTotalSizes().empty()) {
      bool hasStorageOps = false;
      module.walk([&](Operation* op) {
        hasStorageOps |= isa<EmptyOp, StoreOp, LoadOp>(op);
      });
      if (!hasStorageOps) return;
      module->emitOpError()
          << "split-preprocessing was run, but preprocessing-to-cheddar "
             "determined there are no plaintexts to preprocess";
      return signalPassFailure();
    }

    CheddarPreprocessingTypeConverter typeConverter(analysis);
    ConversionTarget target(getContext());
    target.addIllegalDialect<PreprocessingDialect>();
    // Externalized constants are already in the generic backend-neutral form;
    // their target lowering happens after bufferization.
    target.addLegalOp<LoadResourceOp>();
    target.addLegalDialect<bufferization::BufferizationDialect,
                           memref::MemRefDialect, tensor::TensorDialect,
                           arith::ArithDialect, affine::AffineDialect,
                           func::FuncDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<EmptyOpPattern>(typeConverter, &getContext());
    patterns.add<StoreOpPattern, LoadOpPattern>(typeConverter, &getContext(),
                                                analysis);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

}  // namespace mlir::heir::preprocessing
