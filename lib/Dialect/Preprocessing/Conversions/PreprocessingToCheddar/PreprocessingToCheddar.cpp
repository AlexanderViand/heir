#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToCheddar/PreprocessingToCheddar.h"

#include <cstdint>
#include <optional>

#include "lib/Analysis/PreprocessingStorageLayoutAnalysis/PreprocessingStorageLayoutAnalysis.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"  // for the pass's dependentDialects (getDependentDialects)
#include "lib/Dialect/Preprocessing/Conversions/Util.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingDialect.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingOps.h"
#include "lib/Dialect/Preprocessing/IR/PreprocessingTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace preprocessing {

#define GEN_PASS_DEF_PREPROCESSINGTOCHEDDAR
#include "lib/Dialect/Preprocessing/Conversions/PreprocessingToCheddar/PreprocessingToCheddar.h.inc"

namespace {

// CHEDDAR represents a single plaintext as a rank-0
// `tensor<!cheddar.plaintext>`. A memref element type cannot be a tensor, so
// the storage memref holds the unwrapped scalar element; store/load bridge via
// tensor.extract/from_elements.
Type unwrapRank0Tensor(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t))
    if (rt.getRank() == 0) return rt.getElementType();
  return t;
}

FailureOr<int> getElementTypeIndex(PreprocessingStorageType storageTy,
                                   Type elementTy) {
  auto elementTypes = storageTy.getElementTypes();
  auto it = llvm::find(elementTypes, elementTy);
  if (it == elementTypes.end()) return failure();
  return std::distance(elementTypes.begin(), it);
}

// 1-to-N: a PreprocessingStorageType (one SSA value holding plaintexts of one
// or more element types) maps to one flat memref per element type, sized by the
// layout analysis, with the rank-0-tensor element unwrapped to its scalar.
class CheddarPreprocessingTypeConverter : public TypeConverter {
 public:
  explicit CheddarPreprocessingTypeConverter(
      const PreprocessingStorageLayoutAnalysis& analysis) {
    addConversion([](Type type) { return type; });
    addConversion([&analysis](PreprocessingStorageType type,
                              SmallVectorImpl<Type>& results) {
      for (Type elementTy : type.getElementTypes()) {
        int64_t totalSize = analysis.getTotalSize(elementTy).value_or(0);
        results.push_back(
            MemRefType::get({totalSize}, unwrapRank0Tensor(elementTy)));
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
    SmallVector<Value> allocOps;
    for (Type memrefTy : resultTypes)
      allocOps.push_back(memref::AllocOp::create(rewriter, op.getLoc(),
                                                 cast<MemRefType>(memrefTy)));
    rewriter.replaceOpWithMultiple(op, {allocOps});
    return success();
  }
};

// Resolve (storage element index, flat linear index) shared by store/load.
struct SiteIndex {
  int elemIndex;
  Value linearIndex;
};
static FailureOr<SiteIndex> resolveSite(
    Operation* op, PreprocessingStorageType storageTy, Type elementType,
    uint32_t siteId, ArrayRef<ValueRange> indexRanges,
    const PreprocessingStorageLayoutAnalysis& analysis,
    ConversionPatternRewriter& rewriter) {
  FailureOr<int> elemIndex = getElementTypeIndex(storageTy, elementType);
  if (failed(elemIndex)) return failure();
  SmallVector<Value> flatIndices;
  for (ValueRange r : indexRanges) {
    if (r.size() != 1)
      return op->emitOpError() << "Expected exactly one SSA value per index";
    flatIndices.push_back(r.front());
  }
  std::optional<SiteLayout> layout = analysis.getLayout(elementType, siteId);
  if (!layout.has_value())
    return op->emitOpError() << "Missing layout for site ID " << siteId;
  FailureOr<Value> index = preprocessing::getLinearIndex(
      rewriter, op->getLoc(), op, layout->offset, flatIndices);
  if (failed(index)) return failure();
  return SiteIndex{*elemIndex, index.value()};
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
    auto storageTy = cast<PreprocessingStorageType>(op.getStorage().getType());
    FailureOr<SiteIndex> site =
        resolveSite(op, storageTy, op.getElementType(), op.getSiteId(),
                    adaptor.getIndices(), analysis, rewriter);
    if (failed(site)) return failure();
    ValueRange storageValues = adaptor.getStorage();
    if (site->elemIndex >= (int)storageValues.size())
      return op->emitOpError() << "Storage index out of bounds";

    // The stored payload is a rank-0 tensor<!cheddar.plaintext>; unwrap to the
    // scalar that the memref holds.
    Value value = adaptor.getValue().front();
    if (auto rt = dyn_cast<RankedTensorType>(value.getType());
        rt && rt.getRank() == 0)
      value =
          tensor::ExtractOp::create(rewriter, op.getLoc(), value, ValueRange{});
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, value, storageValues[site->elemIndex], site->linearIndex);
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
    auto storageTy = cast<PreprocessingStorageType>(op.getStorage().getType());
    FailureOr<SiteIndex> site =
        resolveSite(op, storageTy, op.getElementType(), op.getSiteId(),
                    adaptor.getIndices(), analysis, rewriter);
    if (failed(site)) return failure();
    ValueRange storageValues = adaptor.getStorage();
    if (site->elemIndex >= (int)storageValues.size())
      return op->emitOpError() << "Storage index out of bounds";

    Value loaded = memref::LoadOp::create(rewriter, op.getLoc(),
                                          storageValues[site->elemIndex],
                                          site->linearIndex);
    // Re-wrap the scalar payload into the rank-0 tensor the consumers expect.
    if (auto rt = dyn_cast<RankedTensorType>(op.getResult().getType());
        rt && rt.getRank() == 0) {
      rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, rt,
                                                          ValueRange{loaded});
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
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }
    if (analysis.getTotalSizes().empty()) {
      module->emitOpError()
          << "split-preprocessing was run, but preprocessing-to-cheddar "
             "determined there are no plaintexts to preprocess.";
      signalPassFailure();
      return;
    }

    CheddarPreprocessingTypeConverter typeConverter(analysis);

    ConversionTarget target(getContext());
    target.addIllegalDialect<PreprocessingDialect>();
    target.addLegalDialect<memref::MemRefDialect, tensor::TensorDialect,
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

}  // namespace preprocessing
}  // namespace heir
}  // namespace mlir
