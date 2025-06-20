#include "lib/Dialect/Openfhe/Conversions/OpenfheToScheme/OpenfheToScheme.h"

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Utils/RewriteUtils/RewriteUtils.h"
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::openfhe {

#define GEN_PASS_DEF_OPENFHETOSCHEME
#include "lib/Dialect/Openfhe/Conversions/OpenfheToScheme/OpenfheToScheme.h.inc"

namespace {

template <typename SrcOp, typename DstOp>
struct ConvertSimpleOp : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands(op->getOperands().begin() + 1,
                                op->getOperands().end());
    rewriter.replaceOpWithNewOp<DstOp>(op, op->getResultTypes(), operands,
                                       op->getAttrs());
    return success();
  }
};

struct ConvertEncryptOp : public OpRewritePattern<openfhe::EncryptOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::EncryptOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<lwe::RLWEEncryptOp>(
        op, op.getCiphertext().getType(), op.getPlaintext(),
        op.getEncryptionKey());
    return success();
  }
};

struct ConvertDecryptOp : public OpRewritePattern<openfhe::DecryptOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::DecryptOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<lwe::RLWEDecryptOp>(
        op, op.getPlaintext().getType(), op.getCiphertext(),
        op.getPrivateKey());
    return success();
  }
};

struct ConvertMakePackedPlaintextOp
    : public OpRewritePattern<openfhe::MakePackedPlaintextOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::MakePackedPlaintextOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getPlaintext().getType();
    rewriter.replaceOpWithNewOp<lwe::RLWEEncodeOp>(
        op, op.getValue(), type.getPlaintextSpace().getEncoding(),
        type.getPlaintextSpace().getRing());
    return success();
  }
};

struct ConvertMakeCKKSPackedPlaintextOp
    : public OpRewritePattern<openfhe::MakeCKKSPackedPlaintextOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::MakeCKKSPackedPlaintextOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getPlaintext().getType();
    rewriter.replaceOpWithNewOp<lwe::RLWEEncodeOp>(
        op, op.getValue(), type.getPlaintextSpace().getEncoding(),
        type.getPlaintextSpace().getRing());
    return success();
  }
};

template <typename DstOp>
struct ConvertRotOp : public OpRewritePattern<openfhe::RotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::RotOp op,
                                PatternRewriter &rewriter) const override {
    auto indexAttr = op.getIndexAttr();
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getType(), op.getCiphertext(),
                                       indexAttr);
    return success();
  }
};

template <typename DstOp>
struct ConvertRelinOp : public OpRewritePattern<openfhe::RelinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::RelinOp op,
                                PatternRewriter &rewriter) const override {
    auto ctType = op.getCiphertext().getType();
    int dim = ctType.getCiphertextSpace().getSize();
    SmallVector<int32_t> fromBasis;
    for (int i = 0; i < dim; ++i) fromBasis.push_back(i);
    SmallVector<int32_t> toBasis = {0, 1};
    auto fromAttr = rewriter.getDenseI32ArrayAttr(fromBasis);
    auto toAttr = rewriter.getDenseI32ArrayAttr(toBasis);
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getType(), op.getCiphertext(),
                                       fromAttr, toAttr);
    return success();
  }
};

template <typename DstOp>
struct ConvertModReduceOp : public OpRewritePattern<openfhe::ModReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::ModReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto toRing = op.getOutput().getType().getCiphertextSpace().getRing();
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getType(), op.getCiphertext(),
                                       toRing);
    return success();
  }
};

template <typename DstOp>
struct ConvertLevelReduceOp : public OpRewritePattern<openfhe::LevelReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::LevelReduceOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getType(), op.getCiphertext(),
                                       op.getLevelToDropAttr());
    return success();
  }
};

struct ConvertBootstrapOp : public OpRewritePattern<openfhe::BootstrapOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(openfhe::BootstrapOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ckks::BootstrapOp>(op, op.getType(),
                                                   op.getCiphertext());
    return success();
  }
};

}  // namespace

struct OpenfheToScheme : public impl::OpenfheToSchemeBase<OpenfheToScheme> {
  using OpenfheToSchemeBase::OpenfheToSchemeBase;

  void runOnOperation() override {
  MLIRContext *context = &getContext();
  auto module = getOperation();

  std::string schemeName = scheme;
  if (schemeName.empty()) {
    if (moduleIsCKKS(module))
      schemeName = "ckks";
    else if (moduleIsBGV(module) || moduleIsBFV(module))
      schemeName = "bgv";
  }

  RewritePatternSet patterns(context);

  // Operations common to all schemes
  patterns.add<ConvertEncryptOp, ConvertDecryptOp, ConvertMakePackedPlaintextOp,
               ConvertMakeCKKSPackedPlaintextOp>(context);

  if (schemeName == "ckks") {
    patterns
        .add<ConvertSimpleOp<openfhe::AddOp, ckks::AddOp>,
             ConvertSimpleOp<openfhe::SubOp, ckks::SubOp>,
             ConvertSimpleOp<openfhe::MulNoRelinOp, ckks::MulOp>,
             ConvertSimpleOp<openfhe::NegateOp, ckks::NegateOp>,
             ConvertSimpleOp<openfhe::AddPlainOp, ckks::AddPlainOp>,
             ConvertSimpleOp<openfhe::SubPlainOp, ckks::SubPlainOp>,
             ConvertSimpleOp<openfhe::MulPlainOp, ckks::MulPlainOp>,
             ConvertRotOp<ckks::RotateOp>, ConvertRelinOp<ckks::RelinearizeOp>,
             ConvertModReduceOp<ckks::RescaleOp>,
             ConvertLevelReduceOp<ckks::LevelReduceOp>, ConvertBootstrapOp>(
            context);
  } else {
    patterns.add<ConvertSimpleOp<openfhe::AddOp, bgv::AddOp>,
                 ConvertSimpleOp<openfhe::SubOp, bgv::SubOp>,
                 ConvertSimpleOp<openfhe::MulNoRelinOp, bgv::MulOp>,
                 ConvertSimpleOp<openfhe::NegateOp, bgv::NegateOp>,
                 ConvertSimpleOp<openfhe::AddPlainOp, bgv::AddPlainOp>,
                 ConvertSimpleOp<openfhe::SubPlainOp, bgv::SubPlainOp>,
                 ConvertSimpleOp<openfhe::MulPlainOp, bgv::MulPlainOp>,
                 ConvertRotOp<bgv::RotateColumnsOp>,
                 ConvertRelinOp<bgv::RelinearizeOp>,
                 ConvertModReduceOp<bgv::ModulusSwitchOp>,
                 ConvertLevelReduceOp<bgv::LevelReduceOp>>(context);
  }

  walkAndApplyPatterns(module, std::move(patterns));
  }
};

}  // namespace mlir::heir::openfhe
