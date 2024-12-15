#include "lib/Dialect/BGV/Conversions/BGVToOpenfhe/BGVToOpenfhe.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Dialect/LWE/Conversions/RlweToOpenfhe/RlweToOpenfhe.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOOPENFHE
#include "lib/Dialect/BGV/Conversions/BGVToOpenfhe/BGVToOpenfhe.h.inc"

using ConvertNegateOp = ConvertRlweUnaryOp<lwe::RNegateOp, openfhe::NegateOp>;
using ConvertAddOp = ConvertRlweBinOp<lwe::RAddOp, openfhe::AddOp>;
using ConvertSubOp = ConvertRlweBinOp<lwe::RSubOp, openfhe::SubOp>;
using ConvertMulOp = ConvertRlweBinOp<lwe::RMulOp, openfhe::MulNoRelinOp>;
using ConvertAddPlainOp =
    ConvertRlweCiphertextPlaintextOp<AddPlainOp, openfhe::AddPlainOp>;
using ConvertMulPlainOp =
    ConvertRlweCiphertextPlaintextOp<MulPlainOp, openfhe::MulPlainOp>;
using ConvertRotateOp = ConvertRlweRotateOp<RotateOp>;
using ConvertRelinOp = ConvertRlweRelinOp<RelinearizeOp>;

struct ConvertModulusSwitchOp : public OpConversionPattern<ModulusSwitchOp> {
  ConvertModulusSwitchOp(mlir::MLIRContext *context)
      : OpConversionPattern<ModulusSwitchOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModulusSwitchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = getContextualCryptoContext(op.getOperation());
    if (failed(result)) return result;

    Value cryptoContext = result.value();
    rewriter.replaceOp(op, rewriter.create<openfhe::ModReduceOp>(
                               op.getLoc(), op.getOutput().getType(),
                               cryptoContext, adaptor.getInput()));
    return success();
  }
};

struct BGVToOpenfhe : public impl::BGVToOpenfheBase<BGVToOpenfhe> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToOpenfheTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<openfhe::OpenfheDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target.addIllegalDialect<lwe::LWEDialect>();
    // We can keep/ignore the following ops, which the emitter can handle??
    target.addLegalOp<lwe::ReinterpretUnderlyingTypeOp, lwe::RLWEDecodeOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg = op.getFunctionType().getNumInputs() > 0 &&
                                 mlir::isa<openfhe::CryptoContextType>(
                                     *op.getFunctionType().getInputs().begin());
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsDialects<lwe::LWEDialect, bgv::BGVDialect>(op) ||
              hasCryptoContextArg);
    });

    patterns.add<AddCryptoContextArg<bgv::BGVDialect>, ConvertAddOp,
                 ConvertSubOp, ConvertAddPlainOp, ConvertMulOp,
                 ConvertMulPlainOp, ConvertNegateOp, ConvertRotateOp,
                 ConvertRelinOp, ConvertModulusSwitchOp, lwe::ConvertEncodeOp,
                 lwe::ConvertEncryptOp, lwe::ConvertDecryptOp>(typeConverter,
                                                               context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
