#include "lib/Conversion/PolyExtToPolynomial/PolyExtToPolynomial.h"

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "lib/Dialect/PolyExt/IR/PolyExtOps.h"
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         //from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
namespace mlir::heir {

#define GEN_PASS_DEF_POLYEXTTOPOLYNOMIAL
#include "lib/Conversion/PolyExtToPolynomial/PolyExtToPolynomial.h.inc"

struct ConvertGadgetProduct
    : public OpConversionPattern<poly_ext::GadgetProduct> {
  ConvertGadgetProduct(mlir::MLIRContext *context)
      : OpConversionPattern<poly_ext::GadgetProduct>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      poly_ext::GadgetProduct op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // TODO: Make modulus extension and digit decomposition toggleable

    // TODO: Make extension modulus a pass parameter?
    APInt p(32, 12345);

    // Modulus Extension
    auto polynomialType = llvm::cast<polynomial::PolynomialType>(
        getElementTypeOrSelf(adaptor.getLhs()));
    auto ring = polynomialType.getRing();

    auto extendedRing = polynomial::RingAttr::get(
        ring.getCoefficientType(),
        // TODO: Update bitwidth correctly!
        rewriter.getI32IntegerAttr(
            (ring.getCoefficientModulus().getValue() * p).getLimitedValue()),
        ring.getPolynomialModulus());
    auto extendedType =
        polynomial::PolynomialType::get(rewriter.getContext(), extendedRing);
    auto plaintextModulus = rewriter.getI32IntegerAttr(
        42);  // TODO: Where to get plaintext mod from?

    auto lhsExt = b.create<poly_ext::CModSwitchOp>(
        extendedType, adaptor.getLhs(), plaintextModulus);
    // TODO: instead: create a new ksk op with P multiplied in and mod PQ
    auto rhsExt = b.create<poly_ext::CModSwitchOp>(
        extendedType, adaptor.getRhs(), plaintextModulus);

    // TODO: do we just rely on the convention that ksk should be rhs?

    // auto rhsExtTimesP = b.create<polynomial::MulScalarOp>(
    // rhsExt, rewriter.getI32IntegerAttr(p.getLimitedValue()));

    // TODO: Digit Decomposition

    // Actual product
    auto mul = b.create<polynomial::MulOp>(lhsExt, rhsExt);

    // Delta "correction" factor?

    // Modswitch down
    auto result =
        b.create<poly_ext::CModSwitchOp>(polynomialType, mul, plaintextModulus);

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct PolyExtToPolynomial
    : public impl::PolyExtToPolynomialBase<PolyExtToPolynomial> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<polynomial::PolynomialDialect>();
    target.addIllegalDialect<poly_ext::PolyExtDialect>();

    // TODO: Add lowering for cmod_switch
    target.addLegalOp<poly_ext::CModSwitchOp>();

    // TODO: Add lowering for ksk_delta
    target.addLegalOp<poly_ext::KSKDelta>();

    // TODO: Add lowering for digit_decompose
    target.addLegalOp<poly_ext::DigitDecompoose>();

    patterns.add<ConvertGadgetProduct>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
