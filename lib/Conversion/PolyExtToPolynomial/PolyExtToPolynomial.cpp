#include "lib/Conversion/PolyExtToPolynomial/PolyExtToPolynomial.h"

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_POLYEXTTOPOLYNOMIAL
#include "lib/Conversion/PolyExtToPolynomial/PolyExtToPolynomial.h.inc"

// Remove this class if no type conversions are necessary
class PolyExtToPolynomialTypeConverter : public TypeConverter {
 public:
  PolyExtToPolynomialTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    // FIXME: implement, replace FooType with the type that needs
    // to be converted or remove this class
    // addConversion([ctx](FooType type) -> Type { return type; });
  }
};

// // FIXME: rename to Convert<OpName>Op
// struct ConvertFooOp : public OpConversionPattern<FooOp> {
//   ConvertFooOp(mlir::MLIRContext *context)
//       : OpConversionPattern<FooOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       FooOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     // FIXME: implement
//     return failure();
//   }
// };

struct PolyExtToPolynomial
    : public impl::PolyExtToPolynomialBase<PolyExtToPolynomial> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    PolyExtToPolynomialTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<polynomial::PolynomialDialect>();
    target.addIllegalDialect<poly_ext::PolyExtDialect>();

    // patterns.add<ConvertFooOp>(typeConverter, context);

    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir
