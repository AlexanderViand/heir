#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Utils/Approximation/CaratheodoryFejer.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "llvm/include/llvm/ADT/APFloat.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "polynomial-approximation"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_POLYNOMIALAPPROXIMATION
#include "lib/Transforms/PolynomialApproximation/PolynomialApproximation.h.inc"

constexpr int64_t kDefaultDegree = 5;
constexpr double kDefaultDomainLower = -1.0;
constexpr double kDefaultDomainUpper = 1.0;

constexpr double kDefaultPositiveRangeLower = 0.1;
constexpr double kDefaultPositiveRangeUpper = 2.0;

constexpr double kDefaultNonNegativeRangeLower = 0.0;
constexpr double kDefaultNonNegativeRangeUpper = 2.0;

using polynomial::ChebyshevPolynomial;
using polynomial::EvalOp;
using polynomial::PolynomialType;
using polynomial::RingAttr;
using polynomial::TypedChebyshevPolynomialAttr;

// Emit an error on `op` and return failure
// if any coefficient in `poly` is non-finite (NaN or infinity).
LogicalResult checkApproximationFinite(Operation* op,
                                       const ChebyshevPolynomial& poly) {
  for (const APFloat& c : poly.getTerms()) {
    if (!c.isFinite()) {
      return op->emitError()
             << "polynomial approximation produced a non-finite coefficient "
                "(NaN or infinity); this might mean the function is not "
                "defined on the requested domain. Set `domain_lower`/"
                "`domain_upper` on the op to a domain where the function is "
                "defined.";
    }
  }
  return success();
}

inline APFloat absf(const APFloat& x) {
  return APFloat(std::abs(x.convertToDouble()));
}
inline APFloat acos(const APFloat& x) {
  return APFloat(std::acos(x.convertToDouble()));
}
inline APFloat acosh(const APFloat& x) {
  return APFloat(std::acosh(x.convertToDouble()));
}
inline APFloat asin(const APFloat& x) {
  return APFloat(std::asin(x.convertToDouble()));
}
inline APFloat asinh(const APFloat& x) {
  return APFloat(std::asinh(x.convertToDouble()));
}
inline APFloat atan(const APFloat& x) {
  return APFloat(std::atan(x.convertToDouble()));
}
inline APFloat atanh(const APFloat& x) {
  return APFloat(std::atanh(x.convertToDouble()));
}
inline APFloat cbrt(const APFloat& x) {
  return APFloat(std::cbrt(x.convertToDouble()));
}
inline APFloat ceil(const APFloat& x) {
  return APFloat(std::ceil(x.convertToDouble()));
}
inline APFloat cos(const APFloat& x) {
  return APFloat(std::cos(x.convertToDouble()));
}
inline APFloat cosh(const APFloat& x) {
  return APFloat(std::cosh(x.convertToDouble()));
}
inline APFloat erf(const APFloat& x) {
  return APFloat(std::erf(x.convertToDouble()));
}
inline APFloat erfc(const APFloat& x) {
  return APFloat(std::erfc(x.convertToDouble()));
}
inline APFloat exp(const APFloat& x) {
  return APFloat(std::exp(x.convertToDouble()));
}
inline APFloat exp2(const APFloat& x) {
  return APFloat(std::exp2(x.convertToDouble()));
}
inline APFloat expm1(const APFloat& x) {
  return APFloat(std::expm1(x.convertToDouble()));
}
inline APFloat floor(const APFloat& x) {
  return APFloat(std::floor(x.convertToDouble()));
}
inline APFloat log(const APFloat& x) {
  return APFloat(std::log(x.convertToDouble()));
}
inline APFloat log10(const APFloat& x) {
  return APFloat(std::log10(x.convertToDouble()));
}
inline APFloat log1p(const APFloat& x) {
  return APFloat(std::log1p(x.convertToDouble()));
}
inline APFloat log2(const APFloat& x) {
  return APFloat(std::log2(x.convertToDouble()));
}
inline APFloat round(const APFloat& x) {
  return APFloat(std::round(x.convertToDouble()));
}
// not available on apple cmath?
// inline APFloat _roundeven(const APFloat &x) {
//   return APFloat(roundeven(x.convertToDouble()));
// }
inline APFloat rsqrt(const APFloat& x) {
  return APFloat(1.0 / std::sqrt(x.convertToDouble()));
}
inline APFloat sin(const APFloat& x) {
  return APFloat(std::sin(x.convertToDouble()));
}
inline APFloat sinh(const APFloat& x) {
  return APFloat(std::sinh(x.convertToDouble()));
}
inline APFloat sqrt(const APFloat& x) {
  return APFloat(std::sqrt(x.convertToDouble()));
}
inline APFloat tan(const APFloat& x) {
  return APFloat(std::tan(x.convertToDouble()));
}
inline APFloat tanh(const APFloat& x) {
  return APFloat(std::tanh(x.convertToDouble()));
}
inline APFloat trunc(const APFloat& x) {
  return APFloat(std::trunc(x.convertToDouble()));
}
inline APFloat sign(const APFloat& x) {
  return APFloat(x.isNegative() ? -1.0 : (x.isZero() ? 0.0 : 1.0));
}
inline APFloat sigmoid(const APFloat& x) {
  return APFloat(1.0) / (APFloat(1.0 + std::exp(-x.convertToDouble())));
}

// Binary ops
inline APFloat atan2(const APFloat& lhs, const APFloat& rhs) {
  return APFloat(std::atan2(lhs.convertToDouble(), rhs.convertToDouble()));
}
inline APFloat fpowi(const APFloat& lhs, const APFloat& rhs) {
  return APFloat(std::pow(lhs.convertToDouble(), rhs.convertToDouble()));
}
inline APFloat powf(const APFloat& lhs, const APFloat& rhs) {
  return APFloat(std::pow(lhs.convertToDouble(), rhs.convertToDouble()));
}
inline APFloat copysign(const APFloat& lhs, const APFloat& rhs) {
  return APFloat::copySign(lhs, rhs);
}

// The user of these ops (the polynomial approximation routines) don't see the
// types of the possibly constant operand, which may be an f32 while the caller
// is using APFloats with f64 semnatics.  So we convert both operands to double
// precision and avoid this.  A better approach may be to have the polynomial
// approximation routines take as input the float semantics used to create
// APFloats internally.
inline APFloat maxf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::maximum(lhsConverted, rhsConverted);
}
inline APFloat minf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::minimum(lhsConverted, rhsConverted);
}
inline APFloat maxnumf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::maximumnum(lhsConverted, rhsConverted);
}
inline APFloat minnumf(const APFloat& lhs, const APFloat& rhs) {
  APFloat lhsConverted = APFloat(lhs.convertToDouble());
  APFloat rhsConverted = APFloat(rhs.convertToDouble());
  return llvm::minimumnum(lhsConverted, rhsConverted);
}

template <typename OpTy>
struct ConvertUnaryOp : public OpRewritePattern<OpTy> {
  ConvertUnaryOp(mlir::MLIRContext* context,
                 const std::function<APFloat(APFloat)>& cppFunc,
                 double lower = kDefaultDomainLower,
                 double upper = kDefaultDomainUpper)
      : OpRewritePattern<OpTy>(context, /*benefit=*/1),
        cppFunc(cppFunc),
        lower(lower),
        upper(upper) {}

 public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = op.getContext();
    IntegerAttr degreeAttr = op->hasAttr("degree")
                                 ? cast<IntegerAttr>(op->getAttr("degree"))
                                 : rewriter.getI32IntegerAttr(kDefaultDegree);
    if (op->hasAttr("domain_lower") &&
        !isa<FloatAttr>(op->getAttr("domain_lower")))
      return op.emitOpError("domain_lower must be a floating-point attribute");
    if (op->hasAttr("domain_upper") &&
        !isa<FloatAttr>(op->getAttr("domain_upper")))
      return op.emitOpError("domain_upper must be a floating-point attribute");
    FloatAttr domainLowerAttr =
        op->hasAttr("domain_lower")
            ? cast<FloatAttr>(op->getAttr("domain_lower"))
            : rewriter.getF64FloatAttr(lower);
    FloatAttr domainUpperAttr =
        op->hasAttr("domain_upper")
            ? cast<FloatAttr>(op->getAttr("domain_upper"))
            : rewriter.getF64FloatAttr(upper);
    polynomial::ChebyshevPolynomial poly =
        approximation::caratheodoryFejerApproximation(
            cppFunc, degreeAttr.getInt(),
            domainLowerAttr.getValue().convertToDouble(),
            domainUpperAttr.getValue().convertToDouble());
    if (failed(checkApproximationFinite(op, poly))) return failure();
    PolynomialType polyType =
        PolynomialType::get(ctx, RingAttr::get(Float64Type::get(ctx)));
    TypedChebyshevPolynomialAttr polyAttr =
        TypedChebyshevPolynomialAttr::get(polyType, poly);
    auto evalOp =
        rewriter.replaceOpWithNewOp<EvalOp>(op, polyAttr, op.getOperand());
    // These attributes need to be preserved when the polynomial is in the
    // Chebyshev basis, so that later passes can apply domain rescaling
    // properly.
    evalOp->setAttr("domain_lower", domainLowerAttr);
    evalOp->setAttr("domain_upper", domainUpperAttr);

    return success();
  }

 private:
  std::function<APFloat(APFloat)> cppFunc;
  double lower;
  double upper;
};

// Return a single value defining a constant (either a splatted tensor or a
// scalar value), or else a failure if the value is non-constant or defined by
// a non-splatted constant.
FailureOr<APFloat> getSingleValueOrSplat(Value value) {
  LLVM_DEBUG(llvm::dbgs() << "Checking if value " << value
                          << " is a constant\n");
  TypedAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return failure();
  }

  if (auto elementsAttr = dyn_cast<ElementsAttr>(attr)) {
    if (elementsAttr.isSplat()) {
      return elementsAttr.getSplatValue<APFloat>();
    }
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValue();
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return APFloat(APFloat::IEEEdouble(), intAttr.getValue());
  }

  return failure();
}

template <typename OpTy>
struct ConvertBinaryConstOp : public OpRewritePattern<OpTy> {
  ConvertBinaryConstOp(mlir::MLIRContext* context,
                       const std::function<APFloat(APFloat, APFloat)>& cppFunc,
                       double lower = kDefaultDomainLower,
                       double upper = kDefaultDomainUpper)
      : OpRewritePattern<OpTy>(context, /*benefit=*/1),
        cppFunc(cppFunc),
        lower(lower),
        upper(upper) {}

 public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumOperands() != 2) {
      return op.emitOpError("Expected 2 operands; should be unreachable!");
    }

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhsConstResult = getSingleValueOrSplat(lhs);
    auto rhsConstResult = getSingleValueOrSplat(rhs);
    if (failed(lhsConstResult) && failed(rhsConstResult)) {
      // Neither operand is a single-valued constant, so we can't approximate.
      // If it's a constant but defined by a non-splatted dense elements attr,
      // we'd need to first run a pass like elementwise-to-affine to unpack the
      // tensor into individual scalars, then loop unroll or else make this pass
      // depend on SCCP analysis to get the constant here.
      return rewriter.notifyMatchFailure(
          op, "neither operand is a single-valued constant");
    }
    bool lhsIsConstant = succeeded(lhsConstResult);
    APFloat constValue =
        lhsIsConstant ? lhsConstResult.value() : rhsConstResult.value();
    Value nonConstOperand = lhsIsConstant ? rhs : lhs;

    // cppFunc is a binary op, so we need to give it the constant value to
    // convert it to a unary op.
    std::function<APFloat(APFloat)> unaryFunc;
    if (lhsIsConstant) {
      unaryFunc = [this, constValue](const APFloat& x) {
        return cppFunc(constValue, x);
      };
    } else {
      unaryFunc = [this, constValue](const APFloat& x) {
        return cppFunc(x, constValue);
      };
    }

    MLIRContext* ctx = op.getContext();
    IntegerAttr degreeAttr = op->hasAttr("degree")
                                 ? cast<IntegerAttr>(op->getAttr("degree"))
                                 : rewriter.getI32IntegerAttr(kDefaultDegree);
    if (op->hasAttr("domain_lower") &&
        !isa<FloatAttr>(op->getAttr("domain_lower")))
      return op.emitOpError("domain_lower must be a floating-point attribute");
    if (op->hasAttr("domain_upper") &&
        !isa<FloatAttr>(op->getAttr("domain_upper")))
      return op.emitOpError("domain_upper must be a floating-point attribute");
    FloatAttr domainLowerAttr =
        op->hasAttr("domain_lower")
            ? cast<FloatAttr>(op->getAttr("domain_lower"))
            : rewriter.getF64FloatAttr(lower);
    FloatAttr domainUpperAttr =
        op->hasAttr("domain_upper")
            ? cast<FloatAttr>(op->getAttr("domain_upper"))
            : rewriter.getF64FloatAttr(upper);
    ChebyshevPolynomial poly = approximation::caratheodoryFejerApproximation(
        unaryFunc, degreeAttr.getInt(),
        domainLowerAttr.getValue().convertToDouble(),
        domainUpperAttr.getValue().convertToDouble());
    if (failed(checkApproximationFinite(op, poly))) return failure();
    PolynomialType polyType =
        PolynomialType::get(ctx, RingAttr::get(Float64Type::get(ctx)));
    TypedChebyshevPolynomialAttr polyAttr =
        TypedChebyshevPolynomialAttr::get(polyType, poly);
    auto evalOp =
        rewriter.replaceOpWithNewOp<EvalOp>(op, polyAttr, nonConstOperand);
    // These attributes need to be preserved when the polynomial is in the
    // Chebyshev basis, so that later passes can apply domain rescaling
    // properly.
    evalOp->setAttr("domain_lower", domainLowerAttr);
    evalOp->setAttr("domain_upper", domainUpperAttr);

    return success();
  }

 private:
  std::function<APFloat(APFloat, APFloat)> cppFunc;
  double lower;
  double upper;
};

// Use a Taylor approximation `e^x = (1 + x/2^k)^(2^k)` evaluated via
// repeated squaring. When the domain is in [-2^k, 1], this is more efficient
// in level consumption than the default polynomial approximation solver.
struct ExpOpTaylorApproximation : public OpRewritePattern<math::ExpOp> {
  ExpOpTaylorApproximation(MLIRContext* context, int64_t defaultK = 7)
      : OpRewritePattern<math::ExpOp>(context, /*benefit=*/2),
        defaultK(defaultK) {}

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    Type type = operand.getType();

    int64_t k = defaultK;
    if (op->hasAttr("degree")) {
      IntegerAttr degreeAttr = dyn_cast<IntegerAttr>(op->getAttr("degree"));
      if (degreeAttr && degreeAttr.getInt() > 0) {
        k = static_cast<int64_t>(
            std::ceil(std::log2(static_cast<double>(degreeAttr.getInt()))));
      }
    }

    double validLower = -static_cast<double>(1ULL << k);
    double validUpper = 1.0;

    if (op->hasAttr("domain_lower")) {
      FloatAttr lowerAttr = dyn_cast<FloatAttr>(op->getAttr("domain_lower"));
      if (!lowerAttr)
        return op.emitOpError(
            "domain_lower must be a floating-point attribute");
      if (lowerAttr.getValueAsDouble() < validLower) {
        return rewriter.notifyMatchFailure(
            op, "domain_lower is less than valid interval bound -2^k");
      }
    }
    if (op->hasAttr("domain_upper")) {
      FloatAttr upperAttr = dyn_cast<FloatAttr>(op->getAttr("domain_upper"));
      if (!upperAttr)
        return op.emitOpError(
            "domain_upper must be a floating-point attribute");
      if (upperAttr.getValueAsDouble() > validUpper) {
        return rewriter.notifyMatchFailure(
            op, "domain_upper exceeds valid interval bound 1.0");
      }
    }

    Type elemType =
        isa<ShapedType>(type) ? cast<ShapedType>(type).getElementType() : type;

    double inv2k = 1.0 / static_cast<double>(1ULL << k);
    TypedAttr scaleAttr;
    TypedAttr oneAttr;

    if (ShapedType shapedType = dyn_cast<ShapedType>(type)) {
      scaleAttr = DenseElementsAttr::get(
          shapedType, rewriter.getFloatAttr(elemType, inv2k));
      oneAttr = DenseElementsAttr::get(shapedType,
                                       rewriter.getFloatAttr(elemType, 1.0));
    } else {
      scaleAttr = rewriter.getFloatAttr(elemType, inv2k);
      oneAttr = rewriter.getFloatAttr(elemType, 1.0);
    }

    Value scaleConst = arith::ConstantOp::create(rewriter, loc, scaleAttr);
    Value oneConst = arith::ConstantOp::create(rewriter, loc, oneAttr);

    Value scaledX = arith::MulFOp::create(rewriter, loc, operand, scaleConst);
    Value current = arith::AddFOp::create(rewriter, loc, scaledX, oneConst);

    for (int64_t i = 0; i < k; ++i) {
      current = arith::MulFOp::create(rewriter, loc, current, current);
    }

    rewriter.replaceOp(op, current);
    return success();
  }

 private:
  int64_t defaultK;
};

// Minimax composite-sign coefficients (Chebyshev basis on [-1, 1]) for the
// degree schedule [15, 15, 27]. Lifted from orion's
// `generate_minimax_sign_coeffs([15,15,27], prec=128, logalpha=6, logerr=12)`,
// which wraps the lattigo minimax solver. The first two polynomials map
// [-1,1] -> [-1,1] converging toward sign(x); the third maps to a [0,1] step
// so that ReLU(x) = x * step(x). These are universal (function-only, not
// model-specific), so we hardcode them rather than re-solving the minimax
// problem inside HEIR.
static const std::vector<double> kCompositeSignPoly0 = {
    -0.0, 0.756018280983,  0.0,  -0.253032654524, 0.0, 0.153152108192,
    -0.0, -0.110901109874, -0.0, 0.087929151952,  0.0, -0.073912657797,
    -0.0, 0.064969979227,  0.0,  -0.436979353428};
static const std::vector<double> kCompositeSignPoly1 = {
    0.0,  1.236891150475,  0.0,  -0.398085355759, -0.0, 0.222488179803,
    -0.0, -0.142359510064, -0.0, 0.095177434385,  -0.0, -0.063848823309,
    -0.0, 0.041804868728,  0.0,  -0.040160164237};
static const std::vector<double> kCompositeSignPoly2 = {
    0.500023841858, 0.625914692879, -4.4641296e-05, -0.182119160891,
    3.664539e-05,   0.083136156201, -2.6313986e-05, -0.039259493351,
    1.6471087e-05,  0.017457883805, -8.940689e-06,  -0.007013411261,
    4.17812e-06,    0.002478481503, -1.664388e-06,  -0.000753055967,
    5.57635e-07,    0.000191995525, -1.5425e-07,    -3.9858889e-05,
    3.4315e-08,     6.462984e-06,   -5.904e-09,     -7.67161e-07,
    7.38e-10,       5.9265e-08,     -6e-11,         -2.236e-09};

// Approximates ReLU as `x * step(x / B)` where `step` is the composite-sign
// approximation (3 chained Chebyshev polys) and `B` is the input bound taken
// from the op's `domain_lower`/`domain_upper` attrs. This matches orion's
// ReLU FHE implementation and is far more accurate than a single low-degree
// polynomial fit to `max(x, 0)` (which has large kink error and extrapolates
// catastrophically outside its fit domain). Only matches the ReLU shape
// `arith.maximumf %x, 0` and is gated behind the pass's `useCompositeRelu`
// option; otherwise the generic single-polynomial ConvertBinaryConstOp path
// handles maximumf.
struct ReluViaCompositeSign : public OpRewritePattern<arith::MaximumFOp> {
  // benefit 2 > the generic ConvertBinaryConstOp benefit (1) so this wins
  // for the ReLU shape when the option is enabled.
  ReluViaCompositeSign(mlir::MLIRContext* context)
      : OpRewritePattern<arith::MaximumFOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(arith::MaximumFOp op,
                                PatternRewriter& rewriter) const override {
    // Identify the ReLU shape: one operand is a constant equal to 0.
    auto lhsConst = getSingleValueOrSplat(op.getLhs());
    auto rhsConst = getSingleValueOrSplat(op.getRhs());
    Value x;
    if (succeeded(rhsConst) && rhsConst.value().isZero()) {
      x = op.getLhs();
    } else if (succeeded(lhsConst) && lhsConst.value().isZero()) {
      x = op.getRhs();
    } else {
      return rewriter.notifyMatchFailure(op, "not a ReLU (max(x, 0)) shape");
    }

    // The ReLU may be scalar (f32) or shaped (tensor<...xf32>); in the
    // torch-linalg-to-ckks flow the maximumf operates on tensors inside a
    // secret.generic, so match on the element type and splat constants.
    Type opType = op.getType();
    Type elemType = getElementTypeOrSelf(opType);
    if (!isa<FloatType>(elemType)) {
      return rewriter.notifyMatchFailure(op, "non-float ReLU operand");
    }

    // Input bound B from the domain attrs; fall back to the default domain.
    double lower = kDefaultDomainLower;
    double upper = kDefaultDomainUpper;
    if (auto a = dyn_cast_or_null<FloatAttr>(op->getAttr("domain_lower")))
      lower = a.getValue().convertToDouble();
    if (auto a = dyn_cast_or_null<FloatAttr>(op->getAttr("domain_upper")))
      upper = a.getValue().convertToDouble();
    double bound = std::max(std::abs(lower), std::abs(upper));
    if (bound == 0.0) bound = 1.0;

    MLIRContext* ctx = op.getContext();
    Location loc = op.getLoc();
    PolynomialType polyType =
        PolynomialType::get(ctx, RingAttr::get(Float64Type::get(ctx)));

    auto makeEval = [&](Value in, const std::vector<double>& coeffs) -> Value {
      // NB: `ChebyshevPolynomial poly(ArrayRef<double>(coeffs))` would be a
      // most-vexing-parse function declaration; bind the ArrayRef to a named
      // variable first so this is unambiguously a value construction.
      ArrayRef<double> coeffRef(coeffs);
      ChebyshevPolynomial poly(coeffRef);
      auto polyAttr = TypedChebyshevPolynomialAttr::get(polyType, poly);
      // Mirror the proven call form used by ConvertUnaryOp/ConvertBinaryConstOp
      // (`replaceOpWithNewOp<EvalOp>(op, polyAttr, operand)`): pass (polyAttr,
      // value); the result type follows the input via AllTypesMatch.
      auto eval = rewriter.create<EvalOp>(loc, polyAttr, in);
      // The sign polys are fit on [-1, 1]; record that for downstream
      // domain-rescaling in LowerPolynomialEval.
      eval->setAttr("domain_lower", rewriter.getF64FloatAttr(-1.0));
      eval->setAttr("domain_upper", rewriter.getF64FloatAttr(1.0));
      return eval.getResult();
    };

    // xs = x * (1/B)  -> prescale into [-1, 1]. Build a splat constant when the
    // operand is shaped so the multiply type-checks against the tensor operand.
    auto invBFloat =
        cast<FloatAttr>(rewriter.getFloatAttr(elemType, 1.0 / bound));
    TypedAttr invBAttr =
        isa<ShapedType>(opType)
            ? cast<TypedAttr>(DenseElementsAttr::get(cast<ShapedType>(opType),
                                                     invBFloat.getValue()))
            : cast<TypedAttr>(invBFloat);
    Value invB = rewriter.create<arith::ConstantOp>(loc, invBAttr);
    Value xs = rewriter.create<arith::MulFOp>(loc, x, invB);
    // step(xs) via the 3-stage composite sign approximation.
    Value s0 = makeEval(xs, kCompositeSignPoly0);
    Value s1 = makeEval(s0, kCompositeSignPoly1);
    Value step = makeEval(s1, kCompositeSignPoly2);
    // ReLU(x) = x * step(x/B)  (step in [0,1]; B>0 so sign unchanged by scale)
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, x, step);
    return success();
  }
};

struct PolynomialApproximation
    : impl::PolynomialApproximationBase<PolynomialApproximation> {
  using PolynomialApproximationBase::PolynomialApproximationBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    // High priority patterns
    patterns.add<ExpOpTaylorApproximation>(context, /*k=*/7);
    if (useCompositeRelu) {
      patterns.add<ReluViaCompositeSign>(context);
    }

    // Math unary ops
    patterns.add<ConvertUnaryOp<math::AbsFOp>>(context, absf);
    patterns.add<ConvertUnaryOp<math::AcosOp>>(context, acos);
    patterns.add<ConvertUnaryOp<math::AcoshOp>>(context, acosh);
    patterns.add<ConvertUnaryOp<math::AsinOp>>(context, asin);
    patterns.add<ConvertUnaryOp<math::AsinhOp>>(context, asinh);
    patterns.add<ConvertUnaryOp<math::AtanOp>>(context, atan);
    patterns.add<ConvertUnaryOp<math::AtanhOp>>(context, atanh);
    patterns.add<ConvertUnaryOp<math::CbrtOp>>(context, cbrt);
    patterns.add<ConvertUnaryOp<math::CeilOp>>(context, ceil);
    patterns.add<ConvertUnaryOp<math::CosOp>>(context, cos);
    patterns.add<ConvertUnaryOp<math::CoshOp>>(context, cosh);
    patterns.add<ConvertUnaryOp<math::ErfOp>>(context, erf);
    patterns.add<ConvertUnaryOp<math::ErfcOp>>(context, erfc);
    patterns.add<ConvertUnaryOp<math::ExpOp>>(context, exp);
    patterns.add<ConvertUnaryOp<math::Exp2Op>>(context, exp2);
    patterns.add<ConvertUnaryOp<math::ExpM1Op>>(context, expm1);
    patterns.add<ConvertUnaryOp<math::FloorOp>>(context, floor);
    patterns.add<ConvertUnaryOp<math::LogOp>>(
        context, log, kDefaultPositiveRangeLower, kDefaultPositiveRangeUpper);
    patterns.add<ConvertUnaryOp<math::Log10Op>>(
        context, log10, kDefaultPositiveRangeLower, kDefaultPositiveRangeUpper);
    patterns.add<ConvertUnaryOp<math::Log1pOp>>(context, log1p);
    patterns.add<ConvertUnaryOp<math::Log2Op>>(
        context, log2, kDefaultPositiveRangeLower, kDefaultPositiveRangeUpper);
    patterns.add<ConvertUnaryOp<math::RoundOp>>(context, round);
    patterns.add<ConvertUnaryOp<math::RsqrtOp>>(
        context, rsqrt, kDefaultPositiveRangeLower, kDefaultPositiveRangeUpper);
    patterns.add<ConvertUnaryOp<math::SinOp>>(context, sin);
    patterns.add<ConvertUnaryOp<math::SinhOp>>(context, sinh);
    patterns.add<ConvertUnaryOp<math::SqrtOp>>(context, sqrt,
                                               kDefaultNonNegativeRangeLower,
                                               kDefaultNonNegativeRangeUpper);
    patterns.add<ConvertUnaryOp<math::TanOp>>(context, tan);
    patterns.add<ConvertUnaryOp<math::TanhOp>>(context, tanh);
    patterns.add<ConvertUnaryOp<math::TruncOp>>(context, trunc);
    patterns.add<ConvertUnaryOp<math_ext::SignOp>>(context, sign);
    patterns.add<ConvertUnaryOp<math_ext::SigmoidOp>>(context, sigmoid);

    // TODO(#1514): Restore with alternative roundeven
    // patterns.add<ConvertUnaryOp<math::RoundEvenOp>>(context, _roundeven);

    // Unsupported math dialect unary ops:
    // math::AbsIOp
    // math::CtlzOp
    // math::CtpopOp
    // math::CttzOp
    // math::IsfiniteOp
    // math::IsinfOp
    // math::IsnanOp
    // math::IsnormalOp

    // Math binary ops (when one argument is statically constant)
    patterns.add<ConvertBinaryConstOp<arith::MaxNumFOp>>(context, maxnumf);
    patterns.add<ConvertBinaryConstOp<arith::MaximumFOp>>(context, maxf);
    patterns.add<ConvertBinaryConstOp<arith::MinNumFOp>>(context, minf);
    patterns.add<ConvertBinaryConstOp<arith::MinimumFOp>>(context, minnumf);
    patterns.add<ConvertBinaryConstOp<math::Atan2Op>>(context, atan2);
    patterns.add<ConvertBinaryConstOp<math::CopySignOp>>(context, copysign);
    patterns.add<ConvertBinaryConstOp<math::FPowIOp>>(context, fpowi);
    patterns.add<ConvertBinaryConstOp<math::PowFOp>>(context, powf);

    // Math ternary ops
    // patterns.add<ConvertUnaryOp<math::FmaOp>>(context, fma);

    // TODO (#1221): Investigate whether folding (default: on) can be skipped
    // here.
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
