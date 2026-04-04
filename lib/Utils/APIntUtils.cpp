#include "lib/Utils/APIntUtils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/APInt.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

/// Cloned after upstream removal in
/// https://github.com/llvm/llvm-project/pull/87644
///
/// Computes the multiplicative inverse of this APInt for a given modulo. The
/// iterative extended Euclidean algorithm is used to solve for this value,
/// however we simplify it to speed up calculating only the inverse, and take
/// advantage of div+rem calculations. We also use some tricks to avoid copying
/// (potentially large) APInts around.
/// WARNING: a value of '0' may be returned,
///          signifying that no multiplicative inverse exists!
APInt multiplicativeInverse(const APInt& x, const APInt& modulo) {
  assert(x.ult(modulo) && "This APInt must be smaller than the modulo");
  // Using the properties listed at the following web page (accessed 06/21/08):
  //   http://www.numbertheory.org/php/euclid.html
  // (especially the properties numbered 3, 4 and 9) it can be proved that
  // BitWidth bits suffice for all the computations in the algorithm implemented
  // below. More precisely, this number of bits suffice if the multiplicative
  // inverse exists, but may not suffice for the general extended Euclidean
  // algorithm.

  auto BitWidth = x.getBitWidth();
  APInt r[2] = {modulo, x};
  APInt t[2] = {APInt(BitWidth, 0), APInt(BitWidth, 1)};
  APInt q(BitWidth, 0);

  unsigned i;
  for (i = 0; r[i ^ 1] != 0; i ^= 1) {
    // An overview of the math without the confusing bit-flipping:
    // q = r[i-2] / r[i-1]
    // r[i] = r[i-2] % r[i-1]
    // t[i] = t[i-2] - t[i-1] * q
    x.udivrem(r[i], r[i ^ 1], q, r[i]);
    t[i] -= t[i ^ 1] * q;
  }

  // If this APInt and the modulo are not coprime, there is no multiplicative
  // inverse, so return 0. We check this by looking at the next-to-last
  // remainder, which is the gcd(*this,modulo) as calculated by the Euclidean
  // algorithm.
  if (r[i] != 1) return APInt(BitWidth, 0);

  // The next-to-last t is the multiplicative inverse.  However, we are
  // interested in a positive inverse. Calculate a positive one from a negative
  // one if necessary. A simple addition of the modulo suffices because
  // abs(t[i]) is known to be less than *this/2 (see the link above).
  if (t[i].isNegative()) t[i] += modulo;

  return std::move(t[i]);
}

APInt modularExponentiation(const APInt& base, const APInt& exponent,
                            const APInt& modulus) {
  APInt res(modulus.getBitWidth(), 1);
  APInt b = base.urem(modulus);
  APInt e = exponent;

  while (e.ugt(0)) {
    if (e[0]) {
      res = modularMultiplication(res, b, modulus);
    }
    b = modularMultiplication(b, b, modulus);
    e = e.lshr(1);
  }
  return res;
}

bool isPrime(const APInt& n) {
  if (n.ult(2)) return false;
  if (n.ult(4)) return true;
  if (!n[0]) return false;

  // Miller-Rabin primality test
  APInt d = n - 1;
  unsigned s = d.countTrailingZeros();
  d = d.lshr(s);

  // Bases to test.
  // Using the first 12 prime bases makes the test deterministic for all
  // 64-bit integers. See https://oeis.org/A014233.
  // We use 20 bases to further reduce the probability of error for
  // arbitrary-precision integers.
  std::vector<uint64_t> bases = {2,  3,  5,  7,  11, 13, 17, 19, 23, 29,
                                 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
  for (uint64_t a : bases) {
    if (n.ule(a)) break;
    APInt x = modularExponentiation(APInt(n.getBitWidth(), a), d, n);
    if (x.isOne() || x == n - 1) continue;
    bool composite = true;
    for (unsigned r = 1; r < s; ++r) {
      x = modularMultiplication(x, x, n);
      if (x == n - 1) {
        composite = false;
        break;
      }
    }
    if (composite) return false;
  }
  return true;
}

std::vector<APInt> factorize(APInt n) {
  std::vector<APInt> factors;
  if (n.ult(2)) return factors;

  unsigned width = n.getBitWidth();
  APInt d(width, 2);
  while (true) {
    APInt wide_d = d.zext(width * 2);
    if ((wide_d * wide_d).ugt(n.zext(width * 2))) break;

    if (n.urem(d).isZero()) {
      factors.push_back(d);
      while (n.urem(d).isZero()) {
        n = n.udiv(d);
      }
    }
    ++d;
  }
  if (n.ugt(1)) {
    factors.push_back(n);
  }
  return factors;
}

APInt canonicalizeUnsignedAPInt(const APInt& value, unsigned minWidth) {
  unsigned activeBits = value.getActiveBits();
  unsigned width = std::max(minWidth, activeBits == 0 ? 1U : activeBits + 1);
  return value.zextOrTrunc(width);
}

IntegerAttr getSignlessIntegerAttr(MLIRContext* context, const APInt& value,
                                   unsigned minWidth) {
  APInt canonical = canonicalizeUnsignedAPInt(value, minWidth);
  auto type = IntegerType::get(context, canonical.getBitWidth());
  return IntegerAttr::get(type, canonical);
}

APInt getNominalPowerOfTwoScaleFromLog2(uint64_t logScale, unsigned minWidth) {
  return canonicalizeUnsignedAPInt(APInt::getOneBitSet(logScale + 1, logScale),
                                   minWidth);
}

APInt multiplyUnsignedAPIntExact(const APInt& lhs, const APInt& rhs,
                                 unsigned minWidth) {
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  APInt wideLhs = lhs.zext(width);
  APInt wideRhs = rhs.zext(width);
  return canonicalizeUnsignedAPInt(
      llvm::APIntOps::muluExtended(wideLhs, wideRhs), minWidth);
}

FailureOr<APInt> divideUnsignedAPIntExact(const APInt& dividend,
                                          const APInt& divisor,
                                          unsigned minWidth) {
  if (divisor.isZero()) return failure();

  unsigned width = std::max(dividend.getBitWidth(), divisor.getBitWidth());
  APInt wideDividend = dividend.zext(width);
  APInt wideDivisor = divisor.zext(width);
  if (!wideDividend.urem(wideDivisor).isZero()) return failure();
  return canonicalizeUnsignedAPInt(wideDividend.udiv(wideDivisor), minWidth);
}

FailureOr<APInt> divideUnsignedAPIntNearest(const APInt& dividend,
                                            const APInt& divisor,
                                            unsigned minWidth) {
  if (divisor.isZero()) return failure();

  unsigned width = std::max(dividend.getBitWidth(), divisor.getBitWidth());
  APInt wideDividend = dividend.zext(width);
  APInt wideDivisor = divisor.zext(width);
  APInt quotient = wideDividend.udiv(wideDivisor);
  APInt remainder = wideDividend.urem(wideDivisor);
  APInt doubledRemainder = remainder.zext(width + 1) * APInt(width + 1, 2);
  APInt compareDivisor = wideDivisor.zext(width + 1);
  if (doubledRemainder.uge(compareDivisor)) {
    quotient += APInt(width, 1);
  }
  return canonicalizeUnsignedAPInt(quotient, minWidth);
}

FailureOr<APInt> solveUnsignedPostRescaleScaleDelta(const APInt& inputScale,
                                                    const APInt& targetScale,
                                                    const APInt& dividedModulus,
                                                    unsigned minWidth) {
  if (inputScale.isZero() || targetScale.isZero() || dividedModulus.isZero()) {
    return failure();
  }

  APInt two = APInt(64, 2);
  APInt one = APInt(64, 1);
  APInt targetTimesTwo = multiplyUnsignedAPIntExact(targetScale, two);
  APInt oneAtWidth = one.zext(targetTimesTwo.getBitWidth());
  APInt lowerFactor = targetTimesTwo - oneAtWidth;
  APInt upperFactor = targetTimesTwo + oneAtWidth;

  APInt lowerNumerator =
      multiplyUnsignedAPIntExact(dividedModulus, lowerFactor);
  APInt upperNumerator =
      multiplyUnsignedAPIntExact(dividedModulus, upperFactor);
  upperNumerator -= APInt(upperNumerator.getBitWidth(), 1);

  APInt denominator = multiplyUnsignedAPIntExact(inputScale, two);
  unsigned boundWidth =
      std::max({lowerNumerator.getBitWidth(), upperNumerator.getBitWidth(),
                denominator.getBitWidth()});
  APInt lowerBound = llvm::APIntOps::RoundingUDiv(
      lowerNumerator.zext(boundWidth), denominator.zext(boundWidth),
      APInt::Rounding::UP);
  APInt upperBound = llvm::APIntOps::RoundingUDiv(
      upperNumerator.zext(boundWidth), denominator.zext(boundWidth),
      APInt::Rounding::DOWN);
  APInt candidate =
      lowerBound.ult(APInt(boundWidth, 1)) ? APInt(boundWidth, 1) : lowerBound;
  if (candidate.ugt(upperBound)) return failure();
  return canonicalizeUnsignedAPInt(candidate, minWidth);
}

FailureOr<APInt> solveUnsignedPostRescaleScaleDeltaChain(
    const APInt& inputScale, const APInt& targetScale,
    ArrayRef<APInt> dividedModuli, unsigned minWidth) {
  if (dividedModuli.empty()) {
    return divideUnsignedAPIntExact(targetScale, inputScale, minWidth);
  }
  if (inputScale.isZero() || targetScale.isZero()) {
    return failure();
  }

  APInt lowerBoundTarget = canonicalizeUnsignedAPInt(targetScale);
  APInt upperBoundTarget = canonicalizeUnsignedAPInt(targetScale);
  APInt two(64, 2);

  for (const APInt& dividedModulus : llvm::reverse(dividedModuli)) {
    if (dividedModulus.isZero() || lowerBoundTarget.isZero()) {
      return failure();
    }

    APInt lowerFactor = multiplyUnsignedAPIntExact(lowerBoundTarget, two);
    lowerFactor -= APInt(lowerFactor.getBitWidth(), 1);
    APInt upperFactor = multiplyUnsignedAPIntExact(upperBoundTarget, two);
    upperFactor += APInt(upperFactor.getBitWidth(), 1);

    APInt lowerNumerator =
        multiplyUnsignedAPIntExact(dividedModulus, lowerFactor);
    APInt upperNumerator =
        multiplyUnsignedAPIntExact(dividedModulus, upperFactor);
    upperNumerator -= APInt(upperNumerator.getBitWidth(), 1);

    unsigned boundWidth =
        std::max(lowerNumerator.getBitWidth(), upperNumerator.getBitWidth());
    APInt denominator = two.zext(boundWidth);
    lowerBoundTarget = llvm::APIntOps::RoundingUDiv(
        lowerNumerator.zext(boundWidth), denominator, APInt::Rounding::UP);
    upperBoundTarget = llvm::APIntOps::RoundingUDiv(
        upperNumerator.zext(boundWidth), denominator, APInt::Rounding::DOWN);
  }

  unsigned boundWidth =
      std::max({lowerBoundTarget.getBitWidth(), upperBoundTarget.getBitWidth(),
                inputScale.getBitWidth()});
  APInt lowerDelta = llvm::APIntOps::RoundingUDiv(
      lowerBoundTarget.zext(boundWidth), inputScale.zext(boundWidth),
      APInt::Rounding::UP);
  APInt upperDelta = llvm::APIntOps::RoundingUDiv(
      upperBoundTarget.zext(boundWidth), inputScale.zext(boundWidth),
      APInt::Rounding::DOWN);
  APInt candidate =
      lowerDelta.ult(APInt(boundWidth, 1)) ? APInt(boundWidth, 1) : lowerDelta;
  if (candidate.ugt(upperDelta)) return failure();
  return canonicalizeUnsignedAPInt(candidate, minWidth);
}

}  // namespace heir
}  // namespace mlir
