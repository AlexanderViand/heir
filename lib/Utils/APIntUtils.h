#ifndef LIB_UTILS_APINTUTILS_H_
#define LIB_UTILS_APINTUTILS_H_

#include <vector>

#include "llvm/include/llvm/ADT/APInt.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

APInt multiplicativeInverse(const APInt& x, const APInt& modulo);

inline APInt modularMultiplication(const APInt& a, const APInt& b,
                                   const APInt& modulus) {
  unsigned width = modulus.getBitWidth();
  APInt wide_a = a.zext(width * 2);
  APInt wide_b = b.zext(width * 2);
  APInt wide_m = modulus.zext(width * 2);
  return (wide_a * wide_b).urem(wide_m).trunc(width);
}

APInt modularExponentiation(const APInt& base, const APInt& exponent,
                            const APInt& modulus);

bool isPrime(const APInt& n);

/// Returns the prime factors of n, without multiplicity. If n < 2, returns an
/// empty vector.
std::vector<APInt> factorize(APInt n);

APInt canonicalizeUnsignedAPInt(const APInt& value, unsigned minWidth = 64);

IntegerAttr getSignlessIntegerAttr(MLIRContext* context, const APInt& value,
                                   unsigned minWidth = 64);

APInt getNominalPowerOfTwoScaleFromLog2(uint64_t logScale,
                                        unsigned minWidth = 64);

APInt multiplyUnsignedAPIntExact(const APInt& lhs, const APInt& rhs,
                                 unsigned minWidth = 64);

FailureOr<APInt> divideUnsignedAPIntExact(const APInt& dividend,
                                          const APInt& divisor,
                                          unsigned minWidth = 64);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_APINTUTILS_H_
