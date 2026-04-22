
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/APIntUtils.h"
#include "llvm/include/llvm/ADT/APInt.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using ::llvm::APInt;

TEST(APIntUtilsTest, ModularExponentiation) {
  APInt base(64, 2);
  APInt exp(64, 10);
  APInt mod(64, 1000);
  EXPECT_EQ(modularExponentiation(base, exp, mod), 24);  // 2^10 = 1024

  EXPECT_EQ(modularExponentiation(APInt(64, 3), APInt(64, 4), APInt(64, 100)),
            81);
}

TEST(APIntUtilsTest, IsPrime) {
  EXPECT_TRUE(isPrime(APInt(64, 2)));
  EXPECT_TRUE(isPrime(APInt(64, 3)));
  EXPECT_TRUE(isPrime(APInt(64, 5)));
  EXPECT_TRUE(isPrime(APInt(64, 7)));
  EXPECT_TRUE(isPrime(APInt(64, 11)));
  EXPECT_TRUE(isPrime(APInt(64, 13)));
  EXPECT_TRUE(isPrime(APInt(64, 17)));
  EXPECT_TRUE(isPrime(APInt(64, 19)));
  EXPECT_TRUE(isPrime(APInt(64, 23)));

  EXPECT_FALSE(isPrime(APInt(64, 1)));
  EXPECT_FALSE(isPrime(APInt(64, 4)));
  EXPECT_FALSE(isPrime(APInt(64, 6)));
  EXPECT_FALSE(isPrime(APInt(64, 8)));
  EXPECT_FALSE(isPrime(APInt(64, 9)));
  EXPECT_FALSE(isPrime(APInt(64, 15)));
  EXPECT_FALSE(isPrime(APInt(64, 21)));
  EXPECT_FALSE(isPrime(APInt(64, 25)));

  // Larger primes
  EXPECT_TRUE(isPrime(APInt(64, 65537)));
  EXPECT_TRUE(isPrime(APInt(64, 114689)));
}

TEST(APIntUtilsTest, Factorize12) {
  auto factors = factorize(APInt(64, 12));
  ASSERT_EQ(factors.size(), 2);
  EXPECT_EQ(factors[0], 2);
  EXPECT_EQ(factors[1], 3);
}

TEST(APIntUtilsTest, Factorize65536) {
  auto factors = factorize(APInt(64, 65536));  // 2^16
  ASSERT_EQ(factors.size(), 1);
  EXPECT_EQ(factors[0], 2);
}

TEST(APIntUtilsTest, Factorize114688) {
  // 114689 - 1 = 114688 = 2^14 * 7
  auto factors = factorize(APInt(64, 114688));
  ASSERT_EQ(factors.size(), 2);
  EXPECT_EQ(factors[0], 2);
  EXPECT_EQ(factors[1], 7);
}

TEST(APIntUtilsTest, FactorizePrime) {
  auto factors = factorize(APInt(64, 53));
  ASSERT_EQ(factors.size(), 1);
  EXPECT_EQ(factors[0], 53);
}

TEST(APIntUtilsTest, DivideUnsignedAPIntNearest) {
  auto result = divideUnsignedAPIntNearest(APInt(64, 10), APInt(64, 3));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(*result, APInt(64, 3));

  result = divideUnsignedAPIntNearest(APInt(64, 11), APInt(64, 3));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(*result, APInt(64, 4));
}

TEST(APIntUtilsTest, SolveUnsignedPostRescaleScaleDelta) {
  auto delta = solveUnsignedPostRescaleScaleDelta(APInt(64, 8), APInt(64, 3),
                                                  APInt(64, 11));
  ASSERT_TRUE(succeeded(delta));
  EXPECT_EQ(*delta, APInt(64, 4));

  auto impossible = solveUnsignedPostRescaleScaleDelta(
      APInt(64, 8), APInt(64, 1), APInt(64, 3));
  EXPECT_TRUE(failed(impossible));
}

TEST(APIntUtilsTest, SolveUnsignedPostRescaleScaleDeltaChain) {
  SmallVector<APInt> moduli = {APInt(64, 11), APInt(64, 5)};
  auto delta = solveUnsignedPostRescaleScaleDeltaChain(APInt(64, 8),
                                                       APInt(64, 1), moduli);
  ASSERT_TRUE(succeeded(delta));
  EXPECT_EQ(*delta, APInt(64, 4));

  auto impossible = solveUnsignedPostRescaleScaleDeltaChain(
      APInt(64, 8), APInt(64, 0), moduli);
  EXPECT_TRUE(failed(impossible));
}

}  // namespace
}  // namespace heir
}  // namespace mlir
