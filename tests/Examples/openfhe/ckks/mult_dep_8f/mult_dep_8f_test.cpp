#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/mult_dep_8f/mult_dep_8f_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(MultDep8FTest, RunTest) {
  auto cryptoContext = mult_dep__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = mult_dep__configure_crypto_context(cryptoContext, secretKey);

  // x = 0.5, expected: x^8 = 0.00390625
  float x = 0.5;
  float expected = 0.00390625;

  auto arg0Encrypted = mult_dep__encrypt__arg0(cryptoContext, x, publicKey);
  auto outputEncrypted = mult_dep(cryptoContext, arg0Encrypted);
  auto actual =
      mult_dep__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-3);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
