#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/simple_sumf/simple_sumf_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(SimpleSumfTest, RunTest) {
  auto cryptoContext = simple_sum__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      simple_sum__configure_crypto_context(cryptoContext, secretKey);

  // Input: 32 values from 0.1 to 3.2
  std::vector<float> arg0(32);
  float expected = 0.0;
  for (int i = 0; i < 32; ++i) {
    arg0[i] = 0.1 * (i + 1);
    expected += arg0[i];
  }

  auto arg0Encrypted =
      simple_sum__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto outputEncrypted = simple_sum(cryptoContext, arg0Encrypted);
  auto actual =
      simple_sum__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-1);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
