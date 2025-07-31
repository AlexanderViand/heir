#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/composite_scaling/composite_scaling_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(CompositeScalingTest, RunTest) {
  auto cryptoContext = dot_product__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      dot_product__configure_crypto_context(cryptoContext, secretKey);

  std::vector<float> arg0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> arg1 = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

  auto ciphertext0 = dot_product__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto ciphertext1 = dot_product__encrypt__arg1(cryptoContext, arg1, publicKey);
  auto ciphertext_result = dot_product(cryptoContext, ciphertext0, ciphertext1);
  auto actual = dot_product__decrypt__result0(cryptoContext, ciphertext_result,
                                              secretKey);

  // Calculate expected: (1*2 + 2*2 + 3*2 + 4*2 + 5*2 + 6*2 + 7*2 + 8*2) + 0.1
  // = 72.1
  float expected = 0.1f;  // Initial value from MLIR
  for (int i = 0; i < 8; i++) {
    expected += arg0[i] * arg1[i];
  }

  EXPECT_NEAR(expected, actual, 0.1);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
