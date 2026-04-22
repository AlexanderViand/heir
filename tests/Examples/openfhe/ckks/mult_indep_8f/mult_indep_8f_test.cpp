#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/mult_indep_8f/mult_indep_8f_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(MultIndep8FTest, RunTest) {
  auto cryptoContext = mult_indep__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      mult_indep__configure_crypto_context(cryptoContext, secretKey);

  // All inputs = 0.5, expected: 0.5^8 = 0.00390625
  float arg0 = 0.5, arg1 = 0.5, arg2 = 0.5, arg3 = 0.5;
  float arg4 = 0.5, arg5 = 0.5, arg6 = 0.5, arg7 = 0.5;
  float expected = 0.00390625;

  auto ct0 = mult_indep__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto ct1 = mult_indep__encrypt__arg1(cryptoContext, arg1, publicKey);
  auto ct2 = mult_indep__encrypt__arg2(cryptoContext, arg2, publicKey);
  auto ct3 = mult_indep__encrypt__arg3(cryptoContext, arg3, publicKey);
  auto ct4 = mult_indep__encrypt__arg4(cryptoContext, arg4, publicKey);
  auto ct5 = mult_indep__encrypt__arg5(cryptoContext, arg5, publicKey);
  auto ct6 = mult_indep__encrypt__arg6(cryptoContext, arg6, publicKey);
  auto ct7 = mult_indep__encrypt__arg7(cryptoContext, arg7, publicKey);
  auto outputEncrypted =
      mult_indep(cryptoContext, ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7);
  auto actual =
      mult_indep__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_NEAR(expected, actual, 1e-3);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
