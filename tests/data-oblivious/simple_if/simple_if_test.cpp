#include <cstdint>
#include <vector>

#include "gtest/gtest.h"                               // from @googletest
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/constants.h"                 // from @openfhe
#include "src/pke/include/cryptocontext-fwd.h"         // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"         // from @openfhe
#include "src/pke/include/key/keypair.h"               // from @openfhe
#include "src/pke/include/openfhe.h"                   // from @openfhe
#include "src/pke/include/scheme/bgvrns/gen-cryptocontext-bgvrns-params.h"  // from @openfhe
#include "src/pke/include/scheme/bgvrns/gen-cryptocontext-bgvrns.h"  // from @openfhe

namespace mlir {
namespace heir {
namespace openfhe {

TEST(BinopsTest, TestInput1) {
  auto cryptoContext = simple_if__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = simple_if__configure_crypto_context(cryptoContext, secretKey);

  int32_t n = cryptoContext->GetCryptoParameters()
                  ->GetElementParams()
                  ->GetCyclotomicOrder() /
              2;
  int16_t input = 12;
  int8_t cond = 1;

  int16_t expected = 24;

  auto inputEncrypted =
      simple_if__encrypt__arg0(cryptoContext, input, keyPair.publicKey);
  auto condEncrypted =
      simple_if__encrypt__arg1(cryptoContext, cond, keyPair.publicKey);
  auto outputEncrypted =
      simple_if(cryptoContext, inputEncrypted, condEncrypted);
  auto actual = simple_if__decrypt__result0(cryptoContext, outputEncrypted,
                                            keyPair.secretKey);

  EXPECT_EQ(expected, actual);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
