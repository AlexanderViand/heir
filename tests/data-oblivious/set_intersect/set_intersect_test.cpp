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
  auto cryptoContext = set_intersect__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      set_intersect__configure_crypto_context(cryptoContext, secretKey);

  int32_t n = cryptoContext->GetCryptoParameters()
                  ->GetElementParams()
                  ->GetCyclotomicOrder() /
              2;
  std::vector<int16_t> input1;
  std::vector<int16_t> input2;
  // TODO(#645): support cyclic repetition in add-client-interface
  // I want to do this, but MakePackedPlaintext does not repeat the values.
  // It zero pads, and rotating the zero-padded values will not achieve the
  // rotate-and-reduce trick required for set_intersect
  //
  // = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
  //    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
  //    23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  input1.reserve(n);
  input2.reserve(n);

  for (int i = 0; i < n; ++i) {
    input1.push_back((i % 32) + 1);
    input2.push_back((i % 32) + 1);
  }
  std::vector<int16_t> expected = {1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1};

  auto input1Encrypted =
      set_intersect__encrypt__arg0(cryptoContext, input1, keyPair.publicKey);
  auto input2Encrypted =
      set_intersect__encrypt__arg1(cryptoContext, input2, keyPair.publicKey);
  auto outputEncrypted =
      set_intersect(cryptoContext, input1Encrypted, input2Encrypted);
  auto actual = set_intersect__decrypt__result0(cryptoContext, outputEncrypted,
                                                keyPair.secretKey);

  EXPECT_EQ(expected, actual);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
