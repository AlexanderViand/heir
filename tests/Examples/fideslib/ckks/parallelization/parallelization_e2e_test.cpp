#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/fideslib/ckks/parallelization/parallelization_fideslib_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(ParallelizationE2ETest, RunWithFideslibBackend) {
  auto cryptoContext = rotations__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      rotations__configure_crypto_context(cryptoContext, secretKey);

  // FIDESlib GPU path expects context loaded once keys are ready.
  cryptoContext->LoadContext(publicKey);

  std::vector<int16_t> input(1024, 0);
  input[0] = 1;
  input[1] = 2;
  input[2] = 3;
  input[3] = 4;
  input[4] = 5;
  input[5] = 6;
  input[6] = 7;
  input[7] = 8;

  auto encrypted = rotations__encrypt__arg0(cryptoContext, input, publicKey);
  auto outputEncrypted = rotations(cryptoContext, encrypted);
  auto outputDecrypted =
      rotations__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  ASSERT_GE(outputDecrypted.size(), 8u);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
