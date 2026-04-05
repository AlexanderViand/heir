#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"                             // from @googletest
#include "src/pke/include/encoding/plaintext-fwd.h"  // from @openfhe
#include "tests/Examples/orion/linear_transform/linear_transform_openfhe_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(LinearTransformOpenFheTest, RunTest) {
  auto cryptoContext = linear_transform__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      linear_transform__configure_crypto_context(cryptoContext, secretKey);

  constexpr int kSlots = 4096;
  std::vector<double> input(kSlots, 1.0);
  std::vector<double> matrix(2 * kSlots);
  for (int i = 0; i < kSlots; ++i) {
    matrix[i] = 1.0;
    matrix[kSlots + i] = 2.0;
  }

  std::vector<double> expected(kSlots, 3.0);

  auto plaintext = cryptoContext->MakeCKKSPackedPlaintext(input);
  plaintext->SetLength(input.size());
  auto encryptedInput = cryptoContext->Encrypt(publicKey, plaintext);
  auto start = std::chrono::steady_clock::now();
  auto resultEncrypted =
      linear_transform(cryptoContext, encryptedInput, matrix);
  auto duration = std::chrono::steady_clock::now() - start;
  std::cout
      << "linear_transform call took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms\n";
  PlaintextT decrypted;
  cryptoContext->Decrypt(secretKey, resultEncrypted, &decrypted);
  decrypted->SetLength(expected.size());
  auto result = decrypted->GetRealPackedValue();

  ASSERT_EQ(result.size(), expected.size());
  constexpr double kTolerance = 0.05;
  for (int i = 0; i < kSlots; ++i) {
    EXPECT_NEAR(result[i], expected[i], kTolerance);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
