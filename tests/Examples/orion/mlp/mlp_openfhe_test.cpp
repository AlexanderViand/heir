#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"                             // from @googletest
#include "src/pke/include/encoding/plaintext-fwd.h"  // from @openfhe
#include "tests/Examples/orion/mlp/mlp_openfhe_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(MlpOpenFheTest, RunTest) {
  auto cryptoContext = mlp__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = mlp__configure_crypto_context(cryptoContext, secretKey);

  constexpr int kSlots = 4096;
  std::vector<double> input(kSlots, 1.0);
  std::vector<double> arg0(128 * kSlots, 0.0);
  std::vector<double> arg1(kSlots, 0.0);
  std::vector<double> arg2(128 * kSlots, 0.0);
  std::vector<double> arg3(kSlots, 0.0);
  std::vector<double> arg4(137 * kSlots, 0.0);
  std::vector<double> arg5(kSlots, 0.25);

  for (int i = 0; i < kSlots; ++i) {
    arg0[i] = 1.0 / 64.0;
    arg2[i] = 1.0 / 32.0;
    arg4[i] = 2.0;
  }

  auto plaintext = cryptoContext->MakeCKKSPackedPlaintext(input);
  plaintext->SetLength(input.size());
  auto encryptedInput = cryptoContext->Encrypt(publicKey, plaintext);
  auto start = std::chrono::steady_clock::now();
  auto resultEncrypted =
      mlp(cryptoContext, encryptedInput, arg0, arg1, arg2, arg3, arg4, arg5);
  auto duration = std::chrono::steady_clock::now() - start;
  std::cout
      << "MLP call took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms\n";

  PlaintextT decrypted;
  cryptoContext->Decrypt(secretKey, resultEncrypted, &decrypted);
  decrypted->SetLength(kSlots);
  auto result = decrypted->GetRealPackedValue();

  ASSERT_EQ(result.size(), static_cast<size_t>(kSlots));
  ASSERT_FALSE(result.empty());

  constexpr double kExpected = 0.375;
  constexpr double kTolerance = 0.05;
  for (int i = 0; i < kSlots; ++i) {
    ASSERT_TRUE(std::isfinite(result[i]));
    EXPECT_NEAR(result[i], kExpected, kTolerance);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
