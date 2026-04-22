#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"                             // from @googletest
#include "src/pke/include/encoding/plaintext-fwd.h"  // from @openfhe
#include "tests/Examples/orion/chebyshev/chebyshev_openfhe_fixed_auto_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(ChebyshevOpenFheFixedAutoTest, RunTest) {
  auto cryptoContext = chebyshev__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = chebyshev__configure_crypto_context(cryptoContext, secretKey);

  constexpr int kSlots = 32768;
  std::vector<double> input(kSlots, 0.5);
  std::vector<double> expected(kSlots, 0.125);

  auto plaintext = cryptoContext->MakeCKKSPackedPlaintext(input);
  plaintext->SetLength(input.size());
  auto encryptedInput = cryptoContext->Encrypt(publicKey, plaintext);
  auto start = std::chrono::steady_clock::now();
  auto resultEncrypted = chebyshev(cryptoContext, encryptedInput);
  auto duration = std::chrono::steady_clock::now() - start;
  std::cout
      << "chebyshev fixed-auto call took: "
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
