#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "gtest/gtest.h"                                     // from @googletest
#include "src/pke/include/encoding/plaintext-fwd.h"          // from @openfhe
#include "src/pke/include/schemerns/rns-cryptoparameters.h"  // from @openfhe
#include "tests/Examples/orion/mlp/mlp_openfhe_flexible_auto_ext_debug_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(MlpOpenFheFlexibleAutoExtDebugTest, RunTest) {
  auto cryptoContext = mlp__generate_crypto_context();
  auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(
      cryptoContext->GetCryptoParameters());
  ASSERT_TRUE(cryptoParams);
  auto numModuli = cryptoParams->GetElementParams()->GetParams().size();
  std::cout << "OpenFHE scaling factors:\n";
  for (size_t i = 0; i < numModuli; ++i) {
    std::cout << "  real[" << i
              << "]=" << std::log2(cryptoParams->GetScalingFactorReal(i));
    if (i + 1 < numModuli) {
      std::cout << " big[" << i
                << "]=" << std::log2(cryptoParams->GetScalingFactorRealBig(i));
    }
    std::cout << "\n";
  }
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
  auto resultEncrypted = mlp(cryptoContext, secretKey, encryptedInput, arg0,
                             arg1, arg2, arg3, arg4, arg5);
  auto duration = std::chrono::steady_clock::now() - start;
  std::cout
      << "MLP flexible-auto-ext debug call took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms\n";

  PlaintextT decrypted;
  cryptoContext->Decrypt(secretKey, resultEncrypted, &decrypted);
  decrypted->SetLength(kSlots);
  auto result = decrypted->GetRealPackedValue();

  ASSERT_EQ(result.size(), static_cast<size_t>(kSlots));
  ASSERT_FALSE(result.empty());
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
