#include <cstdint>
#include <vector>

#include "heir_output.h"
#include "src/pke/include/openfhe.h"  // from @openfhe

int main(int argc, char *argv[]) {
  // Took generated parameters from heir_output.cpp
  CCParams<CryptoContextBGVRNS> parameters;
  // TODO(#661): replace this setup with a HEIR-generated helper function
  parameters.SetMultiplicativeDepth(1);
  parameters.SetPlaintextModulus(4295294977);
  CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
  cryptoContext->Enable(PKE);
  cryptoContext->Enable(KEYSWITCH);
  cryptoContext->Enable(LEVELEDSHE);

  KeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();
  cryptoContext->EvalMultKeyGen(keyPair.secretKey);
  cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {1, 2, 4, 8, 16, 31});

  int32_t n = cryptoContext->GetCryptoParameters()
                  ->GetElementParams()
                  ->GetCyclotomicOrder() /
              2;
  int16_t input = 12;
  // bool cond = true
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

  std::cout << "Expected: " << expected << "\n";
  std::cout << "Actual: " << actual << "\n";

  return 0;
}
