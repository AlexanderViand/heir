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

  std::cout << "Expected: " << expected << "\n";
  std::cout << "Actual: " << actual << "\n";

  return 0;
}
