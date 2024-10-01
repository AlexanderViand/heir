
#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextBGVRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT sum_tensor(CryptoContextT v0, CiphertextT v1) {
  std::vector<int16_t> v2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  const auto& v3 = v0->EvalRotate(v1, 16);
  const auto& v4 = v0->EvalAdd(v1, v3);
  const auto& v5 = v0->EvalRotate(v4, 8);
  const auto& v6 = v0->EvalAdd(v4, v5);
  const auto& v7 = v0->EvalRotate(v6, 4);
  const auto& v8 = v0->EvalAdd(v6, v7);
  const auto& v9 = v0->EvalRotate(v8, 2);
  const auto& v10 = v0->EvalAdd(v8, v9);
  const auto& v11 = v0->EvalRotate(v10, 1);
  const auto& v12 = v0->EvalAdd(v10, v11);
  std::vector<int64_t> v2_cast(std::begin(v2), std::end(v2));
  const auto& v13 = v0->MakePackedPlaintext(v2_cast);
  const auto& v14 = v0->EvalMult(v12, v13);
  const auto& v15 = v0->EvalRotate(v14, 31);
  const auto& v16 = v15;
  return v16;
}
CiphertextT sum_tensor__encrypt__arg0(CryptoContextT v17,
                                      std::vector<int16_t> v18,
                                      PublicKeyT v19) {
  std::vector<int64_t> v18_cast(std::begin(v18), std::end(v18));
  const auto& v20 = v17->MakePackedPlaintext(v18_cast);
  const auto& v21 = v17->Encrypt(v19, v20);
  return v21;
}
int16_t sum_tensor__decrypt__result0(CryptoContextT v22, CiphertextT v23,
                                     PrivateKeyT v24) {
  PlaintextT v25;
  v22->Decrypt(v24, v23, &v25);
  int16_t v26 = v25->GetPackedValue()[0];
  return v26;
}
CryptoContextT sum_tensor__generate_crypto_context() {
  CCParamsT v27;
  v27.SetMultiplicativeDepth(1);
  v27.SetPlaintextModulus(4295294977);
  CryptoContextT v28 = GenCryptoContext(v27);
  v28->Enable(PKE);
  v28->Enable(KEYSWITCH);
  v28->Enable(LEVELEDSHE);
  return v28;
}
CryptoContextT sum_tensor__configure_crypto_context(CryptoContextT v29,
                                                    PrivateKeyT v30) {
  v29->EvalMultKeyGen(v30);
  v29->EvalRotateKeyGen(v30, {1, 2, 4, 8, 16, 31});
  return v29;
}
