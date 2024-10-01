
#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextBGVRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT sum_tensor(CryptoContextT v0, CiphertextT v1);
CiphertextT sum_tensor__encrypt__arg0(CryptoContextT v17,
                                      std::vector<int16_t> v18, PublicKeyT v19);
int16_t sum_tensor__decrypt__result0(CryptoContextT v22, CiphertextT v23,
                                     PrivateKeyT v24);
CryptoContextT sum_tensor__generate_crypto_context();
CryptoContextT sum_tensor__configure_crypto_context(CryptoContextT v29,
                                                    PrivateKeyT v30);
