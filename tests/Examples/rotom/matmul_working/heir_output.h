
#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using MutableCiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

CiphertextT matmul(CryptoContextT cc, CiphertextT ct, std::vector<float> v0, std::vector<float> v1, std::vector<float> v2, std::vector<float> v3);
CiphertextT matmul__encrypt__arg0(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk);
std::vector<float> matmul__decrypt__result0(CryptoContextT cc, CiphertextT ct, PrivateKeyT sk);
CryptoContextT matmul__generate_crypto_context();
CryptoContextT matmul__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk);
