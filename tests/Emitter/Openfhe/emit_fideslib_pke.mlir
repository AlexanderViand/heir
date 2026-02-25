// RUN: heir-translate %s --emit-fideslib-pke | FileCheck %s

!params = !openfhe.cc_params
!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!sk = !openfhe.private_key
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext

// CHECK: #include "fideslib/fideslib.hpp"
// CHECK: using namespace fideslib;
// CHECK: CCParamsT params;
// CHECK: params.SetMultiplicativeDepth(3);
// CHECK: params.SetScalingModSize(50);
// CHECK: params.SetBatchSize(8);
// CHECK: params.SetSecurityLevel(fideslib::HEStd_NotSet);
// CHECK: params.SetKeySwitchTechnique(fideslib::HYBRID);
// CHECK: CryptoContextT cc = GenCryptoContext(params);
// CHECK: cc->Enable(PKE);
// CHECK: cc->Enable(KEYSWITCH);
// CHECK: cc->Enable(LEVELEDSHE);
// CHECK: MakeCKKSPackedPlaintext(
// CHECK: Decrypt(
// CHECK: GetCKKSPackedValue()[0].real();
module attributes {scheme.ckks} {
  func.func @smoke_generate_crypto_context() -> !cc {
    %params = openfhe.gen_params {mulDepth = 3 : i64, plainMod = 0 : i64, scalingModSize = 50, batchSize = 8, insecure = true} : () -> !params
    %cc = openfhe.gen_context %params {supportFHE = false} : (!params) -> !cc
    return %cc : !cc
  }

  func.func @smoke_encrypt_decrypt(%cc: !cc, %pk: !pk, %sk: !sk, %input: tensor<4xi64>) -> f64 {
    %pt = openfhe.make_packed_plaintext %cc, %input : (!cc, tensor<4xi64>) -> !pt
    %ct = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
    %pt_out = openfhe.decrypt %cc, %ct, %sk : (!cc, !ct, !sk) -> !pt
    %out = openfhe.decode_ckks %pt_out : !pt -> f64
    return %out : f64
  }
}
