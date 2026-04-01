// RUN: heir-translate %s --emit-fideslib-pke --split-input-file | FileCheck %s

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext

// CHECK:      #include "fideslib/fideslib.hpp"
// CHECK:      using namespace fideslib;
// CHECK-NOT:  EvalKeyT
// CHECK:      CiphertextT test_basic_ops(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[CT1:[^,]*]],
// CHECK-SAME:    CiphertextT [[CT2:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      const auto& [[v1:.*]] = [[CC]]->EvalAdd([[CT1]], [[CT2]]);
// CHECK-NEXT:      const auto& [[v2:.*]] = [[CC]]->EvalMult([[v1]], [[CT2]]);
// CHECK-NEXT:      const auto& [[v3:.*]] = [[CC]]->Rescale([[v2]]);
// CHECK-NEXT:      const auto& [[v4:.*]] = [[CC]]->EvalRotate([[v3]], 1);
// CHECK-NEXT:      return [[v4]];
module attributes {scheme.ckks} {
  func.func @test_basic_ops(%cc : !cc, %ct1 : !ct, %ct2 : !ct) -> !ct {
    %add = openfhe.add %cc, %ct1, %ct2 : (!cc, !ct, !ct) -> !ct
    %mul = openfhe.mul %cc, %add, %ct2 : (!cc, !ct, !ct) -> !ct
    %rescale = openfhe.mod_reduce %cc, %mul : (!cc, !ct) -> !ct
    %rot = openfhe.rot %cc, %rescale {static_shift = 1 : i64} : (!cc, !ct) -> !ct
    return %rot : !ct
  }
}

// -----

!cc = !openfhe.crypto_context
!params = !openfhe.cc_params
!sk = !openfhe.private_key
!pk = !openfhe.public_key
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext

// GenParams: FIDESlib-specific enum namespacing
// CHECK: test_gen_params_and_decrypt
// CHECK:      CCParamsT [[P:.*]];
// CHECK-NEXT: [[P]].SetMultiplicativeDepth(2);
// CHECK-NEXT: [[P]].SetSecurityLevel(fideslib::HEStd_NotSet);
// CHECK-NEXT: [[P]].SetKeySwitchTechnique(fideslib::HYBRID);
// CHECK-NEXT: [[P]].SetScalingTechnique(fideslib::FIXEDMANUAL);
//
// Decrypt: mutable copy for FIDESlib's non-const ciphertext ref
// CHECK:      PlaintextT [[PTOUT:.*]];
// CHECK-NEXT: auto [[CTVAR:.*]]_mutable = [[CTVAR2:.*]];
// CHECK-NEXT: {{.*}}->Decrypt({{.*}}, [[CTVAR]]_mutable, &[[PTOUT]]);
module attributes {scheme.ckks} {
  func.func @test_gen_params_and_decrypt(%cc : !cc, %sk : !sk, %ct : !ct) -> !pt {
    %params = openfhe.gen_params {mulDepth = 2 : i64, plainMod = 0 : i64, evalAddCount = 0 : i64, keySwitchCount = 0 : i64, ringDim = 0 : i64, batchSize = 0 : i64, firstModSize = 0 : i64, scalingModSize = 0 : i64, digitSize = 0 : i64, numLargeDigits = 0 : i64, maxRelinSkDeg = 0 : i64, insecure = true, scalingTechniqueFixedManual = true} : () -> !params
    %pt = openfhe.decrypt %cc, %ct, %sk : (!cc, !ct, !sk) -> !pt
    return %pt : !pt
  }
}

// -----

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext

// FIDESlib uses MakeCKKSPackedPlaintext with int-to-double conversion
// CHECK: test_make_packed_plaintext
// CHECK:      std::vector<double> [[V:.*]]_d;
// CHECK:      MakeCKKSPackedPlaintext
// CHECK-NOT:  MakePackedPlaintext
module attributes {scheme.ckks} {
  func.func @test_make_packed_plaintext(%cc : !cc, %v : tensor<8xi64>) -> !pt {
    %pt = openfhe.make_packed_plaintext %cc, %v : (!cc, tensor<8xi64>) -> !pt
    return %pt : !pt
  }
}
