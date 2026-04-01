#include <cmath>
#include <string>
#include <vector>

#include "fideslib/fideslib.hpp"
#include "gtest/gtest.h"

namespace {

bool approx(double actual, double expected, double tol) {
  return std::abs(actual - expected) <= tol;
}

}  // namespace

TEST(FideslibSmokeTest, EvalAddEvalMultDecrypt) {
  fideslib::CCParams<fideslib::CryptoContextCKKSRNS> params;
  params.SetMultiplicativeDepth(3);
  params.SetScalingModSize(50);
  params.SetBatchSize(8);

  auto cc = fideslib::GenCryptoContext(params);
  cc->Enable(fideslib::PKE);
  cc->Enable(fideslib::KEYSWITCH);
  cc->Enable(fideslib::LEVELEDSHE);

  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc->LoadContext(keys.publicKey);

  std::vector<double> lhs = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> rhs = {5.0, 6.0, 7.0, 8.0};

  auto pt_lhs = cc->MakeCKKSPackedPlaintext(lhs);
  auto pt_rhs = cc->MakeCKKSPackedPlaintext(rhs);

  auto ct_lhs = cc->Encrypt(keys.publicKey, pt_lhs);
  auto ct_rhs = cc->Encrypt(keys.publicKey, pt_rhs);

  auto ct_add = cc->EvalAdd(ct_lhs, ct_rhs);
  auto ct_mul = cc->EvalMult(ct_lhs, ct_rhs);

  fideslib::Plaintext pt_add;
  fideslib::Plaintext pt_mul;

  auto dec_add = cc->Decrypt(ct_add, keys.secretKey, &pt_add);
  auto dec_mul = cc->Decrypt(ct_mul, keys.secretKey, &pt_mul);
  ASSERT_TRUE(dec_add.isValid);
  ASSERT_TRUE(dec_mul.isValid);
  ASSERT_NE(pt_add, nullptr);
  ASSERT_NE(pt_mul, nullptr);

  auto add_vals = pt_add->GetRealPackedValue();
  auto mul_vals = pt_mul->GetRealPackedValue();
  ASSERT_GE(add_vals.size(), 4u);
  ASSERT_GE(mul_vals.size(), 4u);

  const std::vector<double> expect_add = {6.0, 8.0, 10.0, 12.0};
  const std::vector<double> expect_mul = {5.0, 12.0, 21.0, 32.0};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(approx(add_vals[i], expect_add[i], 0.25))
        << "slot=" << i << " add=" << add_vals[i];
    EXPECT_TRUE(approx(mul_vals[i], expect_mul[i], 0.75))
        << "slot=" << i << " mul=" << mul_vals[i];
  }
}
