// End-to-end test for the CHEDDAR ciphertext-ciphertext multiplication
// path: `cheddar.mult` (degree-3 output) + `cheddar.relinearize_rescale`,
// and the fused `cheddar.hmult`.  Verifies they produce equivalent results.

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "tests/Examples/cheddar/cheddar_test_fixture.h"

using cheddar::Context;
using cheddar_test::Ct;
using cheddar_test::Evk;
using cheddar_test::word;

// Generated kernels.
void mult_kernel(Context<word>* ctx, const Evk& mult_key, const Ct& a,
                 const Ct& b, Ct& out);
void hmult_kernel(Context<word>* ctx, const Evk& mult_key, const Ct& a,
                  const Ct& b, Ct& out);

namespace {

constexpr double kTol = 0.05;

}  // namespace

TEST(CheddarMultE2E, Mult) {
  auto f = cheddar_test::MakeFixture();
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {0.5, 1.5, 2.5, 3.5};

  auto ct_a = cheddar_test::EncryptVec(f, a, f.top_level);
  auto ct_b = cheddar_test::EncryptVec(f, b, f.top_level);
  Ct ct_r;
  mult_kernel(f.ctx.get(), f.ui->GetMultiplicationKey(), ct_a, ct_b, ct_r);
  auto result = cheddar_test::DecryptVec(f, ct_r, a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(result[i], a[i] * b[i], kTol);
  }
}

TEST(CheddarMultE2E, HMult) {
  auto f = cheddar_test::MakeFixture();
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {0.5, 1.5, 2.5, 3.5};

  auto ct_a = cheddar_test::EncryptVec(f, a, f.top_level);
  auto ct_b = cheddar_test::EncryptVec(f, b, f.top_level);
  Ct ct_r;
  hmult_kernel(f.ctx.get(), f.ui->GetMultiplicationKey(), ct_a, ct_b, ct_r);
  auto result = cheddar_test::DecryptVec(f, ct_r, a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(result[i], a[i] * b[i], kTol);
  }
}
