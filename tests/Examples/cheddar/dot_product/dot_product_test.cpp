// End-to-end test for the 8-element dot product kernel emitted from
// dot_product.mlir.  Exercises `cheddar.hmult` + chained `cheddar.hrot_add`,
// which combined cover the bulk of what a CKKS-to-CHEDDAR lowering needs to
// emit for inner-product-shaped workloads.

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "tests/Examples/cheddar/cheddar_test_fixture.h"

using cheddar::Context;
using cheddar::UserInterface;
using cheddar_test::Ct;
using cheddar_test::Evk;
using cheddar_test::word;

void dot_product_kernel(Context<word>* ctx, UserInterface<word>* ui,
                        const Evk& mult_key, const Ct& a, const Ct& b, Ct& out);

TEST(CheddarDotProductE2E, EightElements) {
  auto f = cheddar_test::MakeFixture();
  // The kernel needs rotation keys for distances 1, 2, and 4 prepared at the
  // ciphertext level it'll operate at -- which is `top_level - 1` because
  // hmult drops one level via its built-in rescale.
  f.ui->PrepareRotationKey(1, f.top_level);
  f.ui->PrepareRotationKey(2, f.top_level);
  f.ui->PrepareRotationKey(4, f.top_level);

  std::vector<double> a = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<double> b = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  double expected = 0.0;
  for (size_t i = 0; i < a.size(); ++i) expected += a[i] * b[i];

  auto ct_a = cheddar_test::EncryptVec(f, a, f.top_level);
  auto ct_b = cheddar_test::EncryptVec(f, b, f.top_level);
  Ct ct_r;
  dot_product_kernel(f.ctx.get(), f.ui.get(), f.ui->GetMultiplicationKey(),
                     ct_a, ct_b, ct_r);
  auto result = cheddar_test::DecryptVec(f, ct_r, 1);

  EXPECT_NEAR(result[0], expected, 0.01)
      << "Dot product slot[0] doesn't match expected value (got " << result[0]
      << ", expected " << expected << ")";
}
