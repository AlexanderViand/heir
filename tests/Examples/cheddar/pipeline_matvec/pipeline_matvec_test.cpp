#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "matvec.h"

namespace generated = heir::generated::matvec;

TEST(CheddarPipelineMatvecE2E, GeneratedPipelineAndSetupRun) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  generated::Input0 input;
  static float expected[8];
  for (float& value : input) value = dist(gen);
  for (int row = 0; row < 8; ++row) {
    double sum = 0.0;
    for (int column = 0; column < 4; ++column) {
      float weight = static_cast<float>(4 * row + column);
      sum += static_cast<double>(weight) * input[column];
    }
    expected[row] = static_cast<float>(sum);
  }

  auto ctx = generated::Setup();
  auto keys = generated::KeyGen(ctx);
  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(keys.storage, nullptr);
  EXPECT_NO_THROW(keys.storage->GetRotationKey(1));
  EXPECT_NO_THROW(keys.storage->GetRotationKey(2));
  EXPECT_NO_THROW(keys.storage->GetRotationKey(3));

  auto prepared = generated::Preprocess(*ctx, keys.public_key, "");
  ASSERT_NE(std::get<0>(prepared)[0], nullptr);
  generated::CleartextInputs inputs{input};
  auto encrypted = generated::Encrypt(*ctx, keys.secret_key, inputs);
  auto evaluated =
      generated::Evaluate(*ctx, keys.public_key, prepared, encrypted);
  auto repeated =
      generated::Evaluate(*ctx, keys.public_key, prepared, encrypted);

  EXPECT_EQ(ctx->param_.NPToLevel(std::get<0>(encrypted)[0].GetNP()), 1);
  EXPECT_EQ(ctx->param_.NPToLevel(std::get<0>(evaluated)[0].GetNP()), 0);

  generated::Output0 actual =
      generated::Decrypt(*ctx, keys.secret_key, evaluated);
  generated::Output0 repeatedActual =
      generated::Decrypt(*ctx, keys.secret_key, repeated);

  double max_abs_error = 0.0;
  for (int row = 0; row < 8; ++row) {
    ASSERT_TRUE(std::isfinite(actual[row]));
    ASSERT_TRUE(std::isfinite(repeatedActual[row]));
    double error = std::abs(static_cast<double>(actual[row]) - expected[row]);
    ASSERT_TRUE(std::isfinite(error)) << "non-finite error at row " << row;
    EXPECT_NEAR(repeatedActual[row], actual[row], 1e-2)
        << "preprocessed storage changed after first evaluation at row " << row;
    max_abs_error = std::max(max_abs_error, error);
  }
  EXPECT_LT(max_abs_error, 1e-2);
}
