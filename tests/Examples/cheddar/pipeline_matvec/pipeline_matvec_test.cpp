#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "extension/LinearTransform.h"

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using Evk = cheddar::EvaluationKey<word>;
using EvkMap = cheddar::EvkMap<word>;
using LinearTransform = cheddar::LinearTransform<word>;
using UI = cheddar::UserInterface<word>;

void matvec__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                       std::unique_ptr<UI>& ui);
void matvec__encrypt__arg0(cheddar::Context<word>* ctx,
                           const cheddar::Encoder<word>& encoder,
                           const Evk& evk, float x[4], UI* ui,
                           std::array<Ct, 1>& out);
void matvec(cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
            UI* ui, const Evk& evk, const EvkMap& evk_map,
            const std::array<Ct, 1>& x, std::array<Ct, 1>& out);
void matvec__preprocessing(cheddar::Context<word>* ctx,
                           std::shared_ptr<LinearTransform>& transform);
void matvec__preprocessed(cheddar::Context<word>* ctx,
                          const cheddar::Encoder<word>& encoder, UI* ui,
                          const Evk& evk, const EvkMap& evk_map,
                          const std::array<Ct, 1>& input,
                          const std::shared_ptr<LinearTransform>& transform,
                          std::array<Ct, 1>& out);
void matvec__decrypt__result0(cheddar::Context<word>* ctx,
                              const cheddar::Encoder<word>& encoder,
                              const Evk& evk, const std::array<Ct, 1>& in,
                              UI* ui, float* out);

TEST(CheddarPipelineMatvecE2E, GeneratedPipelineAndSetupRun) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  static float x[4];
  static float expected[8];
  for (float& value : x) value = dist(gen);
  for (int row = 0; row < 8; ++row) {
    double sum = 0.0;
    for (int column = 0; column < 4; ++column) {
      float weight = static_cast<float>(4 * row + column);
      sum += static_cast<double>(weight) * x[column];
    }
    expected[row] = static_cast<float>(sum);
  }

  std::shared_ptr<cheddar::Context<word>> ctx;
  std::unique_ptr<UI> ui;
  matvec__configure(ctx, ui);
  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(ui, nullptr);
  EXPECT_NO_THROW(ui->GetRotationKey(1));
  EXPECT_NO_THROW(ui->GetRotationKey(2));
  EXPECT_NO_THROW(ui->GetRotationKey(3));
  const Evk& evk = ui->GetMultiplicationKey();
  const EvkMap& evk_map = ui->GetEvkMap();

  std::array<Ct, 1> encrypted_input;
  std::shared_ptr<LinearTransform> transform;
  matvec__preprocessing(ctx.get(), transform);
  ASSERT_NE(transform, nullptr);

  std::array<Ct, 1> encrypted_output;
  std::array<Ct, 1> repeated_output;
  std::array<Ct, 1> wrapper_output;
  matvec__encrypt__arg0(ctx.get(), ctx->encoder_, evk, x, ui.get(),
                        encrypted_input);
  matvec__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map,
                       encrypted_input, transform, encrypted_output);
  matvec__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map,
                       encrypted_input, transform, repeated_output);
  matvec(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map, encrypted_input,
         wrapper_output);
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted_input[0].GetNP()), 1);
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted_output[0].GetNP()), 0);

  static float actual[8];
  static float repeated[8];
  static float wrapper[8];
  matvec__decrypt__result0(ctx.get(), ctx->encoder_, evk, encrypted_output,
                           ui.get(), actual);
  matvec__decrypt__result0(ctx.get(), ctx->encoder_, evk, repeated_output,
                           ui.get(), repeated);
  matvec__decrypt__result0(ctx.get(), ctx->encoder_, evk, wrapper_output,
                           ui.get(), wrapper);

  double max_abs_error = 0.0;
  for (int row = 0; row < 8; ++row) {
    ASSERT_TRUE(std::isfinite(actual[row]));
    ASSERT_TRUE(std::isfinite(repeated[row]));
    ASSERT_TRUE(std::isfinite(wrapper[row]));
    double error = std::abs(static_cast<double>(actual[row]) - expected[row]);
    ASSERT_TRUE(std::isfinite(error)) << "non-finite error at row " << row;
    EXPECT_NEAR(repeated[row], actual[row], 1e-2)
        << "preprocessed storage changed after first evaluation at row " << row;
    EXPECT_NEAR(wrapper[row], actual[row], 1e-2)
        << "public wrapper disagreed at row " << row;
    max_abs_error = std::max(max_abs_error, error);
  }
  EXPECT_LT(max_abs_error, 1e-2);
}
