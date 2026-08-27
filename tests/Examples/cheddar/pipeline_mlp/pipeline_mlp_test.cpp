#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>

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

void tiny_mlp__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                         std::unique_ptr<UI>& ui);
void tiny_mlp__encrypt__arg0(cheddar::Context<word>* ctx,
                             const cheddar::Encoder<word>& encoder,
                             const Evk& evk, float input[4], UI* ui,
                             std::array<Ct, 1>& out);
void tiny_mlp__preprocessing(
    cheddar::Context<word>* ctx,
    std::array<std::shared_ptr<LinearTransform>, 2>& transforms);
void tiny_mlp__preprocessed(
    cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder, UI* ui,
    const Evk& evk, const EvkMap& evk_map, const std::array<Ct, 1>& input,
    const std::array<std::shared_ptr<LinearTransform>, 2>& transforms,
    std::array<Ct, 1>& out);
void tiny_mlp__decrypt__result0(cheddar::Context<word>* ctx,
                                const cheddar::Encoder<word>& encoder,
                                const Evk& evk, const std::array<Ct, 1>& input,
                                UI* ui, float* out);

TEST(CheddarPipelineMlpE2E, MatchesPlaintextNetworkAndReusesWeights) {
  constexpr float kW0[8][4] = {
      {0.25F, -0.10F, 0.20F, 0.05F}, {-0.15F, 0.30F, 0.10F, -0.20F},
      {0.05F, 0.10F, -0.25F, 0.20F}, {0.20F, 0.15F, 0.05F, -0.10F},
      {-0.10F, 0.05F, 0.30F, 0.15F}, {0.15F, -0.20F, 0.10F, 0.25F},
      {0.10F, 0.20F, -0.15F, 0.05F}, {-0.05F, 0.25F, 0.20F, -0.15F},
  };
  constexpr float kW1[2][8] = {
      {0.20F, -0.10F, 0.15F, 0.05F, -0.20F, 0.10F, 0.25F, -0.05F},
      {-0.15F, 0.20F, 0.05F, -0.10F, 0.15F, 0.25F, -0.05F, 0.10F},
  };
  float input[4] = {0.60F, -0.40F, 0.25F, 0.80F};
  float hidden[8] = {};
  float expected[2] = {};
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 4; ++col) hidden[row] += kW0[row][col] * input[col];
    ASSERT_LE(std::abs(hidden[row]), 1.0F);
    hidden[row] = hidden[row] * hidden[row] * hidden[row];
  }
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 8; ++col)
      expected[row] += kW1[row][col] * hidden[col];

  std::shared_ptr<cheddar::Context<word>> ctx;
  std::unique_ptr<UI> ui;
  tiny_mlp__configure(ctx, ui);
  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(ui, nullptr);
  EXPECT_NO_THROW(ui->GetRotationKey(1));
  EXPECT_NO_THROW(ui->GetRotationKey(2));
  EXPECT_NO_THROW(ui->GetRotationKey(3));
  EXPECT_NO_THROW(ui->GetRotationKey(4));
  const Evk& evk = ui->GetMultiplicationKey();
  const EvkMap& evk_map = ui->GetEvkMap();

  std::array<Ct, 1> encrypted;
  tiny_mlp__encrypt__arg0(ctx.get(), ctx->encoder_, evk, input, ui.get(),
                          encrypted);
  std::array<std::shared_ptr<LinearTransform>, 2> transforms;
  tiny_mlp__preprocessing(ctx.get(), transforms);
  ASSERT_NE(transforms[0], nullptr);
  ASSERT_NE(transforms[1], nullptr);

  std::array<Ct, 1> evaluated;
  std::array<Ct, 1> repeated;
  tiny_mlp__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map,
                         encrypted, transforms, evaluated);
  tiny_mlp__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map,
                         encrypted, transforms, repeated);
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted[0].GetNP()), 4);
  EXPECT_EQ(ctx->param_.NPToLevel(evaluated[0].GetNP()), 0);

  float actual[2];
  float actual_repeated[2];
  tiny_mlp__decrypt__result0(ctx.get(), ctx->encoder_, evk, evaluated, ui.get(),
                             actual);
  tiny_mlp__decrypt__result0(ctx.get(), ctx->encoder_, evk, repeated, ui.get(),
                             actual_repeated);
  for (int row = 0; row < 2; ++row) {
    ASSERT_TRUE(std::isfinite(actual[row]));
    ASSERT_TRUE(std::isfinite(actual_repeated[row]));
    EXPECT_NEAR(actual[row], expected[row], 1e-3) << row;
    EXPECT_NEAR(actual_repeated[row], actual[row], 1e-3) << row;
  }
}
