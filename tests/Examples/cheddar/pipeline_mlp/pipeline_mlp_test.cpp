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
void tiny_mlp__encrypt_inputs(cheddar::Context<word>* ctx, UI* ui,
                              float input[4], Ct out[1]);
void tiny_mlp__preprocess(cheddar::Context<word>* ctx,
                          std::shared_ptr<LinearTransform> transforms[2]);
void tiny_mlp__evaluate(cheddar::Context<word>* ctx, const EvkMap& evk_map,
                        const Ct input[1],
                        const std::shared_ptr<LinearTransform> transforms[2],
                        Ct out[1]);
void tiny_mlp__decrypt_outputs(cheddar::Context<word>* ctx, UI* ui,
                               const Ct input[1], float out[2]);

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
  const EvkMap& evk_map = ui->GetEvkMap();

  Ct encrypted[1];
  tiny_mlp__encrypt_inputs(ctx.get(), ui.get(), input, encrypted);
  std::shared_ptr<LinearTransform> transforms[2];
  tiny_mlp__preprocess(ctx.get(), transforms);
  ASSERT_NE(transforms[0], nullptr);
  ASSERT_NE(transforms[1], nullptr);

  Ct evaluated[1];
  Ct repeated[1];
  tiny_mlp__evaluate(ctx.get(), evk_map, encrypted, transforms, evaluated);
  tiny_mlp__evaluate(ctx.get(), evk_map, encrypted, transforms, repeated);
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted[0].GetNP()), 4);
  EXPECT_EQ(ctx->param_.NPToLevel(evaluated[0].GetNP()), 0);

  float actual[2];
  float actual_repeated[2];
  tiny_mlp__decrypt_outputs(ctx.get(), ui.get(), evaluated, actual);
  tiny_mlp__decrypt_outputs(ctx.get(), ui.get(), repeated, actual_repeated);
  for (int row = 0; row < 2; ++row) {
    ASSERT_TRUE(std::isfinite(actual[row]));
    ASSERT_TRUE(std::isfinite(actual_repeated[row]));
    EXPECT_NEAR(actual[row], expected[row], 1e-3) << row;
    EXPECT_NEAR(actual_repeated[row], actual[row], 1e-3) << row;
  }
}
