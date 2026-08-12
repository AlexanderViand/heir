#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using Evk = cheddar::EvaluationKey<word>;
using EvkMap = cheddar::EvkMap<word>;
using UI = cheddar::UserInterface<word>;

void chebyshev_cube__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                               std::unique_ptr<UI>& ui);
void chebyshev_cube__encrypt__arg0(cheddar::Context<word>* ctx,
                                   const cheddar::Encoder<word>& encoder,
                                   const Evk& evk, float input[8], UI* ui,
                                   std::array<Ct, 1>& out);
void chebyshev_cube(cheddar::Context<word>* ctx,
                    const cheddar::Encoder<word>& encoder, UI* ui,
                    const Evk& evk, const EvkMap& evk_map,
                    const std::array<Ct, 1>& input, std::array<Ct, 1>& out);
void chebyshev_cube__decrypt__result0(cheddar::Context<word>* ctx,
                                      const cheddar::Encoder<word>& encoder,
                                      const Evk& evk,
                                      const std::array<Ct, 1>& input, UI* ui,
                                      float* out);

TEST(CheddarPipelineEvalChebyshevE2E, MatchesPlaintextDegreeFive) {
  float input[8] = {-0.9F, -0.7F, -0.25F, 0.0F, 0.1F, 0.35F, 0.65F, 0.9F};
  std::shared_ptr<cheddar::Context<word>> ctx;
  std::unique_ptr<UI> ui;
  chebyshev_cube__configure(ctx, ui);
  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(ui, nullptr);

  const Evk& evk = ui->GetMultiplicationKey();
  const EvkMap& evk_map = ui->GetEvkMap();
  std::array<Ct, 1> encrypted;
  std::array<Ct, 1> evaluated;
  chebyshev_cube__encrypt__arg0(ctx.get(), ctx->encoder_, evk, input, ui.get(),
                                encrypted);
  chebyshev_cube(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map, encrypted,
                 evaluated);
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted[0].GetNP()), 3);
  EXPECT_EQ(ctx->param_.NPToLevel(evaluated[0].GetNP()), 0);

  float actual[8];
  chebyshev_cube__decrypt__result0(ctx.get(), ctx->encoder_, evk, evaluated,
                                   ui.get(), actual);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(std::isfinite(actual[i]));
    EXPECT_NEAR(actual[i], std::pow(input[i], 5), 1e-3) << i;
  }
}
