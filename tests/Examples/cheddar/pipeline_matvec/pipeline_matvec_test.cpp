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

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using Evk = cheddar::EvaluationKey<word>;
using Pt = cheddar::Plaintext<word>;
using UI = cheddar::UserInterface<word>;

void matvec__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                       std::unique_ptr<UI>& ui);
void matvec__encrypt__arg0(cheddar::Context<word>* ctx,
                           const cheddar::Encoder<word>& encoder, UI* ui,
                           const Evk& evk, float x[4], UI* ui2,
                           std::array<Ct, 1>& out);
void matvec(cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
            UI* ui, const Evk& evk, const std::array<Ct, 1>& x, float W[8][4],
            std::array<Ct, 1>& out);
void matvec__preprocessing(const cheddar::Encoder<word>& encoder,
                           std::array<Pt, 4>& diagonals);
void matvec__preprocessed(cheddar::Context<word>* ctx,
                          const cheddar::Encoder<word>& encoder, UI* ui,
                          const Evk& evk, const std::array<Ct, 1>& input,
                          const std::array<Pt, 4>& diagonals,
                          std::array<Ct, 1>& out);
void matvec__decrypt__result0(cheddar::Context<word>* ctx,
                              const cheddar::Encoder<word>& encoder, UI* ui,
                              const Evk& evk, const std::array<Ct, 1>& in,
                              UI* ui2, float* out);

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
  const Evk& evk = ui->GetMultiplicationKey();

  std::array<Ct, 1> encrypted_input;
  std::array<Pt, 4> diagonals;
  matvec__preprocessing(ctx->encoder_, diagonals);

  std::array<Ct, 1> encrypted_output;
  std::array<Ct, 1> repeated_output;
  matvec__encrypt__arg0(ctx.get(), ctx->encoder_, ui.get(), evk, x, ui.get(),
                        encrypted_input);
  matvec__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, encrypted_input,
                       diagonals, encrypted_output);
  matvec__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, encrypted_input,
                       diagonals, repeated_output);

  static float actual[8];
  static float repeated[8];
  matvec__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk,
                           encrypted_output, ui.get(), actual);
  matvec__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk,
                           repeated_output, ui.get(), repeated);

  double max_abs_error = 0.0;
  for (int row = 0; row < 8; ++row) {
    ASSERT_TRUE(std::isfinite(actual[row]));
    ASSERT_TRUE(std::isfinite(repeated[row]));
    double error = std::abs(static_cast<double>(actual[row]) - expected[row]);
    ASSERT_TRUE(std::isfinite(error)) << "non-finite error at row " << row;
    EXPECT_NEAR(repeated[row], actual[row], 1e-2)
        << "preprocessed storage changed after first evaluation at row " << row;
    max_abs_error = std::max(max_abs_error, error);
  }
  EXPECT_LT(max_abs_error, 1e-2);
}
