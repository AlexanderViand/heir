// End-to-end GPU run of a matrix-vector product (public 8x4 weights times a
// secret 4-vector) via the cheddar backend. The whole lowered module is
// exercised -- the generated `matvec__encrypt__arg0` (encrypt x),
// `matvec__decrypt__result0`, the `matvec__preprocessing` that encodes the
// weight diagonals, and the combined `matvec` kernel -- so the harness just
// calls the generated functions and checks against the plaintext product.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using Evk = cheddar::EvaluationKey<word>;
using UI = cheddar::UserInterface<word>;

void matvec__encrypt__arg0(cheddar::Context<word>* ctx,
                           const cheddar::Encoder<word>& encoder, UI* ui,
                           const Evk& evk, float x[4], UI* ui2,
                           std::array<Ct, 1>& out);
void matvec(cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
            UI* ui, const Evk& evk, const std::array<Ct, 1>& x, float W[8][4],
            std::array<Ct, 1>& out);
void matvec__decrypt__result0(cheddar::Context<word>* ctx,
                              const cheddar::Encoder<word>& encoder, UI* ui,
                              const Evk& evk, const std::array<Ct, 1>& in,
                              UI* ui2, float* out);

namespace {
constexpr int kMaxLevel = 1;
const std::vector<int>& RotationDistances() {
  static const std::vector<int> kD = {1, 2};
  return kD;
}
cheddar::Parameter<word> MakeParam() {
  std::vector<word> main_primes = {36028797018652673ULL, 35184372121601ULL};
  std::vector<word> aux_primes = {1152921504606994433ULL};
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i)
    level_config.emplace_back(i, 0);
  return cheddar::Parameter<word>(
      /*logN=*/13, /*scale=*/static_cast<double>(1ULL << 45), kMaxLevel,
      level_config, main_primes, aux_primes);
}
}  // namespace

TEST(CheddarMatvecE2E, GpuRun) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  static float x[4];
  static float W[8][4];
  static float ref[8];
  for (int i = 0; i < 4; ++i) x[i] = static_cast<float>(dist(gen));
  for (int o = 0; o < 8; ++o) {
    double acc = 0;
    for (int i = 0; i < 4; ++i) {
      W[o][i] = static_cast<float>(dist(gen));
      acc += static_cast<double>(W[o][i]) * x[i];
    }
    ref[o] = static_cast<float>(acc);
  }

  auto param = MakeParam();
  auto ctx = cheddar::Context<word>::Create(param);
  auto ui = std::make_unique<UI>(ctx);
  for (int d : RotationDistances()) ui->PrepareRotationKey(d, kMaxLevel);
  const Evk& evk = ui->GetMultiplicationKey();

  std::array<Ct, 1> cx, cout;
  matvec__encrypt__arg0(ctx.get(), ctx->encoder_, ui.get(), evk, x, ui.get(),
                        cx);
  matvec(ctx.get(), ctx->encoder_, ui.get(), evk, cx, W, cout);
  static float result[8];
  matvec__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk, cout,
                           ui.get(), result);

  double max_abs = 0;
  for (int o = 0; o < 8; ++o)
    max_abs =
        std::max(max_abs, std::abs(static_cast<double>(result[o]) - ref[o]));
  std::printf("matvec: max|d|=%.6e (e.g. fhe[0]=%.5f ref[0]=%.5f)\n", max_abs,
              result[0], ref[0]);
  EXPECT_LT(max_abs, 1e-2);
}
