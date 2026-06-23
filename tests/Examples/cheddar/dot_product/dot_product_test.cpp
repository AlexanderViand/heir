// End-to-end GPU run of a secret-vector dot product via the cheddar backend.
//
// The whole lowered module is exercised: the generated client helpers
// `dot_product__encrypt__arg{0,1}` / `dot_product__decrypt__result0` plus the
// `dot_product` compute kernel. The harness does NO raw-CHEDDAR-API boundary
// work -- the cheddar IR carries the client helpers (like full HEIR-pipeline
// output, with fixed scales), so this driver just builds the context, calls
// the generated encrypt/compute/decrypt functions, and checks the decrypted
// dot product against the plaintext sum.

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

// Generated entry points (see tests/Examples/cheddar/dot_product/BUILD).
void dot_product__encrypt__arg0(cheddar::Context<word>* ctx,
                                const cheddar::Encoder<word>& encoder, UI* ui,
                                const Evk& evk, float a[1024], UI* ui2,
                                std::array<Ct, 1>& out);
void dot_product__encrypt__arg1(cheddar::Context<word>* ctx,
                                const cheddar::Encoder<word>& encoder, UI* ui,
                                const Evk& evk, float b[1][1024], UI* ui2,
                                std::array<Ct, 1>& out);
void dot_product(cheddar::Context<word>* ctx,
                 const cheddar::Encoder<word>& encoder, UI* ui, const Evk& evk,
                 const std::array<Ct, 1>& a, const std::array<Ct, 1>& b,
                 std::array<Ct, 1>& out);
void dot_product__decrypt__result0(cheddar::Context<word>* ctx,
                                   const cheddar::Encoder<word>& encoder,
                                   UI* ui, const Evk& evk,
                                   const std::array<Ct, 1>& in, UI* ui2,
                                   float* out);

namespace {

constexpr int kN = 1024;
constexpr int kMaxLevel = 1;

const std::vector<int>& RotationDistances() {
  static const std::vector<int> kD = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
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

TEST(CheddarDotProductE2E, GpuRun) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  static float a[1024];
  static float b[1][1024];
  double ref = 0.0;
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(dist(gen));
    b[0][i] = static_cast<float>(dist(gen));
    ref += static_cast<double>(a[i]) * static_cast<double>(b[0][i]);
  }

  auto param = MakeParam();
  auto ctx = cheddar::Context<word>::Create(param);
  auto ui = std::make_unique<UI>(ctx);
  for (int d : RotationDistances()) ui->PrepareRotationKey(d, kMaxLevel);
  const Evk& evk = ui->GetMultiplicationKey();

  std::array<Ct, 1> ca, cb, cout;
  dot_product__encrypt__arg0(ctx.get(), ctx->encoder_, ui.get(), evk, a,
                             ui.get(), ca);
  dot_product__encrypt__arg1(ctx.get(), ctx->encoder_, ui.get(), evk, b,
                             ui.get(), cb);
  dot_product(ctx.get(), ctx->encoder_, ui.get(), evk, ca, cb, cout);
  float result = 0.0f;
  dot_product__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk, cout,
                                ui.get(), &result);

  std::printf("dot: fhe=%.6f ref=%.6f |d|=%.6f\n", result, ref,
              std::abs(static_cast<double>(result) - ref));
  EXPECT_NEAR(static_cast<double>(result), ref, 1e-2);
}
