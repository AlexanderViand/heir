// End-to-end GPU run of an element-wise ciphertext product via the cheddar
// backend. The whole lowered module is exercised -- the generated client
// helpers `mult__encrypt__arg{0,1}` / `mult__decrypt__result0` plus the `mult`
// compute kernel -- so the harness does no raw-CHEDDAR-API boundary work: it
// builds the context, calls the generated functions, and checks the decrypted
// product against the plaintext one.

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

void mult__encrypt__arg0(cheddar::Context<word>* ctx,
                         const cheddar::Encoder<word>& encoder, UI* ui,
                         const Evk& evk, float a[1024], UI* ui2,
                         std::array<Ct, 1>& out);
void mult__encrypt__arg1(cheddar::Context<word>* ctx,
                         const cheddar::Encoder<word>& encoder, UI* ui,
                         const Evk& evk, float b[1024], UI* ui2,
                         std::array<Ct, 1>& out);
void mult(cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
          UI* ui, const Evk& evk, const std::array<Ct, 1>& a,
          const std::array<Ct, 1>& b, std::array<Ct, 1>& out);
void mult__decrypt__result0(cheddar::Context<word>* ctx,
                            const cheddar::Encoder<word>& encoder, UI* ui,
                            const Evk& evk, const std::array<Ct, 1>& in,
                            UI* ui2, float* out);

namespace {
constexpr int kN = 1024;
constexpr int kMaxLevel = 1;
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

TEST(CheddarMultE2E, GpuRun) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  static float a[1024], b[1024];
  static float ref[1024];
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(dist(gen));
    b[i] = static_cast<float>(dist(gen));
    ref[i] = a[i] * b[i];
  }

  auto param = MakeParam();
  auto ctx = cheddar::Context<word>::Create(param);
  auto ui = std::make_unique<UI>(ctx);
  const Evk& evk = ui->GetMultiplicationKey();

  std::array<Ct, 1> ca, cb, cout;
  mult__encrypt__arg0(ctx.get(), ctx->encoder_, ui.get(), evk, a, ui.get(), ca);
  mult__encrypt__arg1(ctx.get(), ctx->encoder_, ui.get(), evk, b, ui.get(), cb);
  mult(ctx.get(), ctx->encoder_, ui.get(), evk, ca, cb, cout);
  static float result[1024];
  mult__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk, cout,
                         ui.get(), result);

  double max_abs = 0;
  for (int i = 0; i < kN; ++i)
    max_abs =
        std::max(max_abs, std::abs(static_cast<double>(result[i]) - ref[i]));
  std::printf("mult: max|d|=%.6e (e.g. fhe[0]=%.5f ref[0]=%.5f)\n", max_abs,
              result[0], ref[0]);
  EXPECT_LT(max_abs, 1e-2);
}
