// End-to-end GPU run of a real CHEDDAR bootstrap via the cheddar backend. The
// lowered module's `boot` kernel calls `static_cast<BootContext<word>*>(ctx)->
// Boot(...)`, so the harness builds a BootContext from CHEDDAR's curated
// bootparam_40_64bit parameter set, runs the bootstrap preparation sequence
// (PrepareEvalMod / PrepareEvalSpecialFFT / AddRequiredRotations -> rotation
// keys), then calls the generated encrypt/boot/decrypt helpers and checks that
// bootstrapping preserves the message (boot(x) ~= x). Mirrors CHEDDAR's own
// Bootstrapping unit test, but driven through HEIR-emitted code.

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
#include "core/EvkRequest.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using Evk = cheddar::EvaluationKey<word>;
using UI = cheddar::UserInterface<word>;
using EvkMap = cheddar::EvkMap<word>;

// Generated entry points (see tests/Examples/cheddar/bootstrap/BUILD).
void boot__encrypt__arg0(cheddar::Context<word>* ctx,
                         const cheddar::Encoder<word>& encoder, UI* ui,
                         const Evk& evk, float a[1024], UI* ui2,
                         std::array<Ct, 1>& out);
void boot(cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
          UI* ui, const Evk& evk, const EvkMap& evk_map,
          const std::array<Ct, 1>& in, std::array<Ct, 1>& out);
void boot__decrypt__result0(cheddar::Context<word>* ctx,
                            const cheddar::Encoder<word>& encoder, UI* ui,
                            const Evk& evk, const std::array<Ct, 1>& in,
                            UI* ui2, float* out);

namespace {
constexpr int kN = 1024;
// The encoded message has 1024 slots (the Encode vector length), so the sparse
// bootstrap operates on 1024 of the logN=16 context's 32768 slots.
constexpr int kNumSlots = 1024;

// CHEDDAR's bootparam_40_64bit parameter set (logN=16, scale 2^40, 26-prime
// main chain, 7 auxiliary primes, num_cts_levels=4, num_stc_levels=3).
cheddar::Parameter<word> MakeParam() {
  std::vector<word> main_primes = {
      1125899908022273ULL,  1099515691009ULL,     1099523555329ULL,
      1099525128193ULL,     1099526176769ULL,     1099529060353ULL,
      1099535220737ULL,     1099536138241ULL,     1099537580033ULL,
      1099538104321ULL,     1099540725761ULL,     1099540856833ULL,
      1099543085057ULL,     36028797019488257ULL, 36028797023420417ULL,
      36028797024206849ULL, 36028797025124353ULL, 36028797032202241ULL,
      36028797033644033ULL, 36028797037576193ULL, 36028797048324097ULL,
      36028797048586241ULL, 36028797049896961ULL, 36028797051863041ULL,
      36028797053698049ULL, 36028797054222337ULL};
  std::vector<word> aux_primes = {72057594038321153ULL, 72057594040680449ULL,
                                  72057594042646529ULL, 72057594047889409ULL,
                                  72057594057195521ULL, 72057594058375169ULL,
                                  72057594058899457ULL};
  std::vector<word> ter_primes = {};
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= 26; ++i) level_config.emplace_back(i, 0);
  std::pair<int, int> additional_base = {0, 0};
  cheddar::Parameter<word> p(/*log_degree=*/16,
                             /*base_scale=*/static_cast<double>(1ULL << 40),
                             /*default_encryption_level=*/13, level_config,
                             main_primes, aux_primes, ter_primes,
                             additional_base);
  p.SetDenseHammingWeight(32768);
  p.SetSparseHammingWeight(32);
  return p;
}
}  // namespace

TEST(CheddarBootstrapE2E, GpuRun) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  static float input[1024];
  for (int i = 0; i < kN; ++i) input[i] = static_cast<float>(dist(gen));

  auto param = MakeParam();
  auto boot_ctx = cheddar::BootContext<word>::Create(
      param, cheddar::BootParameter(param.max_level_, /*num_cts_levels=*/4,
                                    /*num_stc_levels=*/3));
  auto ui = std::make_unique<UI>(boot_ctx);

  // Bootstrap preparation (one-time precompute + rotation keys).
  boot_ctx->PrepareEvalMod();
  boot_ctx->PrepareEvalSpecialFFT(kNumSlots);
  cheddar::EvkRequest req;
  boot_ctx->AddRequiredRotations(req, kNumSlots);
  ui->PrepareRotationKey(req);
  const EvkMap& evk_map = ui->GetEvkMap();
  const Evk& evk = ui->GetMultiplicationKey();

  std::array<Ct, 1> cin, cout;
  boot__encrypt__arg0(boot_ctx.get(), boot_ctx->encoder_, ui.get(), evk, input,
                      ui.get(), cin);
  boot(boot_ctx.get(), boot_ctx->encoder_, ui.get(), evk, evk_map, cin, cout);
  static float result[1024];
  boot__decrypt__result0(boot_ctx.get(), boot_ctx->encoder_, ui.get(), evk,
                         cout, ui.get(), result);

  double max_abs = 0;
  for (int i = 0; i < kN; ++i)
    max_abs =
        std::max(max_abs, std::abs(static_cast<double>(result[i]) - input[i]));
  std::printf("bootstrap: max|d|=%.6e (e.g. fhe[0]=%.6f in[0]=%.6f)\n", max_abs,
              result[0], input[0]);
  // Bootstrapping is approximate; expect the message preserved to CKKS boot
  // precision (observed max|d| ~7e-7 for this param set).
  EXPECT_LT(max_abs, 1e-4);
}
