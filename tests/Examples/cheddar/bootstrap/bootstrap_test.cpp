// End-to-end GPU run of a real CHEDDAR bootstrap via the cheddar backend. The
// generated module builds and prepares a BootContext from CHEDDAR's curated
// bootparam_40_64bit parameter set, then the harness calls the generated
// encrypt/boot/decrypt helpers and checks that bootstrapping refreshes the
// level while preserving the message (boot(x) ~= x).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
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
void boot__configure(std::shared_ptr<cheddar::BootContext<word>>& boot_ctx,
                     std::unique_ptr<UI>& ui);
void boot__encrypt__arg0(cheddar::Context<word>* ctx,
                         const cheddar::Encoder<word>& encoder, UI* ui,
                         const Evk& evk, float a[8], UI* ui2,
                         std::array<Ct, 1>& out);
void boot(cheddar::BootContext<word>* ctx,
          const cheddar::Encoder<word>& encoder, UI* ui, const Evk& evk,
          const EvkMap& evk_map, const std::array<Ct, 1>& in,
          std::array<Ct, 1>& out);
void boot__decrypt__result0(cheddar::Context<word>* ctx,
                            const cheddar::Encoder<word>& encoder, UI* ui,
                            const Evk& evk, const std::array<Ct, 1>& in,
                            UI* ui2, float* out);

namespace {
constexpr int kN = 8;
}  // namespace

TEST(CheddarBootstrapE2E, GpuRun) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  static float input[kN];
  for (int i = 0; i < kN; ++i) input[i] = static_cast<float>(dist(gen));

  std::shared_ptr<cheddar::BootContext<word>> boot_ctx;
  std::unique_ptr<UI> ui;
  boot__configure(boot_ctx, ui);
  ASSERT_NE(boot_ctx, nullptr);
  ASSERT_NE(ui, nullptr);
  EXPECT_EQ(boot_ctx->param_.default_encryption_level_, 13);
  EXPECT_EQ(boot_ctx->boot_param_.GetEndLevel(), 10);
  const EvkMap& evk_map = ui->GetEvkMap();
  const Evk& evk = ui->GetMultiplicationKey();

  std::array<Ct, 1> cin, cout;
  boot__encrypt__arg0(boot_ctx.get(), boot_ctx->encoder_, ui.get(), evk, input,
                      ui.get(), cin);
  boot(boot_ctx.get(), boot_ctx->encoder_, ui.get(), evk, evk_map, cin, cout);
  EXPECT_EQ(cin[0].GetNumSlots(), kN);
  EXPECT_EQ(cout[0].GetNumSlots(), kN);
  EXPECT_EQ(boot_ctx->param_.NPToLevel(cin[0].GetNP()), 0);
  EXPECT_EQ(boot_ctx->param_.NPToLevel(cout[0].GetNP()),
            boot_ctx->boot_param_.GetEndLevel());
  EXPECT_EQ(boot_ctx->param_.NPToLevel(cout[0].GetNP()), 10);
  static float result[kN];
  boot__decrypt__result0(boot_ctx.get(), boot_ctx->encoder_, ui.get(), evk,
                         cout, ui.get(), result);

  double max_abs = 0;
  for (int i = 0; i < kN; ++i) {
    double error = std::abs(static_cast<double>(result[i]) - input[i]);
    ASSERT_TRUE(std::isfinite(error)) << "non-finite result at slot " << i;
    max_abs = std::max(max_abs, error);
  }
  std::printf("bootstrap: max|d|=%.6e (e.g. fhe[0]=%.6f in[0]=%.6f)\n", max_abs,
              result[0], input[0]);
  // Bootstrapping is approximate; expect the message preserved to CKKS boot
  // precision (observed max|d| ~7e-7 for this param set).
  EXPECT_LT(max_abs, 1e-4);
}
