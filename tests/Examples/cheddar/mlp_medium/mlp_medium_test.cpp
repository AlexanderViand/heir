// End-to-end GPU run of "mlp-medium": the OrionMLP shape (784 -> 128 -> 128 ->
// 10, BatchNorm folded into the linears) but with SiLU(degree-7 Chebyshev)
// activations instead of Quad. Lowered from the orion-extension IR via the
// Orion -> CHEDDAR patterns: three cheddar.linear_transform ops (BSGS matvec)
// and two cheddar.eval_poly ops (the Chebyshev SiLU, on the canonical domain
// [-1, 1]). This is the chebyshev/eval_poly counterpart to orion_mlp_lt (which
// uses Quad); it needs ~9 multiplicative levels (logN=14, 13-prime chain) and
// runs WITHOUT bootstrapping.
//
// Reference: the PyTorch plaintext forward pass of the same model on the same
// input (MNIST test image 0), checked in as data/expected_logits.bin. The FHE
// run reproduces these 10 logits in slots 0..9; the difference is CKKS noise
// plus the degree-7 Chebyshev SiLU approximation the FHE circuit evaluates.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/EvkRequest.h"
#include "core/Parameter.h"
#include "extension/LinearTransform.h"
#include "extension/StripedMatrix.h"

using word = uint64_t;
using Complex = std::complex<double>;
using Ct = cheddar::Ciphertext<word>;
using Pt = cheddar::Plaintext<word>;
using Evk = cheddar::EvaluationKey<word>;
using EvkMap = cheddar::EvkMap<word>;
using UI = cheddar::UserInterface<word>;

// Generated entry point (see BUILD), signature confirmed against the lowered
// mlp_medium.mlir / mlp_medium_raw.cc.
void mlpmedium(cheddar::Context<word>* ctx,
               const cheddar::Encoder<word>& encoder, UI* ui, const Evk& evk,
               const EvkMap& evk_map, const Ct& in, double fc1_w[128][8192],
               double fc1_b[8192], double fc2_w[128][8192], double fc2_b[8192],
               double fc3_w[137][8192], double fc3_b[8192], Ct& out);

namespace {

constexpr int kSlots = 8192;
constexpr int kInputLevel = 9;       // the model's input ciphertext level
constexpr int kDefaultEncLevel = 12;  // top of the 13-prime chain
constexpr int kNumClasses = 10;

// mlp-medium CKKS params (mlp_medium.mlir module attrs): logN=14, 13 main
// primes, 2 aux primes, scale 2^26.
cheddar::Parameter<word> MakeParam() {
  std::vector<word> main_primes = {
      536903681ULL, 67043329ULL, 67239937ULL, 66813953ULL, 67502081ULL,
      66551809ULL,  67731457ULL, 66420737ULL, 68190209ULL, 65929217ULL,
      68485121ULL,  68681729ULL, 68976641ULL};
  std::vector<word> aux_primes = {536641537ULL, 537133057ULL};
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= 13; ++i) level_config.emplace_back(i, 0);
  return cheddar::Parameter<word>(
      /*logN=*/14, /*scale=*/static_cast<double>(1ULL << 26), kDefaultEncLevel,
      level_config, main_primes, aux_primes);
}

template <typename T>
void ReadInto(std::ifstream& f, T* dst, size_t n) {
  f.read(reinterpret_cast<char*>(dst), n * sizeof(T));
}

// Build a StripedMatrix (centered idx -> diagonal) from diagonal-packed weights;
// used here only to enumerate a LinearTransform's required rotation keys.
cheddar::StripedMatrix MakeStriped(const double w[][kSlots],
                                   const std::vector<int>& idx) {
  cheddar::StripedMatrix m(kSlots, kSlots);
  for (size_t d = 0; d < idx.size(); ++d)
    m[idx[d]] = std::vector<Complex>(w[d], w[d] + kSlots);
  return m;
}

}  // namespace

TEST(CheddarMlpMediumE2E, GpuRun) {
  const std::string base = "tests/Examples/cheddar/mlp_medium/data/";
  static double fc1_w[128][8192], fc2_w[128][8192], fc3_w[137][8192];
  static double fc1_b[8192], fc2_b[8192], fc3_b[8192];
  std::array<double, kSlots> input{};
  auto load = [&](const std::string& name, double* dst, size_t n) {
    std::ifstream f(base + name, std::ios::binary);
    ASSERT_TRUE(f.good()) << "cannot open " << base << name;
    ReadInto(f, dst, n);
    ASSERT_TRUE(f.good()) << name;
  };
  load("fc1_weights.bin", &fc1_w[0][0], 128 * 8192);
  load("fc1_bias.bin", fc1_b, 8192);
  load("fc2_weights.bin", &fc2_w[0][0], 128 * 8192);
  load("fc2_bias.bin", fc2_b, 8192);
  load("fc3_weights.bin", &fc3_w[0][0], 137 * 8192);
  load("fc3_bias.bin", fc3_b, 8192);
  load("input.bin", input.data(), 8192);

  std::array<double, kNumClasses> ref{};
  load("expected_logits.bin", ref.data(), kNumClasses);

  // Diagonal indices: fc1,fc2 use 0..127; fc3 uses 0..127 plus 8183..8191
  // (== -9..-1), matching mlp_medium.mlir.
  std::vector<int> idx128(128);
  for (int i = 0; i < 128; ++i) idx128[i] = i;
  std::vector<int> idx_fc3 = idx128;
  for (int i = 8183; i <= 8191; ++i) idx_fc3.push_back(i);

  auto param = MakeParam();
  auto ctx = cheddar::Context<word>::Create(param);
  auto ui = std::make_unique<UI>(ctx);

  // Rotation keys: the three LinearTransforms' BSGS fan-out + the rotate-and-sum
  // reductions. The (bs, gs) mirror mlp_medium.mlir's explicit attrs (fc1/fc2
  // dense 16/8; fc3 wrap-around -> pure diagonal 8192/1).
  cheddar::EvkRequest req;
  auto add_lt = [&](const cheddar::StripedMatrix& m, int level, int bs, int gs) {
    cheddar::LinearTransform<word> lt(ctx, m, level,
                                      ctx->param_.GetScale(level), bs, gs);
    lt.AddRequiredRotations(req);
  };
  add_lt(MakeStriped(fc1_w, idx128), /*level=*/9, /*bs=*/16, /*gs=*/8);
  add_lt(MakeStriped(fc2_w, idx128), /*level=*/5, /*bs=*/16, /*gs=*/8);
  add_lt(MakeStriped(fc3_w, idx_fc3), /*level=*/1, /*bs=*/8192, /*gs=*/1);
  std::set<int> rots;
  for (const auto& kv : req) rots.insert(kv.first);
  for (int d : {128, 256, 512, 1024, 2048, 4096}) rots.insert(d);
  for (int r : rots) ui->PrepareRotationKey(r, kInputLevel);

  const Evk& evk = ui->GetMultiplicationKey();
  const EvkMap& evk_map = ui->GetEvkMap();

  // Encrypt the packed input at the model's input level.
  std::vector<Complex> msg(kSlots);
  for (int i = 0; i < kSlots; ++i) msg[i] = Complex(input[i], 0.0);
  Pt in_pt;
  ctx->encoder_.Encode(in_pt, kInputLevel, ctx->param_.GetScale(kInputLevel),
                       msg);
  Ct in_ct, out_ct;
  ui->Encrypt(in_ct, in_pt);

  auto t0 = std::chrono::high_resolution_clock::now();
  mlpmedium(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map, in_ct, fc1_w,
            fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, out_ct);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::printf("mlp-medium inference: %.1f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count());

  Pt out_pt;
  ui->Decrypt(out_pt, out_ct);
  std::vector<Complex> dec;
  ctx->encoder_.Decode(dec, out_pt);

  double max_abs = 0;
  for (int i = 0; i < kNumClasses; ++i)
    max_abs = std::max(max_abs, std::abs(dec[i].real() - ref[i]));

  int fhe_argmax = 0, ref_argmax = 0;
  for (int k = 1; k < kNumClasses; ++k) {
    if (dec[k].real() > dec[fhe_argmax].real()) fhe_argmax = k;
    if (ref[k] > ref[ref_argmax]) ref_argmax = k;
  }
  std::printf("max|fhe - pytorch| = %.6g\n", max_abs);
  std::printf("logits[0..9] fhe vs pytorch:\n");
  for (int k = 0; k < kNumClasses; ++k)
    std::printf("  [%d] %9.4f  %9.4f\n", k, dec[k].real(), ref[k]);
  std::printf("predicted class: fhe=%d pytorch=%d\n", fhe_argmax, ref_argmax);

  // FHE evaluates the degree-7 Chebyshev SiLU; the difference from the PyTorch
  // (exact-SiLU) forward is the (deterministic) approximation error plus CKKS
  // noise -- ~0.06 on these logits. Require the logits to stay close and the
  // prediction to agree.
  EXPECT_LT(max_abs, 0.1) << "FHE logits diverge from the PyTorch reference";
  EXPECT_EQ(fhe_argmax, ref_argmax) << "FHE prediction differs from PyTorch";
}
