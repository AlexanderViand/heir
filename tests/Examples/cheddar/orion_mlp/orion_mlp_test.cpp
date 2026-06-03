// End-to-end GPU run of the Orion MLP via the cheddar backend.
//
// The Orion MLP (784->128->128->10, Quad activations) from Orion, with its
// BatchNorm1d layers folded into the preceding Linear. Runs the WHOLE lowered
// module -- `orion_mlp__encrypt__arg0` / `orion_mlp` (compute, inlined) /
// `orion_mlp__decrypt__result0` -- on REAL folded weights and real MNIST
// images, checking the decrypted classification against the plaintext forward
// pass computed here.
//
// The IR's encode scales are baked to CHEDDAR's exact canonical
// `GetScale(level)^k` by bake_scales.py (the emitter emits them verbatim);
// unlike the relu mnist model, the Quad activation needs no further scale
// hand-edits. See HACKS.md #5.
//
// Weights come from a build-time torch dump of the checked-in orion_mlp.pth
// (orion_mlp_weights.bin, BN already folded); images/labels are read straight
// from the checked-in t10k ubyte files; the plaintext reference is computed
// here.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"

using word = uint64_t;
using Complex = std::complex<double>;
using Ct = cheddar::Ciphertext<word>;
using Evk = cheddar::EvaluationKey<word>;
using UI = cheddar::UserInterface<word>;

// Generated entry points (see tests/Examples/cheddar/orion_mlp/BUILD).
// NOTE: signatures confirmed against the lowered orion_mlp.mlir.
void orion_mlp__encrypt__arg0(cheddar::Context<word>* ctx,
                              const cheddar::Encoder<word>& encoder, UI* ui,
                              const Evk& evk, float img[1][784], UI* ui2,
                              std::array<Ct, 1>& out);
void orion_mlp(cheddar::Context<word>* ctx,
               const cheddar::Encoder<word>& encoder, UI* ui, const Evk& evk,
               const std::array<Ct, 1>& in, float w_fc1[128][784],
               float b_fc1[128], float w_fc2[128][128], float b_fc2[128],
               float w_fc3[10][128], float b_fc3[10], std::array<Ct, 1>& out);
void orion_mlp__decrypt__result0(cheddar::Context<word>* ctx,
                                 const cheddar::Encoder<word>& encoder, UI* ui,
                                 const Evk& evk, const std::array<Ct, 1>& in,
                                 UI* ui2, float* logits);

namespace {

// ===========================================================================
// IR-DERIVED PARAMETERS -- from the orion_mlp.mlir module attributes
// (cheddar.logN=14, logDefaultScale=45, cheddar.Q / cheddar.P; max_level =
// #Q-1 = 5) and the distinct hrot distances.
// ===========================================================================
constexpr int kMaxLevel = 5;

const std::vector<int>& RotationDistances() {
  static const std::vector<int> kD = {1,  2,  3,  4,  5,   6,   7,   8,   9,
                                      10, 11, 12, 16, 24,  32,  36,  48,  60,
                                      64, 72, 84, 96, 108, 120, 128, 256, 512};
  return kD;
}

cheddar::Parameter<word> MakeParam() {
  std::vector<word> main_primes = {
      36028797017456641ULL, 35184373006337ULL, 35184370941953ULL,
      35184372744193ULL,    35184371138561ULL, 35184372121601ULL,
  };
  std::vector<word> aux_primes = {
      1152921504607338497ULL,
      1152921504608747521ULL,
  };
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i)
    level_config.emplace_back(i, 0);
  return cheddar::Parameter<word>(
      /*logN=*/14, /*scale=*/static_cast<double>(1ULL << 45), kMaxLevel,
      level_config, main_primes, aux_primes);
}
// ===========================================================================

constexpr int kNumImages = 3;
constexpr double kMean = 0.1307, kStd = 0.3081;  // orion train_mlp Normalize

template <typename T>
void ReadInto(std::ifstream& f, T* dst, size_t n) {
  f.read(reinterpret_cast<char*>(dst), n * sizeof(T));
}

// Plaintext forward pass (the "plaintext world" reference): a 784->128 Quad
// 128->128 Quad 128->10 MLP. PyTorch Linear stores weight [out, in],
// y = x @ W^T + b; Quad(x) = x*x.
std::array<double, 10> PlaintextForward(const float (&w1)[128][784],
                                        const float (&b1)[128],
                                        const float (&w2)[128][128],
                                        const float (&b2)[128],
                                        const float (&w3)[10][128],
                                        const float (&b3)[10],
                                        const std::array<float, 784>& x) {
  std::array<double, 128> h1{};
  for (int o = 0; o < 128; ++o) {
    double acc = b1[o];
    for (int i = 0; i < 784; ++i) acc += static_cast<double>(w1[o][i]) * x[i];
    h1[o] = acc * acc;  // Quad
  }
  std::array<double, 128> h2{};
  for (int o = 0; o < 128; ++o) {
    double acc = b2[o];
    for (int i = 0; i < 128; ++i) acc += static_cast<double>(w2[o][i]) * h1[i];
    h2[o] = acc * acc;  // Quad
  }
  std::array<double, 10> y{};
  for (int o = 0; o < 10; ++o) {
    double acc = b3[o];
    for (int i = 0; i < 128; ++i) acc += static_cast<double>(w3[o][i]) * h2[i];
    y[o] = acc;
  }
  return y;
}

}  // namespace

TEST(CheddarOrionMlpE2E, GpuRun) {
  // Folded weights: build-time torch dump of orion_mlp.pth.
  static float w_fc1[128][784];
  static float b_fc1[128];
  static float w_fc2[128][128];
  static float b_fc2[128];
  static float w_fc3[10][128];
  static float b_fc3[10];
  {
    const std::string wpath =
        "tests/Examples/cheddar/orion_mlp/orion_mlp_weights.bin";
    std::ifstream wf(wpath, std::ios::binary);
    ASSERT_TRUE(wf.good()) << "cannot open " << wpath;
    ReadInto(wf, &w_fc1[0][0], 128 * 784);
    ReadInto(wf, b_fc1, 128);
    ReadInto(wf, &w_fc2[0][0], 128 * 128);
    ReadInto(wf, b_fc2, 128);
    ReadInto(wf, &w_fc3[0][0], 10 * 128);
    ReadInto(wf, b_fc3, 10);
    ASSERT_TRUE(wf.good());
  }

  // Images + labels: read straight from the checked-in t10k ubyte files.
  std::vector<std::array<float, 784>> images(kNumImages);
  std::vector<int> labels(kNumImages);
  {
    const std::string ip =
        "tests/Examples/common/mnist/data/t10k-images-idx3-ubyte";
    const std::string lp =
        "tests/Examples/common/mnist/data/t10k-labels-idx1-ubyte";
    std::ifstream imf(ip, std::ios::binary), lbf(lp, std::ios::binary);
    ASSERT_TRUE(imf.good()) << ip;
    ASSERT_TRUE(lbf.good()) << lp;
    imf.ignore(16);  // magic + dims
    lbf.ignore(8);
    for (int n = 0; n < kNumImages; ++n) {
      unsigned char px[784];
      ReadInto(imf, px, 784);
      for (int j = 0; j < 784; ++j)
        images[n][j] = static_cast<float>((px[j] / 255.0 - kMean) / kStd);
      unsigned char lab;
      ReadInto(lbf, &lab, 1);
      labels[n] = lab;
    }
    ASSERT_TRUE(imf.good() && lbf.good());
  }

  // Plaintext reference logits.
  std::vector<std::array<double, 10>> refs(kNumImages);
  for (int n = 0; n < kNumImages; ++n)
    refs[n] =
        PlaintextForward(w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, images[n]);

  auto param = MakeParam();
  auto ctx = cheddar::Context<word>::Create(param);
  auto ui = std::make_unique<UI>(ctx);
  for (int d : RotationDistances()) ui->PrepareRotationKey(d, kMaxLevel);
  const Evk& evk = ui->GetMultiplicationKey();

  int correct = 0;
  for (int i = 0; i < kNumImages; ++i) {
    float img[1][784];
    for (int j = 0; j < 784; ++j) img[0][j] = images[i][j];

    std::array<Ct, 1> in, out;
    orion_mlp__encrypt__arg0(ctx.get(), ctx->encoder_, ui.get(), evk, img,
                             ui.get(), in);
    orion_mlp(ctx.get(), ctx->encoder_, ui.get(), evk, in, w_fc1, b_fc1, w_fc2,
              b_fc2, w_fc3, b_fc3, out);
    float logits[10] = {0};
    orion_mlp__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk, out,
                                ui.get(), logits);

    int dec_argmax = 0, ref_argmax = 0;
    double max_abs = 0;
    for (int k = 0; k < 10; ++k) {
      if (logits[k] > logits[dec_argmax]) dec_argmax = k;
      if (refs[i][k] > refs[i][ref_argmax]) ref_argmax = k;
      max_abs = std::max(max_abs, std::abs(logits[k] - refs[i][k]));
    }
    if (dec_argmax == labels[i]) ++correct;

    std::printf("img %d: label=%d ref_argmax=%d dec_argmax=%d max|d|=%.4f\n", i,
                labels[i], ref_argmax, dec_argmax, max_abs);
    for (int k = 0; k < 10; ++k)
      std::printf("    [%d] fhe=%9.4f ref=%9.4f\n", k, logits[k], refs[i][k]);

    EXPECT_EQ(dec_argmax, ref_argmax) << "image " << i << " (FHE vs plaintext)";
  }
  std::printf("correct argmax: %d/%d\n", correct, kNumImages);
}
