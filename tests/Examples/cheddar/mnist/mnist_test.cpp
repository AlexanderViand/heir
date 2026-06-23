// End-to-end GPU run of HEIR's MNIST MLP via the cheddar backend.
//
// Runs the WHOLE lowered module -- the generated `mnist__encrypt__arg4` /
// `mnist` (preprocessing + compute, inlined) / `mnist__decrypt__result0`
// functions -- on REAL model weights and real MNIST images, and checks the
// decrypted classification against the plaintext forward pass computed here
// (the ~few-unit logit gap is the relu-polynomial approximation, not FHE
// error).
//
// This harness hardcodes NO scales: the per-op scales live in the lowered IR
// (mnist.mlir) -- weights at CHEDDAR's exact `GetScale(level)` via the
// cheddar-to-emitc bridge, the pre-rescale biases via HACK #7 exact_scale.
//
// Weights come from a build-time torch dump of the checked-in traced_model.pt
// (mnist_weights.bin); images/labels are read straight from the checked-in
// t10k ubyte files; the plaintext reference is the forward pass computed here.

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

// Generated entry points (see tests/Examples/cheddar/mnist/BUILD).
void mnist__encrypt__arg4(cheddar::Context<word>* ctx,
                          const cheddar::Encoder<word>& encoder, UI* ui,
                          const Evk& evk, float img[1][784], UI* ui2,
                          std::array<Ct, 1>& out);
void mnist(cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
           UI* ui, const Evk& evk, float w_fc1[512][784], float b_fc1[512],
           float w_fc2[10][512], float b_fc2[10], const std::array<Ct, 1>& in,
           std::array<Ct, 1>& out);
void mnist__decrypt__result0(cheddar::Context<word>* ctx,
                             const cheddar::Encoder<word>& encoder, UI* ui,
                             const Evk& evk, const std::array<Ct, 1>& in,
                             UI* ui2, float* logits);

namespace {

constexpr int kMaxLevel = 8;

const std::vector<int>& RotationDistances() {
  static const std::vector<int> kD = {
      1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  32,  46,  64,
      69,  92,  115, 128, 138, 161, 184, 207, 230, 253, 256, 276, 299,
      322, 345, 368, 391, 414, 437, 460, 483, 506, 512};
  return kD;
}

cheddar::Parameter<word> MakeMnistParam() {
  std::vector<word> main_primes = {
      36028797017456641ULL, 35184366911489ULL, 35184376545281ULL,
      35184367828993ULL,    35184373989377ULL, 35184368025601ULL,
      35184373006337ULL,    35184368877569ULL, 35184372744193ULL,
  };
  std::vector<word> aux_primes = {
      1152921504608747521ULL,
      1152921504614055937ULL,
      1152921504615628801ULL,
  };
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i)
    level_config.emplace_back(i, 0);
  return cheddar::Parameter<word>(
      /*logN=*/15, /*scale=*/static_cast<double>(1ULL << 45), kMaxLevel,
      level_config, main_primes, aux_primes);
}

constexpr int kNumImages = 3;
constexpr double kMean = 0.1307, kStd = 0.3081;

template <typename T>
void ReadInto(std::ifstream& f, T* dst, size_t n) {
  f.read(reinterpret_cast<char*>(dst), n * sizeof(T));
}

// Plaintext forward pass (the "plaintext world" reference): a 784->512 ReLU
// 512->10 MLP. PyTorch Linear stores weight as [out, in], y = x @ W^T + b.
std::array<double, 10> PlaintextForward(const float (&w1)[512][784],
                                        const float (&b1)[512],
                                        const float (&w2)[10][512],
                                        const float (&b2)[10],
                                        const std::array<float, 784>& x) {
  std::array<double, 512> h{};
  for (int o = 0; o < 512; ++o) {
    double acc = b1[o];
    for (int i = 0; i < 784; ++i) acc += static_cast<double>(w1[o][i]) * x[i];
    h[o] = acc > 0 ? acc : 0;  // ReLU
  }
  std::array<double, 10> y{};
  for (int o = 0; o < 10; ++o) {
    double acc = b2[o];
    for (int i = 0; i < 512; ++i) acc += static_cast<double>(w2[o][i]) * h[i];
    y[o] = acc;
  }
  return y;
}

}  // namespace

TEST(CheddarMnistFullE2E, GpuRun) {
  // Weights: build-time torch dump of traced_model.pt.
  static float w_fc1[512][784];
  static float b_fc1[512];
  static float w_fc2[10][512];
  static float b_fc2[10];
  {
    const std::string wpath = "tests/Examples/cheddar/mnist/mnist_weights.bin";
    std::ifstream wf(wpath, std::ios::binary);
    ASSERT_TRUE(wf.good()) << "cannot open " << wpath;
    ReadInto(wf, &w_fc1[0][0], 512 * 784);
    ReadInto(wf, b_fc1, 512);
    ReadInto(wf, &w_fc2[0][0], 10 * 512);
    ReadInto(wf, b_fc2, 10);
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

  // Plaintext reference logits (true-ReLU forward).
  std::vector<std::array<double, 10>> refs(kNumImages);
  for (int n = 0; n < kNumImages; ++n)
    refs[n] = PlaintextForward(w_fc1, b_fc1, w_fc2, b_fc2, images[n]);

  const int num_images = kNumImages;
  auto param = MakeMnistParam();
  auto ctx = cheddar::Context<word>::Create(param);
  auto ui = std::make_unique<UI>(ctx);
  for (int d : RotationDistances()) ui->PrepareRotationKey(d, kMaxLevel);
  const Evk& evk = ui->GetMultiplicationKey();

  int correct = 0;
  for (int i = 0; i < num_images; ++i) {
    float img[1][784];
    for (int j = 0; j < 784; ++j) img[0][j] = images[i][j];

    std::array<Ct, 1> in, out;
    mnist__encrypt__arg4(ctx.get(), ctx->encoder_, ui.get(), evk, img, ui.get(),
                         in);
    mnist(ctx.get(), ctx->encoder_, ui.get(), evk, w_fc1, b_fc1, w_fc2, b_fc2,
          in, out);
    float logits[10] = {0};
    mnist__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk, out,
                            ui.get(), logits);

    int dec_argmax = 0, ref_argmax = 0;
    double max_abs = 0;
    for (int k = 0; k < 10; ++k) {
      if (logits[k] > logits[dec_argmax]) dec_argmax = k;
      if (refs[i][k] > refs[i][ref_argmax]) ref_argmax = k;
      max_abs = std::max(max_abs, std::abs(logits[k] - refs[i][k]));
    }
    if (dec_argmax == labels[i]) ++correct;

    std::printf("img %d: label=%d ref_argmax=%d dec_argmax=%d max|d|=%.3f\n", i,
                labels[i], ref_argmax, dec_argmax, max_abs);
    for (int k = 0; k < 10; ++k)
      std::printf("    [%d] fhe=%9.4f ref=%9.4f\n", k, logits[k], refs[i][k]);

    EXPECT_EQ(dec_argmax, labels[i]) << "image " << i;
  }
  std::printf("correct argmax: %d/%d\n", correct, num_images);
}
