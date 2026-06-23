// End-to-end GPU run of HEIR's MNIST MLP via the cheddar backend.
//
// Runs the WHOLE lowered module -- the generated `mnist__encrypt__arg4` /
// `mnist` (preprocessing + compute, inlined) / `mnist__decrypt__result0`
// functions -- on REAL model weights and real MNIST images.
//
// What this validates, and how it avoids the classic "argmax happens to
// survive" trap: the FHE circuit does NOT compute ReLU -- HEIR's
// --polynomial-approximation replaced @relu (domain [-20,20]) with its DEFAULT
// degree-5 Caratheodory-Fejer Chebyshev approximation, evaluated via
// Paterson-Stockmeyer (the two ciphertext mults that build T2/T4). A degree-5
// fit of ReLU over a domain that wide is a poor *approximation* (several units
// of logit error vs true ReLU), so comparing FHE output to a true-ReLU forward
// conflates approximation error with FHE error and forces a meaningless
// argmax-only check.
//
// Instead we compute TWO plaintext references:
//   * `ForwardCheb` -- the SAME degree-5 Chebyshev poly the circuit evaluates
//     (coefficients dumped from heir-opt). FHE-vs-this is pure CKKS noise, so
//     we assert a TIGHT bound: this is the real fidelity check on the lowering.
//   * `ForwardRelu` -- the true-ReLU forward, reported as the honest
//     end-to-end accuracy signal (and asserted to match the label).
//
// This harness hardcodes NO scales: the per-op scales live in the lowered IR
// (mnist.mlir), baked to CHEDDAR's exact canonical `GetScale(level)^k` (and the
// relu biases' drifted scales) by bake_scales.py; the emitter emits them
// verbatim.
//
// Weights come from a build-time torch dump of the checked-in traced_model.pt
// (mnist_weights.bin); images/labels are read straight from the checked-in
// t10k ubyte files; the plaintext reference is the forward pass computed here.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
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

double TrueRelu(double x) { return x > 0 ? x : 0; }

// The EXACT activation the FHE circuit evaluates: HEIR's default degree-5
// Caratheodory-Fejer Chebyshev approximation of ReLU on [-20,20]. Coefficients
// (T0..T5) dumped from `heir-opt --polynomial-approximation` on the source
// @relu; the domain [-20,20] maps to the Chebyshev domain [-1,1] via t=x/20.
// Outside [-20,20] this (like the circuit) diverges -- intentionally evaluated
// verbatim so the FHE comparison stays apples-to-apples on those slots too.
double ChebReluApprox(double x) {
  static const double c[6] = {6.3393991028159498,     1.000000e+01,
                              4.3075053667521672,     -8.1728713370568291e-16,
                              -1.2656936164678267,    -5.7991334850882538e-16};
  const double t = x / 20.0;
  double tkm1 = 1.0, tk = t, acc = c[0] + c[1] * t;
  for (int k = 2; k < 6; ++k) {
    const double tkp = 2.0 * t * tk - tkm1;
    acc += c[k] * tkp;
    tkm1 = tk;
    tk = tkp;
  }
  return acc;
}

// 784->512->{activation}->10 MLP forward. PyTorch Linear stores weight as
// [out, in], y = x @ W^T + b. `act` selects the hidden activation so the same
// code yields both the circuit-spec reference (Chebyshev) and the true-ReLU
// reference.
std::array<double, 10> Forward(const float (&w1)[512][784],
                               const float (&b1)[512],
                               const float (&w2)[10][512],
                               const float (&b2)[10],
                               const std::array<float, 784>& x,
                               double (*act)(double)) {
  std::array<double, 512> h{};
  for (int o = 0; o < 512; ++o) {
    double acc = b1[o];
    for (int i = 0; i < 784; ++i) acc += static_cast<double>(w1[o][i]) * x[i];
    h[o] = act(acc);
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

  // Two plaintext references per image: the circuit-spec Chebyshev forward
  // (what the FHE actually evaluates) and the true-ReLU forward (accuracy
  // ground truth).
  std::vector<std::array<double, 10>> refs_cheb(kNumImages), refs_relu(kNumImages);
  for (int n = 0; n < kNumImages; ++n) {
    refs_cheb[n] =
        Forward(w_fc1, b_fc1, w_fc2, b_fc2, images[n], ChebReluApprox);
    refs_relu[n] = Forward(w_fc1, b_fc1, w_fc2, b_fc2, images[n], TrueRelu);
  }

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
    auto t0 = std::chrono::high_resolution_clock::now();
    mnist(ctx.get(), ctx->encoder_, ui.get(), evk, w_fc1, b_fc1, w_fc2, b_fc2,
          in, out);
    auto t1 = std::chrono::high_resolution_clock::now();
    float logits[10] = {0};
    mnist__decrypt__result0(ctx.get(), ctx->encoder_, ui.get(), evk, out,
                            ui.get(), logits);

    int dec_argmax = 0, cheb_argmax = 0, relu_argmax = 0;
    double max_abs_cheb = 0, max_abs_relu = 0;
    for (int k = 0; k < 10; ++k) {
      if (logits[k] > logits[dec_argmax]) dec_argmax = k;
      if (refs_cheb[i][k] > refs_cheb[i][cheb_argmax]) cheb_argmax = k;
      if (refs_relu[i][k] > refs_relu[i][relu_argmax]) relu_argmax = k;
      max_abs_cheb = std::max(max_abs_cheb, std::abs(logits[k] - refs_cheb[i][k]));
      max_abs_relu = std::max(max_abs_relu, std::abs(logits[k] - refs_relu[i][k]));
    }
    if (dec_argmax == labels[i]) ++correct;

    std::printf(
        "img %d: label=%d  fhe_argmax=%d cheb_argmax=%d relu_argmax=%d  "
        "max|fhe-cheb|=%.4f (FHE fidelity)  max|fhe-relu|=%.3f (approx gap)  "
        "%.1f ms\n",
        i, labels[i], dec_argmax, cheb_argmax, relu_argmax, max_abs_cheb,
        max_abs_relu,
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    for (int k = 0; k < 10; ++k)
      std::printf("    [%d] fhe=%9.4f cheb=%9.4f relu=%9.4f\n", k, logits[k],
                  refs_cheb[i][k], refs_relu[i][k]);

    // FHE fidelity: the FHE run must reproduce the degree-5 Chebyshev circuit
    // it lowers from, up to CKKS noise. This is the real correctness signal --
    // observed gap is ~1e-4 on logits of magnitude ~20, so 0.01 is a tight
    // bound that still catches any scale/lowering regression (which would show
    // up as O(0.1) or larger), while staying ~400x below the ReLU-approx gap.
    EXPECT_LT(max_abs_cheb, 0.01)
        << "image " << i << ": FHE diverges from the Chebyshev circuit it "
                            "evaluates (lowering/scale bug, not approx error)";
    EXPECT_EQ(dec_argmax, cheb_argmax)
        << "image " << i << ": FHE prediction differs from the plaintext "
                            "Chebyshev circuit";
    // End-to-end accuracy: the (approximate) prediction still matches the label.
    EXPECT_EQ(dec_argmax, labels[i]) << "image " << i;
  }
  std::printf("correct argmax: %d/%d\n", correct, num_images);
}
