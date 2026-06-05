// End-to-end GPU run of the Orion MLP lowered from the Orion-extension IR that
// the orion-to-lattigo path emits (orion.linear_transform + Quad), via the
// Orion -> CHEDDAR patterns.
//
// Unlike orion_mlp_test (HEIR's own --torch-linalg-to-ckks unrolled rotate-and-
// sum matvecs), this lowers three cheddar.linear_transform ops (CHEDDAR's
// BSGS+hoisting LinearTransform extension) fed diagonal-packed weights. The
// model (diagonal-packed FC weights + biases) and the packed input come from
// the orion export (data/*.bin, f64 in 4096 slots); the input is MNIST test
// image 0 (label 7) flattened to 784 values, zero-padded to 4096 slots.
//
// Reference: the PyTorch plaintext forward pass of the same MLP on the same
// input -- an INDEPENDENT ground truth, not a replay of the cheddar op graph.
// Its 10 output logits are checked in as data/expected_logits.bin. They were
// produced offline from the orion MLP's reference PyTorch model (784 -> 128 ->
// 128 -> 10, BatchNorm folded into the linear layers, Quad activations) run on
// MNIST test image 0; argmax is class 7 (the true label). We assert the FHE run
// reproduces these logits in slots 0..9 within CKKS noise.

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
// orion_mlp_lt.mlir / orion_mlp_lt_raw.cc.
void orionmlp(cheddar::Context<word>* ctx,
              const cheddar::Encoder<word>& encoder, UI* ui, const Evk& evk,
              const EvkMap& evk_map, const Ct& in, double fc1_w[128][4096],
              double fc1_b[4096], double fc2_w[128][4096], double fc2_b[4096],
              double fc3_w[137][4096], double fc3_b[4096], Ct& out);

namespace {

constexpr int kSlots = 4096;
constexpr int kMaxLevel = 5;
constexpr int kNumClasses = 10;

// orion-MLP CKKS params (orion_mlp_lt.mlir module attrs): logN=13, 6 main
// primes, 2 aux primes, scale 2^26.
cheddar::Parameter<word> MakeParam() {
  std::vector<word> main_primes = {536903681ULL, 67043329ULL, 66994177ULL,
                                   67239937ULL,  66961409ULL, 66813953ULL};
  std::vector<word> aux_primes = {536952833ULL, 536690689ULL};
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= 6; ++i) level_config.emplace_back(i, 0);
  return cheddar::Parameter<word>(
      /*logN=*/13, /*scale=*/static_cast<double>(1ULL << 26), kMaxLevel,
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

TEST(CheddarOrionMlpLtE2E, GpuRun) {
  const std::string base = "tests/Examples/cheddar/orion_mlp_lt/data/";
  static double fc1_w[128][4096], fc2_w[128][4096], fc3_w[137][4096];
  static double fc1_b[4096], fc2_b[4096], fc3_b[4096];
  std::array<double, kSlots> input{};
  auto load = [&](const std::string& name, double* dst, size_t n) {
    std::ifstream f(base + name, std::ios::binary);
    ASSERT_TRUE(f.good()) << "cannot open " << base << name;
    ReadInto(f, dst, n);
    ASSERT_TRUE(f.good()) << name;
  };
  load("fc1_weights.bin", &fc1_w[0][0], 128 * 4096);
  load("fc1_bias.bin", fc1_b, 4096);
  load("fc2_weights.bin", &fc2_w[0][0], 128 * 4096);
  load("fc2_bias.bin", fc2_b, 4096);
  load("fc3_weights.bin", &fc3_w[0][0], 137 * 4096);
  load("fc3_bias.bin", fc3_b, 4096);
  load("input.bin", input.data(), 4096);

  // PyTorch ground-truth logits (independent of the cheddar op graph).
  std::array<double, kNumClasses> ref{};
  load("expected_logits.bin", ref.data(), kNumClasses);

  // Diagonal indices (centered to match the IR): fc1,fc2 use 0..127; fc3 uses
  // 0..127 plus 4087..4095 (== -9..-1).
  std::vector<int> idx128(128);
  for (int i = 0; i < 128; ++i) idx128[i] = i;
  std::vector<int> idx_fc3 = idx128;
  for (int i = 4087; i <= 4095; ++i) idx_fc3.push_back(i);

  // FHE run.
  auto param = MakeParam();
  auto ctx = cheddar::Context<word>::Create(param);
  auto ui = std::make_unique<UI>(ctx);

  // Rotation keys: the three LinearTransforms' BSGS fan-out + the rotate-and-sum
  // reductions (hrot_add distances). The (bs, gs) here mirror the explicit
  // attributes baked into orion_mlp_lt.mlir's cheddar.linear_transform ops --
  // CHEDDAR derives the exact required rotations from them, so key generation
  // matches what the emitted kernel rotates by.
  cheddar::EvkRequest req;
  auto add_lt = [&](const cheddar::StripedMatrix& m, int level, int bs, int gs) {
    cheddar::LinearTransform<word> lt(ctx, m, level,
                                      ctx->param_.GetScale(level), bs, gs);
    lt.AddRequiredRotations(req);
  };
  add_lt(MakeStriped(fc1_w, idx128), /*level=*/5, /*bs=*/16, /*gs=*/8);
  add_lt(MakeStriped(fc2_w, idx128), /*level=*/3, /*bs=*/16, /*gs=*/8);
  add_lt(MakeStriped(fc3_w, idx_fc3), /*level=*/1, /*bs=*/4096, /*gs=*/1);
  // Prepare every required rotation (the LTs' fan-out + the rotate-and-sum
  // distances) at the top level so a single key serves all the levels the same
  // rotation index is used at (the LTs run at levels 5/3/1).
  std::set<int> rots;
  for (const auto& kv : req) rots.insert(kv.first);
  for (int d : {128, 256, 512, 1024, 2048}) rots.insert(d);
  for (int r : rots) ui->PrepareRotationKey(r, kMaxLevel);

  const Evk& evk = ui->GetMultiplicationKey();
  const EvkMap& evk_map = ui->GetEvkMap();

  // Encrypt the packed input at the top level.
  std::vector<Complex> msg(kSlots);
  for (int i = 0; i < kSlots; ++i) msg[i] = Complex(input[i], 0.0);
  Pt in_pt;
  ctx->encoder_.Encode(in_pt, kMaxLevel, ctx->param_.GetScale(kMaxLevel), msg);
  Ct in_ct, out_ct;
  ui->Encrypt(in_ct, in_pt);

  auto t0 = std::chrono::high_resolution_clock::now();
  orionmlp(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map, in_ct, fc1_w,
           fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, out_ct);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::printf("LT-lowering inference: %.1f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count());

  Pt out_pt;
  ui->Decrypt(out_pt, out_ct);
  std::vector<Complex> dec;
  ctx->encoder_.Decode(dec, out_pt);

  // Compare the decrypted logits (slots 0..9) against the PyTorch reference.
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

  // CKKS noise on these params is ~1e-2; require the FHE logits to track the
  // PyTorch ground truth well within that, and agree on the prediction.
  EXPECT_LT(max_abs, 0.05) << "FHE logits diverge from the PyTorch reference";
  EXPECT_EQ(fhe_argmax, ref_argmax) << "FHE prediction differs from PyTorch";
  EXPECT_EQ(ref_argmax, 7) << "reference prediction should be class 7";
}
