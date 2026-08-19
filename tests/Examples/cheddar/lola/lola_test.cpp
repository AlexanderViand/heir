#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "extension/LinearTransform.h"

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using Evk = cheddar::EvaluationKey<word>;
using EvkMap = cheddar::EvkMap<word>;
using LinearTransform = cheddar::LinearTransform<word>;
using Pt = cheddar::Plaintext<word>;
using UI = cheddar::UserInterface<word>;

void lola__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                     std::unique_ptr<UI>& ui);
void lola__encrypt__arg0(cheddar::Context<word>* ctx,
                         const cheddar::Encoder<word>& encoder, const Evk& evk,
                         float* input, UI* ui, std::array<Ct, 1>& out);
void lola__preprocessing(
    cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
    std::array<std::shared_ptr<LinearTransform>, 3>& transforms,
    std::array<Pt, 3>& plaintexts);
void lola__preprocessed(
    cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder, UI* ui,
    const Evk& evk, const EvkMap& evk_map, const std::array<Ct, 1>& input,
    const std::array<std::shared_ptr<LinearTransform>, 3>& transforms,
    const std::array<Pt, 3>& plaintexts, std::array<Ct, 1>& out);
void lola__decrypt__result0(cheddar::Context<word>* ctx,
                            const cheddar::Encoder<word>& encoder,
                            const Evk& evk, const std::array<Ct, 1>& input,
                            UI* ui, float* out);

namespace {

uint32_t readBigEndianUint32(std::ifstream& input) {
  std::array<unsigned char, 4> bytes;
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) throw std::runtime_error("truncated MNIST header");
  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         static_cast<uint32_t>(bytes[3]);
}

std::array<float, 28 * 28> loadFirstMnistImage(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open MNIST image file: " + path);
  if (readBigEndianUint32(input) != 2051)
    throw std::runtime_error("invalid MNIST image magic");
  (void)readBigEndianUint32(input);  // image count
  if (readBigEndianUint32(input) != 28 || readBigEndianUint32(input) != 28)
    throw std::runtime_error("invalid MNIST image shape");

  std::array<unsigned char, 28 * 28> pixels;
  input.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
  if (!input) throw std::runtime_error("truncated MNIST image");
  std::array<float, 28 * 28> result;
  for (size_t i = 0; i < pixels.size(); ++i) {
    result[i] = (static_cast<float>(pixels[i]) / 255.0F - 0.1307F) / 0.3081F;
  }
  return result;
}

}  // namespace

TEST(CheddarLoLaE2E, MatchesPlaintextMnistLogits) {
  constexpr std::array<double, 10> kExpected = {
      -8.300922393798828,  -5.588466644287109, -3.6352932453155518,
      -7.802109241485596,  -3.84030818939209,  -5.856731414794922,
      -5.982525825500488,  11.231959342956543, -3.0315234661102295,
      -3.8313705921173096,
  };
  constexpr double kTolerance = 0.1;
  constexpr int kExpectedClass = 7;

  auto image = loadFirstMnistImage(
      "tests/Examples/common/mnist/data/t10k-images-idx3-ubyte");
  float input[1][1][28][28];
  std::copy(image.begin(), image.end(), &input[0][0][0][0]);

  std::shared_ptr<cheddar::Context<word>> ctx;
  std::unique_ptr<UI> ui;
  lola__configure(ctx, ui);
  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(ui, nullptr);
  const Evk& evk = ui->GetMultiplicationKey();
  const EvkMap& evk_map = ui->GetEvkMap();

  std::array<Ct, 1> encrypted;
  lola__encrypt__arg0(ctx.get(), ctx->encoder_, evk, &input[0][0][0][0],
                      ui.get(), encrypted);

  std::array<Pt, 3> plaintexts;
  std::array<std::shared_ptr<LinearTransform>, 3> transforms;
  auto preprocessing_start = std::chrono::steady_clock::now();
  lola__preprocessing(ctx.get(), ctx->encoder_, transforms, plaintexts);
  std::cerr << "LoLA preprocessing took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             preprocessing_start)
                   .count()
            << " seconds\n";
  ASSERT_NE(transforms[0], nullptr);
  ASSERT_NE(transforms[1], nullptr);
  ASSERT_NE(transforms[2], nullptr);

  std::array<Ct, 1> evaluated;
  auto evaluation_start = std::chrono::steady_clock::now();
  lola__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map,
                     encrypted, transforms, plaintexts, evaluated);
  std::cerr << "LoLA evaluation took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             evaluation_start)
                   .count()
            << " seconds\n";
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted[0].GetNP()), 5);
  EXPECT_EQ(ctx->param_.NPToLevel(evaluated[0].GetNP()), 0);

  float actual[1][10];
  lola__decrypt__result0(ctx.get(), ctx->encoder_, evk, evaluated, ui.get(),
                         &actual[0][0]);
  int predicted_class = 0;
  for (size_t i = 0; i < 10; ++i) {
    ASSERT_TRUE(std::isfinite(actual[0][i])) << "logit " << i;
    EXPECT_NEAR(actual[0][i], kExpected[i], kTolerance) << "logit " << i;
    if (actual[0][i] > actual[0][predicted_class]) predicted_class = i;
  }
  EXPECT_EQ(predicted_class, kExpectedClass);
}
