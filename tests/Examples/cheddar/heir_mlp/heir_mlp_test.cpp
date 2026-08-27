#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

void mnist__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                      std::unique_ptr<UI>& ui);
void mnist__encrypt__arg4(cheddar::Context<word>* ctx,
                          const cheddar::Encoder<word>& encoder, const Evk& evk,
                          float* input, UI* ui, std::array<Ct, 1>& out);
void mnist__preprocessing(
    cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder,
    float* second_bias, float* second_weight, float* first_bias,
    float* first_weight,
    std::array<std::shared_ptr<LinearTransform>, 2>& transforms,
    std::array<Pt, 3>& plaintexts);
void mnist__preprocessed(
    cheddar::Context<word>* ctx, const cheddar::Encoder<word>& encoder, UI* ui,
    const Evk& evk, const EvkMap& evk_map, float* first_weight,
    float* first_bias, float* second_weight, float* second_bias,
    const std::array<Ct, 1>& input,
    const std::array<std::shared_ptr<LinearTransform>, 2>& transforms,
    const std::array<Pt, 3>& plaintexts, std::array<Ct, 1>& out);
void mnist__decrypt__result0(cheddar::Context<word>* ctx,
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
  (void)readBigEndianUint32(input);
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

uint16_t readLittleEndianUint16(const std::vector<unsigned char>& bytes,
                                size_t offset) {
  if (offset + 2 > bytes.size()) throw std::runtime_error("truncated ZIP");
  return static_cast<uint16_t>(bytes[offset]) |
         (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t readLittleEndianUint32(const std::vector<unsigned char>& bytes,
                                size_t offset) {
  if (offset + 4 > bytes.size()) throw std::runtime_error("truncated ZIP");
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

std::vector<float> loadStoredFloatTensor(const std::string& path,
                                         const std::string& member,
                                         size_t element_count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open checkpoint: " + path);
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());

  constexpr uint32_t kEndOfCentralDirectorySignature = 0x06054b50;
  size_t end = bytes.size();
  while (end >= 4 && readLittleEndianUint32(bytes, end - 4) !=
                         kEndOfCentralDirectorySignature) {
    --end;
  }
  if (end < 4) throw std::runtime_error("missing ZIP central directory");
  size_t eocd = end - 4;
  uint16_t entry_count = readLittleEndianUint16(bytes, eocd + 10);
  size_t entry = readLittleEndianUint32(bytes, eocd + 16);

  constexpr uint32_t kCentralDirectorySignature = 0x02014b50;
  constexpr uint32_t kLocalFileSignature = 0x04034b50;
  for (uint16_t i = 0; i < entry_count; ++i) {
    if (readLittleEndianUint32(bytes, entry) != kCentralDirectorySignature)
      throw std::runtime_error("invalid ZIP central directory");
    uint16_t name_size = readLittleEndianUint16(bytes, entry + 28);
    uint16_t extra_size = readLittleEndianUint16(bytes, entry + 30);
    uint16_t comment_size = readLittleEndianUint16(bytes, entry + 32);
    std::string name(bytes.begin() + entry + 46,
                     bytes.begin() + entry + 46 + name_size);
    if (name == member) {
      if (readLittleEndianUint16(bytes, entry + 10) != 0)
        throw std::runtime_error("checkpoint tensor is compressed");
      uint32_t size = readLittleEndianUint32(bytes, entry + 24);
      if (size != element_count * sizeof(float))
        throw std::runtime_error("checkpoint tensor has unexpected size");
      size_t local = readLittleEndianUint32(bytes, entry + 42);
      if (readLittleEndianUint32(bytes, local) != kLocalFileSignature)
        throw std::runtime_error("invalid ZIP local header");
      size_t data = local + 30 + readLittleEndianUint16(bytes, local + 26) +
                    readLittleEndianUint16(bytes, local + 28);
      std::vector<float> result(element_count);
      for (size_t j = 0; j < element_count; ++j)
        result[j] = std::bit_cast<float>(
            readLittleEndianUint32(bytes, data + j * sizeof(float)));
      return result;
    }
    entry += 46 + name_size + extra_size + comment_size;
  }
  throw std::runtime_error("missing checkpoint tensor: " + member);
}

}  // namespace

TEST(CheddarHeirMlpE2E, MatchesDegreeFivePlaintextCircuit) {
  // Plaintext evaluation of the checked-in checkpoint on MNIST image zero,
  // using the exact degree-5 Chebyshev coefficients emitted for ReLU on
  // [-20, 20]. This is intentionally stronger than a class-only assertion.
  constexpr std::array<double, 10> kExpected = {
      -8.5350291793657274, -13.938522269897303, -6.7867503201587791,
      0.60811017277978696, -17.41903505739177,  -6.3793769709381314,
      -20.849203501714406, 13.854514072845355,  -3.0537463477429165,
      0.18866433784546457,
  };
  constexpr double kTolerance = 0.15;
  constexpr int kExpectedClass = 7;

  auto image = loadFirstMnistImage(
      "tests/Examples/common/mnist/data/t10k-images-idx3-ubyte");
  float input[1][784];
  std::copy(image.begin(), image.end(), &input[0][0]);

  const std::string checkpoint =
      "tests/Examples/common/mnist/data/traced_model.pt";
  std::vector<float> first_weight =
      loadStoredFloatTensor(checkpoint, "traced_model/data/0", 512 * 784);
  std::vector<float> first_bias =
      loadStoredFloatTensor(checkpoint, "traced_model/data/1", 512);
  std::vector<float> second_weight =
      loadStoredFloatTensor(checkpoint, "traced_model/data/2", 10 * 512);
  std::vector<float> second_bias =
      loadStoredFloatTensor(checkpoint, "traced_model/data/3", 10);
  std::shared_ptr<cheddar::Context<word>> ctx;
  std::unique_ptr<UI> ui;
  mnist__configure(ctx, ui);
  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(ui, nullptr);
  const Evk& evk = ui->GetMultiplicationKey();
  const EvkMap& evk_map = ui->GetEvkMap();

  std::array<Ct, 1> encrypted;
  mnist__encrypt__arg4(ctx.get(), ctx->encoder_, evk, &input[0][0], ui.get(),
                       encrypted);

  std::array<Pt, 3> plaintexts;
  std::array<std::shared_ptr<LinearTransform>, 2> transforms;
  auto preprocessing_start = std::chrono::steady_clock::now();
  mnist__preprocessing(ctx.get(), ctx->encoder_, second_bias.data(),
                       second_weight.data(), first_bias.data(),
                       first_weight.data(), transforms, plaintexts);
  std::cerr << "HeirMLP preprocessing took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             preprocessing_start)
                   .count()
            << " seconds\n";
  ASSERT_NE(transforms[0], nullptr);
  ASSERT_NE(transforms[1], nullptr);

  std::array<Ct, 1> evaluated;
  auto evaluation_start = std::chrono::steady_clock::now();
  mnist__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map,
                      first_weight.data(), first_bias.data(),
                      second_weight.data(), second_bias.data(), encrypted,
                      transforms, plaintexts, evaluated);
  std::cerr << "HeirMLP evaluation took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             evaluation_start)
                   .count()
            << " seconds\n";
  EXPECT_EQ(ctx->param_.NPToLevel(encrypted[0].GetNP()), 6);
  EXPECT_EQ(ctx->param_.NPToLevel(evaluated[0].GetNP()), 0);

  std::array<Ct, 1> evaluated_again;
  mnist__preprocessed(ctx.get(), ctx->encoder_, ui.get(), evk, evk_map,
                      first_weight.data(), first_bias.data(),
                      second_weight.data(), second_bias.data(), encrypted,
                      transforms, plaintexts, evaluated_again);

  float actual[1][10];
  mnist__decrypt__result0(ctx.get(), ctx->encoder_, evk, evaluated, ui.get(),
                          &actual[0][0]);
  int predicted_class = 0;
  for (size_t i = 0; i < 10; ++i) {
    ASSERT_TRUE(std::isfinite(actual[0][i])) << "logit " << i;
    EXPECT_NEAR(actual[0][i], kExpected[i], kTolerance) << "logit " << i;
    if (actual[0][i] > actual[0][predicted_class]) predicted_class = i;
  }
  EXPECT_EQ(predicted_class, kExpectedClass);

  float repeated[1][10];
  mnist__decrypt__result0(ctx.get(), ctx->encoder_, evk, evaluated_again,
                          ui.get(), &repeated[0][0]);
  for (size_t i = 0; i < 10; ++i) {
    ASSERT_TRUE(std::isfinite(repeated[0][i])) << "repeated logit " << i;
    EXPECT_NEAR(repeated[0][i], actual[0][i], kTolerance)
        << "repeated logit " << i;
  }
}
