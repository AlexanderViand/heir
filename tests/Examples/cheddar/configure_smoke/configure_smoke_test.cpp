#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"

using word = uint64_t;
using Ct = cheddar::Ciphertext<word>;
using Pt = cheddar::Plaintext<word>;
using UI = cheddar::UserInterface<word>;

void rotate__configure(std::shared_ptr<cheddar::Context<word>>& ctx,
                       std::unique_ptr<UI>& ui);
void rotate(cheddar::Context<word>* ctx, UI* ui, const Ct& input, Ct& output);

TEST(CheddarConfigureSmoke, GeneratedSetupAndRotationRun) {
  std::shared_ptr<cheddar::Context<word>> ctx;
  std::unique_ptr<UI> ui;
  rotate__configure(ctx, ui);

  ASSERT_NE(ctx, nullptr);
  ASSERT_NE(ui, nullptr);
  EXPECT_NO_THROW(ui->GetRotationKey(2));
  EXPECT_NO_THROW(ui->GetRotationKey(7));

  constexpr int kSlots = 4096;
  std::vector<cheddar::Complex> message;
  message.reserve(kSlots);
  for (int i = 0; i < kSlots; ++i)
    message.emplace_back(static_cast<double>((i * 17) % 101) / 101.0,
                         static_cast<double>((i * 29) % 89) / 178.0);
  Pt plaintext;
  ctx->encoder_.Encode(plaintext, /*level=*/2,
                       ctx->param_.GetScale(/*level=*/2), message);
  Ct input;
  ui->Encrypt(input, plaintext);

  Ct output;
  rotate(ctx.get(), ui.get(), input, output);

  // Compare the emitted call sequence against the real scale-snu operations.
  // A nonconstant complex message makes rotation direction and distance
  // observable, unlike the previous all-ones input.
  Ct rotated_reference;
  ctx->HRot(rotated_reference, input, ui->GetRotationKey(7), 7);
  Ct reference;
  ctx->HRotAdd(reference, rotated_reference, input, ui->GetRotationKey(2), 2);

  Pt decrypted;
  ui->Decrypt(decrypted, output);
  std::vector<cheddar::Complex> decoded;
  ctx->encoder_.Decode(decoded, decrypted);
  ASSERT_EQ(decoded.size(), kSlots);

  Pt decrypted_reference;
  ui->Decrypt(decrypted_reference, reference);
  std::vector<cheddar::Complex> decoded_reference;
  ctx->encoder_.Decode(decoded_reference, decrypted_reference);
  ASSERT_EQ(decoded_reference.size(), kSlots);

  double max_abs_error = 0.0;
  for (int i = 0; i < kSlots; ++i) {
    ASSERT_TRUE(std::isfinite(decoded[i].real()));
    ASSERT_TRUE(std::isfinite(decoded[i].imag()));
    double error = std::abs(decoded[i] - decoded_reference[i]);
    ASSERT_TRUE(std::isfinite(error)) << "non-finite error at slot " << i;
    max_abs_error = std::max(max_abs_error, error);
  }
  EXPECT_LT(max_abs_error, 1e-2);
}
