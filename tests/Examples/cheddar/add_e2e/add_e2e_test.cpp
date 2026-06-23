// End-to-end test for the CHEDDAR emitter (`heir-opt --cheddar-to-emitc
// | heir-translate --mlir-to-cpp`). The compute kernels are emitted from
// `add_e2e.mlir`; this driver does context/UI setup + encode/encrypt
// boundary work directly via the CHEDDAR C++ API, calls the generated
// kernels, and checks the decoded result.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"

using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;

// Generated kernels (linked from add_e2e_lib.cc, produced by heir-translate
// --mlir-to-cpp out of add_e2e.mlir). Destination-passing style: result is
// passed as a trailing reference out-param; caller pre-declares it.
void add_kernel(Context<word>* ctx, const Ciphertext<word>& a,
                const Ciphertext<word>& b, Ciphertext<word>& out);
void sub_kernel(Context<word>* ctx, const Ciphertext<word>& a,
                const Ciphertext<word>& b, Ciphertext<word>& out);
void add_plain_kernel(Context<word>* ctx, const Ciphertext<word>& a,
                      const Plaintext<word>& p, Ciphertext<word>& out);

namespace {

constexpr int kLogN = 13;
constexpr double kScale = static_cast<double>(1ULL << 36);

struct CkksFixture {
  std::shared_ptr<Context<word>> ctx;
  std::unique_ptr<UserInterface<word>> ui;
  int top_level;
};

CkksFixture MakeFixture() {
  std::vector<word> main_primes = {
      36028797017456641ULL, 35184366911489ULL, 35184376545281ULL,
      35184367828993ULL,    35184373989377ULL,
  };
  std::vector<word> aux_primes = {
      1152921504608747521ULL,
      1152921504614055937ULL,
  };
  std::vector<std::pair<int, int>> level_config;
  for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i) {
    level_config.emplace_back(i, 0);
  }
  int max_level = static_cast<int>(main_primes.size()) - 1;
  static Parameter<word> param(kLogN, kScale, max_level, level_config,
                               main_primes, aux_primes);
  CkksFixture f;
  f.ctx = Context<word>::Create(param);
  f.ui = std::make_unique<UserInterface<word>>(f.ctx);
  f.top_level = max_level;
  return f;
}

Ciphertext<word> EncryptVec(const CkksFixture& f,
                            const std::vector<double>& msg) {
  std::vector<Complex> complex_msg(msg.begin(), msg.end());
  Plaintext<word> pt;
  f.ctx->encoder_.Encode(pt, f.top_level, kScale, complex_msg);
  Ciphertext<word> ct;
  f.ui->Encrypt(ct, pt);
  return ct;
}

Plaintext<word> EncodeVec(const CkksFixture& f,
                          const std::vector<double>& msg) {
  std::vector<Complex> complex_msg(msg.begin(), msg.end());
  Plaintext<word> pt;
  f.ctx->encoder_.Encode(pt, f.top_level, kScale, complex_msg);
  return pt;
}

std::vector<double> DecryptVec(const CkksFixture& f, const Ciphertext<word>& ct,
                               size_t n) {
  Plaintext<word> pt;
  f.ui->Decrypt(pt, ct);
  std::vector<Complex> out;
  f.ctx->encoder_.Decode(out, pt);
  std::vector<double> real_out(n);
  for (size_t i = 0; i < n; ++i) real_out[i] = out[i].real();
  return real_out;
}

}  // namespace

TEST(CheddarAddE2E, Add) {
  auto f = MakeFixture();
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {0.5, 1.0, 1.5, 2.0};

  auto ct_a = EncryptVec(f, a);
  auto ct_b = EncryptVec(f, b);
  Ciphertext<word> ct_r;
  add_kernel(f.ctx.get(), ct_a, ct_b, ct_r);
  auto result = DecryptVec(f, ct_r, a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(result[i], a[i] + b[i], 0.01);
  }
}

TEST(CheddarAddE2E, Sub) {
  auto f = MakeFixture();
  std::vector<double> a = {3.0, 5.0, 7.0, 9.0};
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0};

  auto ct_a = EncryptVec(f, a);
  auto ct_b = EncryptVec(f, b);
  Ciphertext<word> ct_r;
  sub_kernel(f.ctx.get(), ct_a, ct_b, ct_r);
  auto result = DecryptVec(f, ct_r, a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(result[i], a[i] - b[i], 0.01);
  }
}

TEST(CheddarAddE2E, AddPlain) {
  auto f = MakeFixture();
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {0.5, 1.0, 1.5, 2.0};

  auto ct_a = EncryptVec(f, a);
  auto pt_b = EncodeVec(f, b);
  Ciphertext<word> ct_r;
  add_plain_kernel(f.ctx.get(), ct_a, pt_b, ct_r);
  auto result = DecryptVec(f, ct_r, a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(result[i], a[i] + b[i], 0.01);
  }
}
