#include <array>
#include <cstdint>

#include "core/Context.h"
#include "gtest/gtest.h"

using namespace cheddar;
using word = uint64_t;

void abi_outer(const std::array<Ciphertext<word>, 1>& input,
               std::array<Ciphertext<word>, 1>& output);

TEST(CheddarEmitCAbiTest, CallsGeneratedPayloadBoundaryTwice) {
  std::array<Ciphertext<word>, 1> input;
  std::array<Ciphertext<word>, 1> output;
  abi_outer(input, output);
  abi_outer(input, output);
}
