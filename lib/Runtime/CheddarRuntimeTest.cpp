#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <utility>

#include "lib/Runtime/CheddarRuntime.h"

namespace heir {
namespace {

using NestedArray = std::array<std::array<std::array<float, 4>, 3>, 2>;
static_assert(
    std::is_same_v<decltype(data(std::declval<NestedArray&>())), float*>);

TEST(CheddarRuntimeTest, NestedArrayDataIsFlat) {
  NestedArray value{};
  EXPECT_EQ(data(value), &value[0][0][0]);
}

}  // namespace
}  // namespace heir
