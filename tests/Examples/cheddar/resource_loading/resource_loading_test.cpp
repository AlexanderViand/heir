#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>

float load_float_resource();
int64_t load_integer_resource();

namespace {

template <typename T, size_t N>
void writeResource(const char* path, const T (&values)[N]) {
  std::ofstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  stream.write(reinterpret_cast<const char*>(values), sizeof(values));
  ASSERT_TRUE(stream.good());
}

TEST(CheddarResourceLoadingTest, LoadsFloatAndIntegerResources) {
  const char* testTmpdir = std::getenv("TEST_TMPDIR");
  ASSERT_NE(testTmpdir, nullptr);
  ASSERT_EQ(chdir(testTmpdir), 0);

  const float weights[] = {1.25f, 2.5f, 3.75f, 5.0f};
  const int64_t indices[] = {17, 23, 42, 99};
  writeResource("weights.bin", weights);
  writeResource("indices.bin", indices);

  EXPECT_FLOAT_EQ(load_float_resource(), weights[0]);
  EXPECT_EQ(load_integer_resource(), indices[0]);
}

}  // namespace
