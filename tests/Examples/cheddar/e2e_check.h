// Minimal googletest-shaped check macros for the CHEDDAR GPU e2e tests.
//
// These tests are compiled with the system toolchain to match libcheddar.so
// (see bazel/cheddar/e2e.bzl), so they cannot link the hermetically built
// googletest. This header provides the small subset the tests use with the
// same spelling, so a test body reads identically either way: TEST(...)
// expands to a main() that runs the body and exits nonzero if any EXPECT_*
// failed; ASSERT_TRUE exits immediately.

#ifndef TESTS_EXAMPLES_CHEDDAR_E2E_CHECK_H_
#define TESTS_EXAMPLES_CHEDDAR_E2E_CHECK_H_

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace cheddar_e2e {

inline int& failures() {
  static int count = 0;
  return count;
}

// Prints "file:line: check failed: <expr> <streamed message>" on failure and
// supports gtest's `<< "context"` tail. Fatal checks exit in the destructor,
// after the streamed message is complete.
class CheckStream {
 public:
  CheckStream(bool failed, bool fatal, const char* expr, const char* file,
              int line)
      : hasFailed(failed), isFatal(fatal) {
    if (hasFailed) {
      ++failures();
      std::cerr << file << ":" << line << ": check failed: " << expr << " ";
    }
  }
  template <typename T>
  CheckStream& operator<<(const T& value) {
    if (hasFailed) std::cerr << value;
    return *this;
  }
  ~CheckStream() {
    if (hasFailed) {
      std::cerr << std::endl;
      if (isFatal) std::exit(1);
    }
  }

 private:
  bool hasFailed;
  bool isFatal;
};

}  // namespace cheddar_e2e

#define CHEDDAR_E2E_CHECK_(failed, fatal, expr) \
  ::cheddar_e2e::CheckStream(failed, fatal, expr, __FILE__, __LINE__)

#define EXPECT_TRUE(c) CHEDDAR_E2E_CHECK_(!(c), false, #c)
#define ASSERT_TRUE(c) CHEDDAR_E2E_CHECK_(!(c), true, #c)
#define EXPECT_EQ(a, b) CHEDDAR_E2E_CHECK_(!((a) == (b)), false, #a " == " #b)
#define EXPECT_LT(a, b) CHEDDAR_E2E_CHECK_(!((a) < (b)), false, #a " < " #b)
#define EXPECT_NEAR(a, b, tol)                               \
  CHEDDAR_E2E_CHECK_(!(std::abs((a) - (b)) <= (tol)), false, \
                     "|" #a " - " #b "| <= " #tol)

// The e2e binaries hold exactly one test: expand it to main() directly.
#define TEST(suite, name)                          \
  static void cheddar_e2e_test_body();             \
  int main() {                                     \
    cheddar_e2e_test_body();                       \
    return ::cheddar_e2e::failures() == 0 ? 0 : 1; \
  }                                                \
  static void cheddar_e2e_test_body()

#endif  // TESTS_EXAMPLES_CHEDDAR_E2E_CHECK_H_
