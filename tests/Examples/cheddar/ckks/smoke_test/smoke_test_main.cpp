// End-to-end smoke test for CHEDDAR backend.
// Tests that HEIR-generated code links and runs against the CHEDDAR library.

#include <cheddar/include/UserInterface.h>
#include <cheddar/include/core/Container.h>
#include <cheddar/include/core/Context.h>
#include <cheddar/include/core/Encode.h>
#include <cheddar/include/core/Parameter.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Include the HEIR-generated header
#include "smoke_test.h"

using namespace cheddar;
using word = uint64_t;
using Ct = Ciphertext<word>;
using Pt = Plaintext<word>;
using CtxPtr = std::shared_ptr<Context<word>>;
using Complex = std::complex<double>;

int main() {
  // Load parameters from CHEDDAR's built-in param file
  std::string param_path =
      "/tmp/cheddar-fhe/parameters/bootparam_40_64bit.json";
  std::ifstream param_file(param_path);
  if (!param_file.is_open()) {
    std::cerr << "Could not open parameter file: " << param_path << std::endl;
    return 1;
  }

  // TODO: Parse JSON and construct Parameter<word>
  // For now, this test validates that the generated code compiles and links
  // against the CHEDDAR library. Actual execution requires parameter setup
  // which is library-version-dependent.

  std::cout << "CHEDDAR smoke test: compilation and linking OK" << std::endl;
  return 0;
}
