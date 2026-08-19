load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end: generate the real scale-snu BootContext setup, encrypt at level 0,
# refresh with cheddar.boot (-> BootContext::Boot), decrypt, and check both the
# refreshed level and boot(x) ~= x. The MLIR carries CHEDDAR's curated
# bootparam_40_64bit modulus chain because the bootstrap needs substantially
# more depth than an ordinary arithmetic program.
heir_opt(
    name = "bootstrap_emitc",
    src = "bootstrap.mlir",
    generated_filename = "bootstrap_emitc.mlir",
    pass_flags = [
        "--cheddar-configure-crypto-context=entry-function=boot",
        "--cheddar-to-emitc",
    ],
)

heir_translate(
    name = "bootstrap_cpp_raw",
    src = ":bootstrap_emitc.mlir",
    generated_filename = "bootstrap_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "bootstrap_lib_src",
    srcs = [":bootstrap_raw.cc"],
    outs = ["bootstrap_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/bootstrap/BUILD.
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "core/Context.h"
#include "core/Encode.h"
#include "core/EvkRequest.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"
#include "UserInterface.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :bootstrap_raw.cc) >> $@
""",
)

cc_library(
    name = "bootstrap_lib",
    srcs = [":bootstrap_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "bootstrap_test",
    timeout = "long",
    srcs = ["bootstrap_test.cpp"],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":bootstrap_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
