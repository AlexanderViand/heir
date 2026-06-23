load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end: element-wise product of two secret 1024-vectors. The cheddar-level IR is a full module (client
# encrypt/decrypt helpers + compute), so the harness just calls the generated
# functions; lower it the rest of the way to emitc.
heir_opt(
    name = "mult_e2e_emitc",
    src = "mult_e2e.mlir",
    generated_filename = "mult_e2e_emitc.mlir",
    pass_flags = [
        "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map",
        "--inline=inlining-threshold=4294967295",
        "--fold-memref-alias-ops",
        "--canonicalize",
        "--cheddar-to-emitc",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "mult_e2e_cpp_raw",
    src = ":mult_e2e_emitc.mlir",
    generated_filename = "mult_e2e_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "mult_e2e_lib_src",
    srcs = [":mult_e2e_raw.cc"],
    outs = ["mult_e2e_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/mult_e2e/BUILD.
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "UserInterface.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :mult_e2e_raw.cc) >> $@
""",
)

cc_library(
    name = "mult_e2e_lib",
    srcs = [":mult_e2e_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "mult_e2e_test",
    srcs = ["mult_e2e_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":mult_e2e_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
