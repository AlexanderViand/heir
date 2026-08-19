load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Compile the trained LoLA MNIST CNN from the common high-level linalg input.
# This is deliberately separate from the small kernel smoke tests: its embedded
# dense-resource weights and convolution exercise realistic model scale.
heir_opt(
    name = "lola_emitc",
    src = "@heir//tests/Examples/common:lola.mlir",
    externalize_constants = True,
    generated_filename = "lola_emitc.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--linalg-canonicalizations",
        "--torch-linalg-to-ckks=min-slot-count=2048",
        "--scheme-to-cheddar=entry-function=lola",
        "--cheddar-to-emitc",
    ],
)

heir_translate(
    name = "lola_cpp_raw",
    src = ":lola_emitc.mlir",
    generated_filename = "lola_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "lola_lib_src",
    srcs = [":lola_raw.cc"],
    outs = ["lola_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit.
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "extension/LinearTransform.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :lola_raw.cc) >> $@
""",
)

cc_library(
    name = "lola_lib",
    srcs = [":lola_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "//lib/Runtime:CleartextResource",
        "@cheddar",
    ],
)

cc_test(
    name = "lola_test",
    timeout = "eternal",
    srcs = ["lola_test.cpp"],
    data = [
        ":lola_emitc",
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
    ],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":lola_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
