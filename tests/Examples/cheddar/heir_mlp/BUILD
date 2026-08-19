load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

heir_opt(
    name = "heir_mlp_emitc",
    src = "@heir//tests/Examples/common/mnist:mnist.mlir",
    externalize_constants = True,
    generated_filename = "heir_mlp_emitc.mlir",
    pass_flags = [
        "--ext-const-threshold=1",
        "--annotate-module=backend=cheddar scheme=ckks",
        "--linalg-canonicalizations",
        "--torch-linalg-to-ckks=min-slot-count=1024 greedy-level-budget=10",
        "--scheme-to-cheddar=entry-function=mnist",
        "--cheddar-to-emitc",
    ],
)

heir_translate(
    name = "heir_mlp_cpp_raw",
    src = ":heir_mlp_emitc.mlir",
    generated_filename = "heir_mlp_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "heir_mlp_lib_src",
    srcs = [":heir_mlp_raw.cc"],
    outs = ["heir_mlp_lib.cc"],
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
#include "extension/EvalPoly.h"
#include "extension/LinearTransform.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :heir_mlp_raw.cc) >> $@
""",
)

cc_library(
    name = "heir_mlp_lib",
    srcs = [":heir_mlp_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "//lib/Runtime:CleartextResource",
        "@cheddar",
    ],
)

cc_test(
    name = "heir_mlp_test",
    timeout = "eternal",
    srcs = ["heir_mlp_test.cpp"],
    data = [
        ":heir_mlp_emitc",
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:traced_model.pt",
    ],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":heir_mlp_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
