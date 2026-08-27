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
    name = "pipeline_mlp_emitc",
    src = "@heir//tests/Examples/common:tiny_mlp.mlir",
    externalize_constants = False,
    generated_filename = "pipeline_mlp_emitc.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--linalg-canonicalizations",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-cheddar=entry-function=tiny_mlp",
        "--cheddar-to-emitc",
    ],
)

heir_translate(
    name = "pipeline_mlp_cpp_raw",
    src = ":pipeline_mlp_emitc.mlir",
    generated_filename = "pipeline_mlp_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "pipeline_mlp_lib_src",
    srcs = [":pipeline_mlp_raw.cc"],
    outs = ["pipeline_mlp_lib.cc"],
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
cat $(location :pipeline_mlp_raw.cc) >> $@
""",
)

cc_library(
    name = "pipeline_mlp_lib",
    srcs = [":pipeline_mlp_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "pipeline_mlp_test",
    timeout = "long",
    srcs = ["pipeline_mlp_test.cpp"],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":pipeline_mlp_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
