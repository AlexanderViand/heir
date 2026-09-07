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
    name = "pipeline_eval_chebyshev_emitc",
    src = "@heir//tests/Examples/common:chebyshev_cube.mlir",
    externalize_constants = False,
    generated_filename = "pipeline_eval_chebyshev_emitc.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-cheddar=entry-function=chebyshev_cube",
        "--cheddar-to-emitc",
    ],
)

heir_translate(
    name = "pipeline_eval_chebyshev_cpp_raw",
    src = ":pipeline_eval_chebyshev_emitc.mlir",
    generated_filename = "pipeline_eval_chebyshev_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "pipeline_eval_chebyshev_lib_src",
    srcs = [":pipeline_eval_chebyshev_raw.cc"],
    outs = ["pipeline_eval_chebyshev_lib.cc"],
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
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :pipeline_eval_chebyshev_raw.cc) >> $@
""",
)

cc_library(
    name = "pipeline_eval_chebyshev_lib",
    srcs = [":pipeline_eval_chebyshev_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "pipeline_eval_chebyshev_test",
    timeout = "long",
    srcs = ["pipeline_eval_chebyshev_test.cpp"],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":pipeline_eval_chebyshev_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
