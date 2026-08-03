load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Cheap no-GPU helper to dump CHEDDAR's exact per-level GetScale for the
# orion-MLP parameter set (used to bake orion_mlp_lt.mlir's bias-encode scales).
cc_binary(
    name = "scale_dump",
    srcs = ["scale_dump.cpp"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

heir_opt(
    name = "orion_mlp_lt_emitc",
    src = "orion_mlp_lt.mlir",
    generated_filename = "orion_mlp_lt_emitc.mlir",
    pass_flags = [
        "--one-shot-bufferize=bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map",
        "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true",
        "--fold-memref-alias-ops",
        "--canonicalize",
        "--convert-to-emitc",
        "--cheddar-emitc-boundary",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "orion_mlp_lt_cpp_raw",
    src = ":orion_mlp_lt_emitc.mlir",
    generated_filename = "orion_mlp_lt_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "orion_mlp_lt_lib_src",
    srcs = [":orion_mlp_lt_raw.cc"],
    outs = ["orion_mlp_lt_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/orion_mlp_lt/BUILD.
#include <complex>
#include <cstdint>
#include "tests/Examples/cheddar/cheddar_emit_prelude.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :orion_mlp_lt_raw.cc) >> $@
""",
)

cc_library(
    name = "orion_mlp_lt_lib",
    srcs = [":orion_mlp_lt_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@heir//tests/Examples/cheddar:cheddar_emit_prelude",
    ],
)

cc_test(
    name = "orion_mlp_lt_test",
    srcs = ["orion_mlp_lt_test.cpp"],
    data = glob(["data/*.bin"]),
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":orion_mlp_lt_lib",
        "@cheddar",
        "@googletest//:gtest_main",
        "@heir//tests/Examples/cheddar:cheddar_test_fixture",
    ],
)
