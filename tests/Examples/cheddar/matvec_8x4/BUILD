load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Drives the kernel through:
#   1. one-shot bufferization, lowering tensor<8x!cheddar.*> to
#      memref<8x!cheddar.*> with in-place reuse where liveness allows.
#   2. canonicalize, which drops the bufferization's degenerate scf.for
#      iter_arg.
#   3. scf -> emitc / arith -> emitc lowering for the loop body.
#   4. cheddar-to-emitc for the dialect ops + memref-of-cheddar shape,
#      which also lifts the function signature to DPS (the matvec result
#      memref becomes a `std::array<Ct, 8>&` out-param).
#   5. reconcile-unrealized-casts to clean up the index<->size_t bridges
#      that SCFToEmitC and the cheddar type converter cooperate on.
heir_opt(
    name = "matvec_8x4_emitc",
    src = "matvec_8x4.mlir",
    generated_filename = "matvec_8x4_emitc.mlir",
    pass_flags = [
        "--one-shot-bufferize=bufferize-function-boundaries=true",
        "--canonicalize",
        "--convert-scf-to-emitc",
        "--convert-arith-to-emitc",
        "--cheddar-to-emitc",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "matvec_8x4_cpp_raw",
    src = ":matvec_8x4_emitc.mlir",
    generated_filename = "matvec_8x4_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "matvec_8x4_lib_src",
    srcs = [":matvec_8x4_raw.cc"],
    outs = ["matvec_8x4_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/matvec_8x4/BUILD.
#include <array>
#include <cstdint>
#include <utility>
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "UserInterface.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :matvec_8x4_raw.cc) >> $@
""",
)

cc_library(
    name = "matvec_8x4_lib",
    srcs = [":matvec_8x4_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "matvec_8x4_test",
    srcs = ["matvec_8x4_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":matvec_8x4_lib",
        "@cheddar",
        "@googletest//:gtest_main",
        "@heir//tests/Examples/cheddar:cheddar_test_fixture",
    ],
)
