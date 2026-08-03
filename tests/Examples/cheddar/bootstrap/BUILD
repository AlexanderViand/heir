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
# bootparam_40_64bit parameter set (used to bake bootstrap.mlir's encode scale).
cc_binary(
    name = "scale_dump",
    srcs = ["scale_dump.cpp"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

# End-to-end: a real CHEDDAR GPU bootstrap. Encrypt at level 0, refresh with
# cheddar.boot (-> BootContext::Boot), decrypt, and check boot(x) ~= x. The IR is
# a curated cheddar-level module (like the lattigo bootstrap test): HEIR's param
# generation can't produce CHEDDAR's deep bootstrap modulus chain, so the harness
# drives the real bootparam_40_64bit param set and the IR only carries the
# level-0 encode (baked to the exact GetScale(0)) + the boot/encrypt/decrypt ops.
heir_opt(
    name = "bootstrap_emitc",
    src = "bootstrap.mlir",
    generated_filename = "bootstrap_emitc.mlir",
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
    srcs = ["bootstrap_test.cpp"],
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
