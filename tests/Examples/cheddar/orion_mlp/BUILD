load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end: the Orion MLP (784->128->128->10, Quad activations, BatchNorm
# folded into the preceding Linear) lowered the rest of the way to emitc and
# run on real folded weights + real MNIST images, checking the decrypted
# classification against a plaintext reference computed in the harness. Much
# simpler than the mnist relu model: Quad is a single ciphertext square, so the
# IR needs only the mechanical canonical-scale bake (bake_scales.py), no relu
# scale-align hand-edits.
heir_opt(
    name = "orion_mlp_emitc",
    # Cheddar-level IR (already past --scheme-to-cheddar); lower it to emitc.
    src = "orion_mlp.mlir",
    generated_filename = "orion_mlp_emitc.mlir",
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
    name = "orion_mlp_cpp_raw",
    src = ":orion_mlp_emitc.mlir",
    generated_filename = "orion_mlp_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "orion_mlp_lib_src",
    srcs = [":orion_mlp_raw.cc"],
    outs = ["orion_mlp_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/orion_mlp/BUILD.
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
cat $(location :orion_mlp_raw.cc) >> $@
""",
)

# Extract + BN-fold the trained weights from the checked-in orion_mlp.pth at
# build time (the C++ harness can't read a .pth). Nothing derived is committed.
py_binary(
    name = "dump_weights",
    srcs = ["dump_weights.py"],
    deps = ["@heir_pip_deps//torch"],
)

genrule(
    name = "orion_mlp_weights",
    srcs = ["@heir//tests/Examples/common/mnist/data:orion_mlp.pth"],
    outs = ["orion_mlp_weights.bin"],
    cmd = "$(location :dump_weights) $(location @heir//tests/Examples/common/mnist/data:orion_mlp.pth) $@",
    tools = [":dump_weights"],
)

cc_library(
    name = "orion_mlp_lib",
    srcs = [":orion_mlp_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "orion_mlp_test",
    srcs = ["orion_mlp_test.cpp"],
    data = [
        ":orion_mlp_weights.bin",
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
    ],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":orion_mlp_lib",
        "@cheddar",
        "@googletest//:gtest_main",
        "@heir//tests/Examples/cheddar:cheddar_test_fixture",
    ],
)
