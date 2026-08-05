load("@heir//bazel/cheddar:e2e.bzl", "cheddar_e2e_test")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end: lower the WHOLE common/mnist/mnist.mlir (client functions
# included) -- encrypt / mnist (split-preprocessing joint entry) / decrypt --
# and run it on real weights + images, checking the decrypted classification
# against a plaintext reference computed in the harness.
heir_opt(
    name = "mnist_emitc",
    # Cheddar-level IR (already past --scheme-to-cheddar) with baked canonical
    # scales + the relu scale-align hand-edits; lower it the rest
    # of the way to emitc.
    src = "mnist.mlir",
    generated_filename = "mnist_emitc.mlir",
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
    name = "mnist_cpp_raw",
    src = ":mnist_emitc.mlir",
    generated_filename = "mnist_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "mnist_lib_src",
    srcs = [":mnist_raw.cc"],
    outs = ["mnist_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/mnist/BUILD.
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
cat $(location :mnist_raw.cc) >> $@
""",
)

# Extract the trained weights from the checked-in traced_model.pt at build
# time (the C++ harness can't read a .pt). Nothing derived is committed.
py_binary(
    name = "dump_weights",
    srcs = ["dump_weights.py"],
    deps = ["@heir_pip_deps//torch"],
)

genrule(
    name = "mnist_weights",
    srcs = ["@heir//tests/Examples/common/mnist/data:traced_model.pt"],
    outs = ["mnist_weights.bin"],
    cmd = "$(location :dump_weights) $(location @heir//tests/Examples/common/mnist/data:traced_model.pt) $@",
    tools = [":dump_weights"],
)

cheddar_e2e_test(
    name = "mnist_test",
    srcs = [
        "mnist_test.cpp",
        ":mnist_lib_src",
    ],
    hdrs = ["@heir//tests/Examples/cheddar:e2e_check.h"],
    data = [
        ":mnist_weights.bin",
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
    ],
)
