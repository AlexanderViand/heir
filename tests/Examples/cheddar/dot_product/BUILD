load("@heir//bazel/cheddar:e2e.bzl", "cheddar_e2e_test")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end: dot product of two secret 1024-vectors. The cheddar-level IR is a
# full module (client encrypt/decrypt helpers + compute), so the harness just
# calls the generated functions; lower it the rest of the way to emitc.
heir_opt(
    name = "dot_product_emitc",
    src = "dot_product.mlir",
    generated_filename = "dot_product_emitc.mlir",
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
    name = "dot_product_cpp_raw",
    src = ":dot_product_emitc.mlir",
    generated_filename = "dot_product_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "dot_product_lib_src",
    srcs = [":dot_product_raw.cc"],
    outs = ["dot_product_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/dot_product/BUILD.
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
cat $(location :dot_product_raw.cc) >> $@
""",
)

cheddar_e2e_test(
    name = "dot_product_test",
    srcs = [
        "dot_product_test.cpp",
        ":dot_product_lib_src",
    ],
    hdrs = ["@heir//tests/Examples/cheddar:e2e_check.h"],
)
