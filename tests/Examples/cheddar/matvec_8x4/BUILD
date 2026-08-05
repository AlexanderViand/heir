load("@heir//bazel/cheddar:e2e.bzl", "cheddar_e2e_test")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end: public 8x4 matrix times a secret 4-vector. The cheddar-level IR is a full module (client
# encrypt/decrypt helpers + compute), so the harness just calls the generated
# functions; lower it the rest of the way to emitc.
heir_opt(
    name = "matvec_8x4_emitc",
    src = "matvec_8x4.mlir",
    generated_filename = "matvec_8x4_emitc.mlir",
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
cat $(location :matvec_8x4_raw.cc) >> $@
""",
)

cheddar_e2e_test(
    name = "matvec_8x4_test",
    srcs = [
        "matvec_8x4_test.cpp",
        ":matvec_8x4_lib_src",
    ],
    hdrs = ["@heir//tests/Examples/cheddar:e2e_check.h"],
)
