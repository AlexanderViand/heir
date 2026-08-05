load("@heir//bazel/cheddar:e2e.bzl", "cheddar_e2e_test", "cheddar_system_cc_binary")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Cheap no-GPU helper to dump CHEDDAR's exact per-level GetScale for the
# mlp-medium parameter set (used to bake mlp_medium.mlir's bias-encode scales).
cheddar_system_cc_binary(
    name = "scale_dump",
    srcs = ["scale_dump.cpp"],
)

heir_opt(
    name = "mlp_medium_emitc",
    src = "mlp_medium.mlir",
    generated_filename = "mlp_medium_emitc.mlir",
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
    name = "mlp_medium_cpp_raw",
    src = ":mlp_medium_emitc.mlir",
    generated_filename = "mlp_medium_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "mlp_medium_lib_src",
    srcs = [":mlp_medium_raw.cc"],
    outs = ["mlp_medium_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/mlp_medium/BUILD.
#include <complex>
#include <cstdint>
#include "tests/Examples/cheddar/cheddar_emit_prelude.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :mlp_medium_raw.cc) >> $@
""",
)

cheddar_e2e_test(
    name = "mlp_medium_test",
    srcs = [
        "mlp_medium_test.cpp",
        ":mlp_medium_lib_src",
    ],
    hdrs = [
        "@heir//tests/Examples/cheddar:cheddar_emit_prelude.h",
        "@heir//tests/Examples/cheddar:e2e_check.h",
    ],
    data = glob(["data/*.bin"]),
)
