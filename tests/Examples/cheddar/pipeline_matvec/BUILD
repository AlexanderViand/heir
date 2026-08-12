load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end coverage starting from the common linalg matvec input. Unlike the
# Cheddar-level examples, this exercises the complete Scheme-to-Cheddar path as
# well as generated crypto-context setup, EmitC lowering, and the real runtime.
heir_opt(
    name = "pipeline_matvec_emitc",
    src = "@heir//tests/Examples/common:matvec_8x4.mlir",
    externalize_constants = False,
    generated_filename = "pipeline_matvec_emitc.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--linalg-canonicalizations",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-cheddar=entry-function=matvec",
        "--arith-expand",
        "--lower-affine",
        "--eliminate-empty-tensors",
        "--cheddar-bufferize",
        "--convert-linalg-to-loops",
        "--inline",
        "--fold-memref-alias-ops",
        "--cse",
        "--canonicalize",
        "--drop-equivalent-buffer-results",
        "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true",
        "--canonicalize",
        "--ownership-based-buffer-deallocation",
        "--canonicalize",
        "--buffer-deallocation-simplification",
        "--bufferization-lower-deallocations",
        "--convert-to-emitc=filter-dialects=cheddar,arith,scf,memref",
        "--cheddar-emitc-boundary",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "pipeline_matvec_cpp_raw",
    src = ":pipeline_matvec_emitc.mlir",
    generated_filename = "pipeline_matvec_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "pipeline_matvec_lib_src",
    srcs = [":pipeline_matvec_raw.cc"],
    outs = ["pipeline_matvec_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/pipeline_matvec/BUILD.
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
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :pipeline_matvec_raw.cc) >> $@
""",
)

cc_library(
    name = "pipeline_matvec_lib",
    srcs = [":pipeline_matvec_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "pipeline_matvec_test",
    timeout = "long",
    srcs = ["pipeline_matvec_test.cpp"],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":pipeline_matvec_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
