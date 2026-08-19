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
    name = "abi_emitc",
    src = "abi.mlir",
    generated_filename = "abi_emitc.mlir",
    pass_flags = [
        "--convert-to-emitc=filter-dialects=cheddar",
        "--cheddar-emitc-boundary",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "abi_cpp_raw",
    src = ":abi_emitc.mlir",
    generated_filename = "abi_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "abi_lib_src",
    srcs = [":abi_raw.cc"],
    outs = ["abi_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/emitc_abi/BUILD.
#include <array>
#include <cstdint>
#include "core/Context.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :abi_raw.cc) >> $@
""",
)

cc_library(
    name = "abi_lib",
    srcs = [":abi_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "abi_test",
    srcs = ["abi_test.cpp"],
    linkstatic = False,
    target_compatible_with = requires_cheddar(),
    deps = [
        ":abi_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
