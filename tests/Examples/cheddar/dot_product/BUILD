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
    name = "dot_product_emitc",
    src = "dot_product.mlir",
    generated_filename = "dot_product_emitc.mlir",
    pass_flags = ["--cheddar-to-emitc"],
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
#include <cstdint>
#include <utility>
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "UserInterface.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :dot_product_raw.cc) >> $@
""",
)

cc_library(
    name = "dot_product_lib",
    srcs = [":dot_product_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "dot_product_test",
    srcs = ["dot_product_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":dot_product_lib",
        "@cheddar",
        "@googletest//:gtest_main",
        "@heir//tests/Examples/cheddar:cheddar_test_fixture",
    ],
)
