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
    name = "dot_product_cheddar_mlir",
    src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    generated_filename = "dot_product_cheddar.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=32768",
        "--scheme-to-cheddar",
    ],
)

heir_translate(
    name = "dot_product_cpp",
    src = ":dot_product_cheddar.mlir",
    generated_filename = "dot_product_lib.cpp",
    pass_flags = ["--emit-cheddar"],
)

heir_translate(
    name = "dot_product_header",
    src = ":dot_product_cheddar.mlir",
    generated_filename = "dot_product_lib.h",
    pass_flags = ["--emit-cheddar-header"],
)

cc_library(
    name = "dot_product_lib",
    srcs = [":dot_product_cpp"],
    hdrs = ["dot_product_lib.h"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

cc_test(
    name = "dot_product_8f_test",
    size = "enormous",
    srcs = ["dot_product_8f_test.cpp"],
    tags = ["notap"],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":dot_product_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
