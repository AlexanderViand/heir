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
    name = "simple_sumf_cheddar_mlir",
    src = "@heir//tests/Examples/common:simple_sumf.mlir",
    generated_filename = "simple_sumf_cheddar.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 ckks-reconcile-policy=default-scale-schedule ckks-scale-policy=precise",
        "--scheme-to-cheddar",
    ],
)

heir_translate(
    name = "simple_sumf_cpp",
    src = ":simple_sumf_cheddar.mlir",
    generated_filename = "simple_sumf_lib.cpp",
    pass_flags = ["--emit-cheddar"],
)

heir_translate(
    name = "simple_sumf_header",
    src = ":simple_sumf_cheddar.mlir",
    generated_filename = "simple_sumf_lib.h",
    pass_flags = ["--emit-cheddar-header"],
)

cc_library(
    name = "simple_sumf_lib",
    srcs = [":simple_sumf_cpp"],
    hdrs = ["simple_sumf_lib.h"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

cc_test(
    name = "simple_sumf_test",
    size = "enormous",
    srcs = ["simple_sumf_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":simple_sumf_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
