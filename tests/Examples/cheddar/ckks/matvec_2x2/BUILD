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
    name = "matvec_2x2_cheddar_mlir",
    src = "@heir//tests/Examples/common:matvec_2x2.mlir",
    generated_filename = "matvec_2x2_cheddar.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-cheddar",
    ],
)

heir_translate(
    name = "matvec_2x2_cpp",
    src = ":matvec_2x2_cheddar.mlir",
    generated_filename = "matvec_2x2_lib.cpp",
    pass_flags = ["--emit-cheddar"],
)

heir_translate(
    name = "matvec_2x2_header",
    src = ":matvec_2x2_cheddar.mlir",
    generated_filename = "matvec_2x2_lib.h",
    pass_flags = ["--emit-cheddar-header"],
)

cc_library(
    name = "matvec_2x2_lib",
    srcs = [":matvec_2x2_cpp"],
    hdrs = ["matvec_2x2_lib.h"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

cc_test(
    name = "matvec_2x2_test",
    size = "enormous",
    srcs = ["matvec_2x2_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":matvec_2x2_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
