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
    name = "matvec_512x784_cheddar_mlir",
    src = "@heir//tests/Examples/common:matvec_512x784.mlir",
    generated_filename = "matvec_512x784_cheddar.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 ckks-reconcile-policy=canonical-per-level",
        "--scheme-to-cheddar",
    ],
)

heir_translate(
    name = "matvec_512x784_cpp",
    src = ":matvec_512x784_cheddar.mlir",
    generated_filename = "matvec_512x784_lib.cpp",
    pass_flags = ["--emit-cheddar"],
)

heir_translate(
    name = "matvec_512x784_header",
    src = ":matvec_512x784_cheddar.mlir",
    generated_filename = "matvec_512x784_lib.h",
    pass_flags = ["--emit-cheddar-header"],
)

cc_library(
    name = "matvec_512x784_lib",
    srcs = [":matvec_512x784_cpp"],
    hdrs = ["matvec_512x784_lib.h"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

cc_test(
    name = "matvec_512x784_test",
    size = "enormous",
    srcs = ["matvec_512x784_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":matvec_512x784_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
