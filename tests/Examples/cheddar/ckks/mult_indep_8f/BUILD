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
    name = "mult_indep_cheddar_mlir",
    src = "@heir//tests/Examples/common:mult_indep_8f.mlir",
    generated_filename = "mult_indep_cheddar.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 ckks-reconcile-policy=default-scale-schedule",
        "--scheme-to-cheddar",
    ],
)

heir_translate(
    name = "mult_indep_cpp",
    src = ":mult_indep_cheddar.mlir",
    generated_filename = "mult_indep_lib.cpp",
    pass_flags = ["--emit-cheddar"],
)

heir_translate(
    name = "mult_indep_header",
    src = ":mult_indep_cheddar.mlir",
    generated_filename = "mult_indep_lib.h",
    pass_flags = ["--emit-cheddar-header"],
)

cc_library(
    name = "mult_indep_lib",
    srcs = [":mult_indep_cpp"],
    hdrs = ["mult_indep_lib.h"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

cc_test(
    name = "mult_indep_8f_test",
    size = "enormous",
    srcs = ["mult_indep_8f_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":mult_indep_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
