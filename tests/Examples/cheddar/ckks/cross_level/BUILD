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
    name = "cross_level_cheddar_mlir",
    src = "cross_level.mlir",
    generated_filename = "cross_level_cheddar.mlir",
    pass_flags = [
        "--mlir-to-ckks=ciphertext-degree=4 modulus-switch-before-first-mul=true first-mod-bits=59 scaling-mod-bits=45 ckks-reconcile-policy=default-scale-schedule",
        "--scheme-to-cheddar",
    ],
)

heir_translate(
    name = "cross_level_cpp",
    src = ":cross_level_cheddar.mlir",
    generated_filename = "cross_level_lib.cpp",
    pass_flags = ["--emit-cheddar"],
)

heir_translate(
    name = "cross_level_header",
    src = ":cross_level_cheddar.mlir",
    generated_filename = "cross_level_lib.h",
    pass_flags = ["--emit-cheddar-header"],
)

cc_library(
    name = "cross_level_lib",
    srcs = [":cross_level_cpp"],
    hdrs = ["cross_level_lib.h"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

cc_test(
    name = "cross_level_test",
    size = "enormous",
    srcs = ["cross_level_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":cross_level_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
