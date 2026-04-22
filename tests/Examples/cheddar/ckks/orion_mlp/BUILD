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
    name = "mlp_cheddar_mlir",
    src = "@heir//tests/Examples/orion/mlp:mlp.mlir",
    generated_filename = "mlp_cheddar.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 ckks-scale-policy=precise ckks-reconcile-policy=default-scale-schedule",
        "--scheme-to-cheddar",
    ],
)

heir_translate(
    name = "mlp_cpp",
    src = ":mlp_cheddar.mlir",
    generated_filename = "mlp_lib.cpp",
    pass_flags = ["--emit-cheddar"],
)

heir_translate(
    name = "mlp_header",
    src = ":mlp_cheddar.mlir",
    generated_filename = "mlp_lib.h",
    pass_flags = ["--emit-cheddar-header"],
)

cc_library(
    name = "mlp_lib",
    srcs = [":mlp_cpp"],
    hdrs = [":mlp_header"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

cc_test(
    name = "mlp_test",
    size = "enormous",
    srcs = ["mlp_test.cpp"],
    data = ["@cheddar//:all_srcs"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":mlp_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
