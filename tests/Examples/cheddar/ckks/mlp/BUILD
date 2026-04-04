load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Step 1: Lower MNIST MLIR to cheddar dialect
heir_opt(
    name = "mnist_cheddar_mlir",
    src = "@heir//tests/Examples/common/mnist:mnist.mlir",
    generated_filename = "mnist_cheddar.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024 split-preprocessing=8 ckks-reconcile-policy=canonical-per-level",
        "--scheme-to-cheddar",
    ],
)

# Step 2: Generate C++ from cheddar dialect
heir_translate(
    name = "mnist_cpp",
    src = ":mnist_cheddar.mlir",
    generated_filename = "mnist_lib.cpp",
    pass_flags = [
        "--emit-cheddar",
    ],
)

heir_translate(
    name = "mnist_header",
    src = ":mnist_cheddar.mlir",
    generated_filename = "mnist_lib.h",
    pass_flags = [
        "--emit-cheddar-header",
    ],
)

cc_library(
    name = "mnist_lib",
    srcs = [":mnist_cpp"],
    hdrs = [":mnist_header"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
)

# Step 3: End-to-end test
cc_test(
    name = "mnist_test",
    size = "enormous",
    srcs = ["mnist_test.cpp"],
    data = [
        "@cheddar//:all_srcs",  # For parameter JSON files
    ],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":mnist_lib",
        "@cheddar",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
        "@googletest//:gtest_main",
    ],
)
