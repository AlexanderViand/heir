load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Step 1: Lower the Orion MLP MLIR to cheddar dialect
heir_opt(
    name = "mlp_cheddar_mlir",
    src = "@heir//tests/Examples/orion/mlp:mlp.mlir",
    generated_filename = "mlp_cheddar.mlir",
    pass_flags = [
        "--lwe-to-cheddar",
        "--cheddar-fuse-ops",
        "--cheddar-configure-crypto-context",
    ],
)

# Step 2: Generate C++ from cheddar dialect
heir_translate(
    name = "mlp_cpp",
    src = ":mlp_cheddar.mlir",
    generated_filename = "mlp_lib.cpp",
    pass_flags = [
        "--emit-cheddar",
    ],
)

heir_translate(
    name = "mlp_header",
    src = ":mlp_cheddar.mlir",
    generated_filename = "mlp_lib.h",
    pass_flags = [
        "--emit-cheddar-header",
    ],
)

cc_library(
    name = "mlp_lib",
    srcs = [":mlp_cpp"],
    hdrs = [":mlp_header"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcuda",
        "-lcudart",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
    ],
)

# Step 3: End-to-end test
cc_test(
    name = "mlp_test",
    srcs = ["mlp_test.cpp"],
    tags = ["notap"],  # Requires GPU, not run in CI
    target_compatible_with = requires_cheddar(),
    deps = [
        ":mlp_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
