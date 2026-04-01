load("@heir//tools:heir-cheddar.bzl", "cheddar_lib")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cheddar_lib(
    name = "smoke_test_lib",
    mlir_src = "smoke_test.mlir",
)

cc_test(
    name = "smoke_test",
    srcs = ["smoke_test_main.cpp"],
    tags = ["notap"],  # Requires GPU, not run in CI
    deps = [
        ":smoke_test_lib",
        "@cheddar",
    ],
)
