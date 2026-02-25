load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@heir//bazel:fideslib/config.bzl", "fideslib_deps", "requires_fideslib")
load("@heir//bazel/openfhe:copts.bzl", "OPENMP_COPTS", "OPENMP_LINKOPTS")

package(default_applicable_licenses = ["@heir//:license"])

cc_test(
    name = "smoke_test",
    srcs = ["smoke_test.cc"],
    copts = OPENMP_COPTS + ["-std=c++20"],
    linkopts = OPENMP_LINKOPTS + [
        "-L/usr/local/cuda/lib64",
        "-lcuda",
        "-lcudart",
    ],
    target_compatible_with = requires_fideslib(),
    deps = fideslib_deps([
        "@openfhe//:core",
        "@openfhe//:pke",
        "@googletest//:gtest_main",
    ]),
)
