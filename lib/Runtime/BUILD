load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CleartextResource",
    hdrs = ["CleartextResource.h"],
)

cc_library(
    name = "CheddarRuntime",
    hdrs = ["CheddarRuntime.h"],
)

cc_test(
    name = "CheddarRuntimeTest",
    srcs = ["CheddarRuntimeTest.cpp"],
    deps = [
        ":CheddarRuntime",
        "@googletest//:gtest_main",
    ],
)
