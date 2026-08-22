load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# The header-only runtime the generated C++ includes as <heir/runtime/...>.
# The wheel ships this tree as heir/include/heir/runtime.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CleartextResource",
    hdrs = ["CleartextResource.h"],
    strip_include_prefix = "/include",
)

cc_library(
    name = "CheddarRuntime",
    hdrs = ["CheddarRuntime.h"],
    strip_include_prefix = "/include",
)

cc_test(
    name = "CheddarRuntimeTest",
    srcs = ["CheddarRuntimeTest.cpp"],
    deps = [
        ":CheddarRuntime",
        "@googletest//:gtest_main",
    ],
)
