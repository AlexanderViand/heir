load("@rules_cc//cc:cc_library.bzl", "cc_library")

# The header-only runtime the generated C++ includes as <heir/runtime/...>.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CheddarRuntime",
    hdrs = ["CheddarRuntime.h"],
    strip_include_prefix = "/include",
)
