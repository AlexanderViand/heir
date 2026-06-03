load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "cheddar_test_fixture",
    hdrs = ["cheddar_test_fixture.h"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

# Shared shims (RunLinearTransform / RunEvalPoly) that cheddar-to-emitc lowers
# cheddar.linear_transform / cheddar.eval_poly into. Single source of truth,
# #include'd by each example's generated kernel instead of being copy-pasted
# into every genrule.
cc_library(
    name = "cheddar_emit_prelude",
    hdrs = ["cheddar_emit_prelude.h"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)
