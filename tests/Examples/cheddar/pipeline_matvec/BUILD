load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# End-to-end coverage starting from the common linalg matvec input. Unlike the
# Cheddar-level examples, this exercises the complete Scheme-to-Cheddar path as
# well as generated crypto-context setup, EmitC lowering, and the real runtime.
heir_opt(
    name = "pipeline_matvec_emitc",
    src = "@heir//tests/Examples/common:matvec_8x4.mlir",
    externalize_constants = False,
    generated_filename = "pipeline_matvec_emitc.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--linalg-canonicalizations",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-cheddar=entry-function=matvec",
        "--cheddar-to-emitc",
        "--cheddar-emitc-entry-interface=entry-function=matvec",
    ],
)

heir_translate(
    name = "pipeline_matvec_header",
    src = ":pipeline_matvec_emitc.mlir",
    generated_filename = "matvec.h",
    pass_flags = [
        "--mlir-to-cpp",
        "--file-id=header",
    ],
)

heir_translate(
    name = "pipeline_matvec_source",
    src = ":pipeline_matvec_emitc.mlir",
    generated_filename = "matvec.cc",
    pass_flags = [
        "--mlir-to-cpp",
        "--file-id=source",
    ],
)

cc_library(
    name = "pipeline_matvec_lib",
    srcs = [":matvec.cc"],
    hdrs = [":matvec.h"],
    strip_include_prefix = ".",
    target_compatible_with = requires_cheddar(),
    deps = [
        "//lib/Runtime:CheddarRuntime",
        "@cheddar",
    ],
)

cc_test(
    name = "pipeline_matvec_test",
    timeout = "long",
    srcs = ["pipeline_matvec_test.cpp"],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":pipeline_matvec_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
