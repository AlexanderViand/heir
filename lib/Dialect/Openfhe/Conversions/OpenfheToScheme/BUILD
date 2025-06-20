load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OpenfheToScheme",
    srcs = ["OpenfheToScheme.cpp"],
    hdrs = ["OpenfheToScheme.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Utils/RewriteUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

add_heir_transforms(
    header_filename = "OpenfheToScheme.h.inc",
    pass_name = "OpenfheToScheme",
    td_file = "OpenfheToScheme.td",
)
