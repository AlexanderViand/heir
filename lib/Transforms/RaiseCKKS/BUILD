load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RaiseCKKS",
    srcs = ["RaiseCKKS.cpp"],
    hdrs = ["RaiseCKKS.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Orion/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "RaiseCKKS",
)
