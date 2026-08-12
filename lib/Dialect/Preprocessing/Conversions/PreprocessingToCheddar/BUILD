load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PreprocessingToCheddar",
    srcs = ["PreprocessingToCheddar.cpp"],
    hdrs = ["PreprocessingToCheddar.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/PreprocessingStorageLayoutAnalysis",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Preprocessing/Conversions:Util",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "PreprocessingToCheddar.h.inc",
    pass_name = "PreprocessingToCheddar",
    td_file = "PreprocessingToCheddar.td",
)
