load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":CheddarBufferize",
        ":pass_inc_gen",
    ],
)

cc_library(
    name = "CheddarBufferize",
    srcs = ["CheddarBufferize.cpp"],
    hdrs = ["CheddarBufferize.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "BufferizableOpInterfaceImpl",
    srcs = ["BufferizableOpInterfaceImpl.cpp"],
    hdrs = ["BufferizableOpInterfaceImpl.h"],
    deps = [
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:ops_inc_gen",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Cheddar",
    td_file = "Passes.td",
)
