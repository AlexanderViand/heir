load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BufferizableOpInterfaceImpl",
    srcs = ["BufferizableOpInterfaceImpl.cpp"],
    hdrs = ["BufferizableOpInterfaceImpl.h"],
    deps = [
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ConfigureCryptoContext",
    srcs = ["ConfigureCryptoContext.cpp"],
    hdrs = ["ConfigureCryptoContext.h"],
    deps = [
        ":configure_crypto_context_inc_gen",
        "@heir//lib/Analysis/RotationAnalysis",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Utils:TransformUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "FuseOps",
    srcs = ["FuseOps.cpp"],
    hdrs = ["FuseOps.h"],
    deps = [
        ":fuse_ops_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "FreeIntermediates",
    srcs = ["FreeIntermediates.cpp"],
    hdrs = ["FreeIntermediates.h"],
    deps = [
        ":free_intermediates_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "PrepareLinearTransforms",
    srcs = ["PrepareLinearTransforms.cpp"],
    hdrs = ["PrepareLinearTransforms.h"],
    deps = [
        ":prepare_linear_transforms_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "configure_crypto_context_inc_gen",
    header_filename = "ConfigureCryptoContext.h.inc",
    pass_name = "CheddarConfigureCryptoContext",
    td_file = "ConfigureCryptoContext.td",
)

add_heir_transforms(
    generated_target_name = "fuse_ops_inc_gen",
    header_filename = "FuseOps.h.inc",
    pass_name = "CheddarFuseOps",
    td_file = "FuseOps.td",
)

add_heir_transforms(
    generated_target_name = "free_intermediates_inc_gen",
    header_filename = "FreeIntermediates.h.inc",
    pass_name = "CheddarFreeIntermediates",
    td_file = "FreeIntermediates.td",
)

add_heir_transforms(
    generated_target_name = "prepare_linear_transforms_inc_gen",
    header_filename = "PrepareLinearTransforms.h.inc",
    pass_name = "CheddarPrepareLinearTransforms",
    td_file = "PrepareLinearTransforms.td",
)
