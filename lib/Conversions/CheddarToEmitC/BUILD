load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# The linear-transform runtime is a checked-in header (reviewable, and
# CI-compiled via the tests/Conversions/CheddarToEmitC/compile build_test on
# emitted kernels containing it); this genrule wraps it into a raw-string
# literal the pass embeds, so generated kernels stay self-contained.
genrule(
    name = "lintrans_runtime_inc",
    srcs = ["CheddarLintransRuntime.h"],
    outs = ["CheddarLintransRuntime.inc"],
    cmd = "(printf 'R\"shim(\\n' && cat $(SRCS) && printf ')shim\"\\n') > $@",
)

# Exported so consumers can also #include the runtime directly (the injected
# copy carries the same include guard).
exports_files(["CheddarLintransRuntime.h"])

cc_library(
    name = "CheddarToEmitC",
    srcs = [
        "CheddarToEmitC.cpp",
        ":lintrans_runtime_inc",
    ],
    hdrs = ["CheddarToEmitC.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ConvertToEmitCInterface",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToEmitC",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "CheddarToEmitC.h.inc",
    pass_name = "CheddarToEmitC",
    td_file = "CheddarToEmitC.td",
)
