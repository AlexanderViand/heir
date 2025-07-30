load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "OpenfheDialect.cpp",
        "OpenfheOps.cpp",
    ],
    hdrs = [
        "OpenfheAttributes.h",
        "OpenfheDialect.h",
        "OpenfheEnums.h",
        "OpenfheOps.h",
        "OpenfheTypes.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "OpenfheDialect.td",
        "OpenfheEnums.td",
        "OpenfheOps.td",
        "OpenfheTypes.td",
    ],
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Openfhe",
    kind = "dialect",
    td_file = "OpenfheDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Openfhe",
    kind = "type",
    td_file = "OpenfheTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Openfhe",
    kind = "op",
    td_file = "OpenfheOps.td",
    deps = [
        ":enums_inc_gen",
        ":td_files",
        "@heir//lib/Dialect/LWE/IR:td_files",
    ],
)

add_heir_dialect_library(
    name = "enums_inc_gen",
    dialect = "Openfhe",
    kind = "enum",
    td_file = "OpenfheEnums.td",
    deps = [
        ":td_files",
    ],
)

# Custom because we need to pass -attrdefs-dialect=openfhe
gentbl_cc_library(
    name = "attributes_inc_gen",
    tbl_outs = {
        "OpenfheAttributes.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=openfhe",
        ],
        "OpenfheAttributes.cpp.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect=openfhe",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OpenfheEnums.td",
    deps = [":td_files"],
)
