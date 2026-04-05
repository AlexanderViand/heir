load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "linear_transform",
    go_library_name = "lineartransform",
    heir_opt_flags = [
        "--annotate-orion=linear-transform-impl-style=opaque",
        "--scheme-to-lattigo",
    ],
    mlir_src = ":linear_transform.mlir",
)

go_test(
    name = "linear_transform_test",
    srcs = ["linear_transform_test.go"],
    embed = [":lineartransform"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)

openfhe_end_to_end_test(
    name = "linear_transform_openfhe_test",
    generated_lib_header = "linear_transform_openfhe_lib.h",
    heir_opt_flags = [
        "--annotate-orion=linear-transform-impl-style=opaque",
        "--scheme-to-openfhe=openfhe-scaling-technique=fixed-manual",
    ],
    mlir_src = ":linear_transform.mlir",
    test_src = "linear_transform_openfhe_test.cpp",
)

openfhe_end_to_end_test(
    name = "linear_transform_openfhe_fixed_auto_test",
    generated_lib_header = "linear_transform_openfhe_fixed_auto_lib.h",
    heir_opt_flags = [
        "--annotate-orion=linear-transform-impl-style=opaque",
        "--scheme-to-openfhe=openfhe-scaling-technique=fixed-auto",
    ],
    mlir_src = ":linear_transform.mlir",
    test_src = "linear_transform_openfhe_fixed_auto_test.cpp",
)

openfhe_end_to_end_test(
    name = "linear_transform_openfhe_flexible_auto_test",
    generated_lib_header = "linear_transform_openfhe_flexible_auto_lib.h",
    heir_opt_flags = [
        "--annotate-orion=linear-transform-impl-style=opaque",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto",
    ],
    mlir_src = ":linear_transform.mlir",
    test_src = "linear_transform_openfhe_flexible_auto_test.cpp",
)

openfhe_end_to_end_test(
    name = "linear_transform_openfhe_flexible_auto_ext_test",
    generated_lib_header = "linear_transform_openfhe_flexible_auto_ext_lib.h",
    heir_opt_flags = [
        "--annotate-orion=linear-transform-impl-style=opaque",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto-ext",
    ],
    mlir_src = ":linear_transform.mlir",
    test_src = "linear_transform_openfhe_flexible_auto_ext_test.cpp",
)
