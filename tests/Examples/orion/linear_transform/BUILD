load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "linear_transform",
    go_library_name = "lineartransform",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30",
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
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=fixed-manual",
        "--scheme-to-openfhe=openfhe-scaling-technique=fixed-manual",
    ],
    mlir_src = ":linear_transform.mlir",
    test_src = "linear_transform_openfhe_test.cpp",
)

openfhe_end_to_end_test(
    name = "linear_transform_openfhe_flexible_auto_test",
    generated_lib_header = "linear_transform_openfhe_flexible_auto_lib.h",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=flexible-auto",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto",
    ],
    mlir_src = ":linear_transform.mlir",
    test_src = "linear_transform_openfhe_flexible_auto_test.cpp",
)

openfhe_end_to_end_test(
    name = "linear_transform_openfhe_flexible_auto_ext_test",
    generated_lib_header = "linear_transform_openfhe_flexible_auto_ext_lib.h",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=flexible-auto-ext",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto-ext",
    ],
    mlir_src = ":linear_transform.mlir",
    test_src = "linear_transform_openfhe_flexible_auto_ext_test.cpp",
)
