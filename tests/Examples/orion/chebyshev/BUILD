load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "chebyshev",
    go_library_name = "chebyshev",
    heir_opt_flags = [
        "--annotate-orion",
        "--scheme-to-lattigo",
    ],
    mlir_src = ":chebyshev.mlir",
)

go_test(
    name = "chebyshev_test",
    srcs = ["chebyshev_test.go"],
    embed = [":chebyshev"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)

openfhe_end_to_end_test(
    name = "chebyshev_openfhe_test",
    size = "enormous",
    timeout = "long",
    generated_lib_header = "chebyshev_openfhe_lib.h",
    heir_opt_flags = [
        "--annotate-orion",
        "--populate-scale-ckks=openfhe-scaling-technique=fixed-manual",
        "--scheme-to-openfhe=openfhe-scaling-technique=fixed-manual",
    ],
    mlir_src = ":chebyshev.mlir",
    test_src = "chebyshev_openfhe_test.cpp",
)

openfhe_end_to_end_test(
    name = "chebyshev_openfhe_flexible_auto_test",
    size = "enormous",
    timeout = "long",
    generated_lib_header = "chebyshev_openfhe_flexible_auto_lib.h",
    heir_opt_flags = [
        "--annotate-orion",
        "--populate-scale-ckks=openfhe-scaling-technique=flexible-auto",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto",
    ],
    mlir_src = ":chebyshev.mlir",
    test_src = "chebyshev_openfhe_flexible_auto_test.cpp",
)

openfhe_end_to_end_test(
    name = "chebyshev_openfhe_flexible_auto_ext_test",
    size = "enormous",
    timeout = "long",
    generated_lib_header = "chebyshev_openfhe_flexible_auto_ext_lib.h",
    heir_opt_flags = [
        "--annotate-orion",
        "--populate-scale-ckks=openfhe-scaling-technique=flexible-auto-ext",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto-ext",
    ],
    mlir_src = ":chebyshev.mlir",
    test_src = "chebyshev_openfhe_flexible_auto_ext_test.cpp",
)
