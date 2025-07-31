# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "composite_scaling_test",
    generated_lib_header = "composite_scaling_lib.h",
    heir_opt_flags = [
        "--mlir-to-ckks",
        "--canonicalize",
        "--ckks-to-lwe",
        "--lwe-to-openfhe",
        "--canonicalize",
        "--cse",
        "--openfhe-configure-crypto-context=scaling-technique=COMPOSITESCALINGAUTO scaling-mod-size=98 first-mod-size=100 register-word-size=32",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    tags = ["notap"],
    test_src = "composite_scaling_test.cpp",
)
