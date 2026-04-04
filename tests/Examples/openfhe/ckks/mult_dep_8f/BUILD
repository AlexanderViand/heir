# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "mult_dep_8f_test",
    generated_lib_header = "mult_dep_8f_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/common:mult_dep_8f.mlir",
    tags = ["notap"],
    test_src = "mult_dep_8f_test.cpp",
)
