# A library build target that encapsulates the HEIR-generated code.
load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "matmul_test",
    generated_lib_header = "matmul_lib.h",
    heir_opt_flags = [
        "--mlir-to-ckks=ciphertext-degree=16384",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/rotom/matmul:matmul.mlir",
    tags = ["notap"],
    test_src = "matmul_test.cpp",
)
