# See README.md for setup required to run these tests

load("@heir//tests/data-oblivious/end-to-end:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "sum_tensor_test",
    generated_lib_header = "sum_tensor_lib.h",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=sum_tensor ciphertext-degree=32",
    mlir_src = "sum_tensor.mlir",
    tags = ["notap"],
    test_src = "sum_tensor_test.cpp",
)
