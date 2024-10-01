# See README.md for setup required to run these tests

load("@heir//tests/data-oblivious/end-to-end:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "sum_tensor_in_range_test",
    generated_lib_header = "sum_tensor_in_range_lib.h",
    mlir_src = "sum_tensor_in_range.mlir",
    tags = ["notap"],
    test_src = "sum_tensor_in_range_test.cpp",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=sumTensorInRange ciphertext-degree=16",
)

