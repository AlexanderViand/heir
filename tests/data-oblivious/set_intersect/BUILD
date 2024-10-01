# See README.md for setup required to run these tests

load("@heir//tests/data-oblivious/end-to-end:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "set_intersect_test",
    generated_lib_header = "set_intersect_lib.h",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=set_intersect ciphertext-degree=16",
    mlir_src = "set_intersect.mlir",
    tags = ["notap"],
    test_src = "set_intersect_test.cpp",
)
