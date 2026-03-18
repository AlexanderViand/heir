load("@heir//tests/Examples/fideslib:test.bzl", "fideslib_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

fideslib_end_to_end_test(
    name = "parallelization_e2e_test",
    generated_lib_header = "parallelization_fideslib_lib.h",
    heir_opt_flags = [],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/openfhe/ckks/parallelization:parallelization.mlir",
    tags = ["notap"],
    test_src = "parallelization_e2e_test.cpp",
)
