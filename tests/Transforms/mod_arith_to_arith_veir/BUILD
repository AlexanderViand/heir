load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "@heir//tests:test_utilities",
        # An optional, prebuilt veir-opt binary, discovered via the
        # HEIR_VEIR_OPT_PATH environment variable (see MODULE.bazel). When
        # unavailable, this filegroup is empty and the tests are skipped via
        # the `veir` lit feature.
        "@veir//:veir-opt",
    ],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)
