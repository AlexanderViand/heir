# This build file is necessary to mark this directory as a subpackage for bazel
# to have access to the files.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Launcher for the system-toolchain-compiled GPU e2e tests (see e2e.bzl).
exports_files(["run_e2e_test.sh"])
