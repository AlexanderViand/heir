# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "simple_sumf",
    go_library_name = "simplesumf",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=2048 first-mod-bits=0",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:simple_sumf.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.
go_test(
    name = "simplesumf_test",
    srcs = ["simple_sumf_test.go"],
    embed = [":simplesumf"],
)
