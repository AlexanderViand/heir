load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Two chained composite ReLUs under the bootstrap waterline: the sign chains
# exceed the level budget, forcing a mid-circuit bootstrap — the minimal
# bootstrap + composite interaction (the single-relu variant fits the budget
# with no bootstrap and passes).
heir_lattigo_lib(
    name = "relu2_composite",
    go_library_name = "relu2composite",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 use-lintrans-kernels=true modulus-switch-after-mul=true preserve-poly-eval=true use-composite-relu=true ckks-bootstrap-waterline=25",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:relu2_composite.mlir",
)

go_test(
    name = "relu2composite_test",
    size = "medium",
    srcs = ["relu2_composite_test.go"],
    embed = [":relu2composite"],
)
