load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Composite-sign ReLU (x * step(x/B), three chained Chebyshev evals) through
# the same flags medusa's pytorch-heir-lattigo path uses (minus the bootstrap
# waterline: the chain fits the generated modulus chain without bootstraps).
# First runtime coverage of lattigo.ckks.chebyshev.
heir_lattigo_lib(
    name = "relu_composite",
    go_library_name = "relucomposite",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 use-lintrans-kernels=true modulus-switch-after-mul=true preserve-poly-eval=true use-composite-relu=true ckks-bootstrap-waterline=25",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:relu_composite.mlir",
)

go_test(
    name = "relucomposite_test",
    size = "medium",
    srcs = ["relu_composite_test.go"],
    embed = [":relucomposite"],
)
