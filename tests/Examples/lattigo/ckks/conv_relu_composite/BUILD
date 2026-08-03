load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Padded lintrans conv feeding a composite-sign ReLU — the TCN block prefix
# in miniature, with the same medusa flags.
heir_lattigo_lib(
    name = "conv_relu_composite",
    go_library_name = "convrelucomposite",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 use-lintrans-kernels=true modulus-switch-after-mul=true preserve-poly-eval=true use-composite-relu=true ckks-bootstrap-waterline=25",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv_relu_composite.mlir",
)

go_test(
    name = "convrelucomposite_test",
    size = "large",
    srcs = ["conv_relu_composite_test.go"],
    embed = [":convrelucomposite"],
)
