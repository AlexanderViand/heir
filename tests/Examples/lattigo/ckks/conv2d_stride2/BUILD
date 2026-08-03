load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Stride-2 gap-packed conv2d with broadcast bias and a zero-padded output
# channel — the LoLA conv1 in miniature, value-asserted against a
# hand-computed reference.
heir_lattigo_lib(
    name = "conv2d_stride2",
    go_library_name = "conv2dstride2",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=64 use-lintrans-kernels=true modulus-switch-after-mul=true enable-split-preprocessing=true",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv2d_stride2.mlir",
)

go_test(
    name = "conv2dstride2_test",
    size = "large",
    srcs = ["conv2d_stride2_test.go"],
    embed = [":conv2dstride2"],
)
