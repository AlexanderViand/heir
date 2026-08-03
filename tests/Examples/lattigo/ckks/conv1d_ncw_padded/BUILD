load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Unrolled twin of //tests/Examples/lattigo/ckks/conv1d_ncw_padded_lintrans:
# identical model and flags except use-lintrans-kernels, isolating the
# lintrans lowering as the only variable.
heir_lattigo_lib(
    name = "conv1d_ncw_padded",
    go_library_name = "conv1dncwpadded",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 modulus-switch-after-mul=true",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv1d_ncw_padded.mlir",
)

go_test(
    name = "conv1dncwpadded_test",
    size = "small",
    srcs = ["conv1d_ncw_padded_test.go"],
    embed = [":conv1dncwpadded"],
)
