load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# TCN-block-shaped conv (explicit pad, multi-channel, k=3, bias accumulator)
# lowered with the same flags medusa's pytorch-heir-lattigo path uses; the
# lintrans kernel + zero-diagonal compaction must reproduce the plaintext
# conv exactly.
heir_lattigo_lib(
    name = "conv1d_ncw_padded_lintrans",
    go_library_name = "conv1dncwpaddedlintrans",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 use-lintrans-kernels=true modulus-switch-after-mul=true",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv1d_ncw_padded.mlir",
)

go_test(
    name = "conv1dncwpaddedlintrans_test",
    size = "small",
    srcs = ["conv1d_ncw_padded_lintrans_test.go"],
    embed = [":conv1dncwpaddedlintrans"],
)
