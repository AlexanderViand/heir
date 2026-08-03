load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Same conv as //tests/Examples/lattigo/ckks/conv1d_ncw, but lowered with
# use-lintrans-kernels: the conv's expanded-Toeplitz matvec stays compact as
# orion.linear_transform (lattigo lintrans) instead of unrolling, and
# SecretToCKKS drops the Toeplitz matrix's all-zero diagonals.
heir_lattigo_lib(
    name = "conv1d_ncw_lintrans",
    go_library_name = "conv1dncwlintrans",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        # modulus-switch-after-mul mirrors the medusa lattigo path: the
        # compact lintrans output is only rescaled in the after-mul policy;
        # without it the scale accumulates past the modulus at level 0 and
        # decrypts to noise.
        "--mlir-to-ckks=ciphertext-degree=1024 use-lintrans-kernels=true modulus-switch-after-mul=true",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv1d_ncw.mlir",
)

go_test(
    name = "conv1dncwlintrans_test",
    size = "small",
    srcs = ["conv1d_ncw_lintrans_test.go"],
    embed = [":conv1dncwlintrans"],
)
