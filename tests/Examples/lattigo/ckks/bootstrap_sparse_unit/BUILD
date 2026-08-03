load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Pure-lattigo (no HEIR-generated code) check of the exact bootstrap
# configuration HEIR emits for sparse ciphertexts: residual LogN=16 /
# q0~2^55 / scale 2^45, bootstrapping literal {LogN:16, LogSlots:13}, input
# encoded at LogDimensions Cols=13. Isolates whether the value corruption
# observed in the composite-ReLU e2e tests comes from lattigo's sparse
# bootstrap contract or from HEIR's emitted sequence.
go_test(
    name = "bootstrap_sparse_unit_test",
    size = "large",
    srcs = ["bootstrap_sparse_unit_test.go"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/bootstrapping",
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//ring",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
        "@com_github_tuneinsight_lattigo_v6//utils",
    ],
)
