load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# Debug-instrumented twin of //tests/Examples/lattigo/ckks/relu2_composite:
# a __heir_debug decrypt is inserted after every secret op, tracing values
# through prescale -> sign chain -> x*step -> bootstrap -> second relu to
# localize where the mid-circuit bootstrap corrupts the composite chain.
heir_lattigo_lib(
    name = "relu2_composite_debug",
    go_library_name = "relu2compositedbg",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 use-lintrans-kernels=true modulus-switch-after-mul=true preserve-poly-eval=true use-composite-relu=true ckks-bootstrap-waterline=25 debug=true",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "@heir//tests/Examples/common:relu2_composite.mlir",
)

go_test(
    name = "relu2compositedbg_test",
    size = "medium",
    srcs = [
        "heir_debug.go",
        "relu2_composite_debug_test.go",
    ],
    embed = [":relu2compositedbg"],
)
