load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

exports_files(["mlp.mlir"])

heir_lattigo_lib(
    name = "mlp",
    go_library_name = "mlp",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30",
        "--scheme-to-lattigo",
    ],
    mlir_src = ":mlp.mlir",
)

go_test(
    name = "mlp_test",
    srcs = ["mlp_test.go"],
    embed = [":mlp"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)

# Known limitation: the EXT-based linear_transform helper accumulates
# noiseScaleDeg through mul->linear_transform chains. With 6 Q primes,
# nsd exceeds the CRT limb count after the 2nd multiplication. Fixing
# requires nsd-aware noise management (see management.md).
openfhe_end_to_end_test(
    name = "mlp_openfhe_test",
    size = "enormous",
    timeout = "long",
    generated_lib_header = "mlp_openfhe_lib.h",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=fixed-manual",
        "--scheme-to-openfhe=openfhe-scaling-technique=fixed-manual",
    ],
    mlir_src = ":mlp.mlir",
    tags = ["manual"],
    test_src = "mlp_openfhe_test.cpp",
)

openfhe_end_to_end_test(
    name = "mlp_openfhe_flexible_auto_test",
    size = "enormous",
    timeout = "long",
    generated_lib_header = "mlp_openfhe_flexible_auto_lib.h",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=flexible-auto",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto",
    ],
    mlir_src = ":mlp.mlir",
    tags = ["manual"],
    test_src = "mlp_openfhe_flexible_auto_test.cpp",
)

openfhe_end_to_end_test(
    name = "mlp_openfhe_flexible_auto_ext_test",
    size = "enormous",
    timeout = "long",
    generated_lib_header = "mlp_openfhe_flexible_auto_ext_lib.h",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=flexible-auto-ext",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto-ext",
    ],
    mlir_src = ":mlp.mlir",
    tags = ["manual"],
    test_src = "mlp_openfhe_flexible_auto_ext_test.cpp",
)

openfhe_end_to_end_test(
    name = "mlp_openfhe_flexible_auto_ext_debug_test",
    size = "enormous",
    timeout = "long",
    generated_lib_header = "mlp_openfhe_flexible_auto_ext_debug_lib.h",
    heir_opt_flags = [
        "--raise-ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false preserve-structured-ops=true scaling-mod-bits=26 first-mod-bits=30 openfhe-scaling-technique=flexible-auto-ext",
        "--scheme-to-openfhe=openfhe-scaling-technique=flexible-auto-ext insert-debug-handler-calls=true",
    ],
    heir_translate_flags = [
        "--openfhe-debug-helper-include-path=tests/Examples/openfhe/ckks/debug_helper.h",
    ],
    mlir_src = ":mlp.mlir",
    tags = ["manual"],
    test_src = "mlp_openfhe_flexible_auto_ext_debug_test.cpp",
    deps = ["@heir//tests/Examples/openfhe/ckks:debug_helper"],
)
