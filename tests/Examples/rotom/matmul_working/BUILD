# A library build target that encapsulates the HEIR-generated code.
cc_library(
    name = "matmul_codegen",
    srcs = ["heir_output.cpp"],
    hdrs = ["heir_output.h"],
    deps = ["@openfhe//:pke"],
)

# An executable build target that contains your main function and links
# against the above.
cc_binary(
    name = "matmul_test",
    srcs = ["matmul_test.cpp"],
    deps = [
        ":matmul_codegen",
        "@openfhe//:pke",
        "@openfhe//:core",
    ],
)