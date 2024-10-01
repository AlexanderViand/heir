cc_library(
    name = "sum_tensor_codegen",
    srcs = ["heir_output.cpp"],
    hdrs = ["heir_output.h"],
    deps = ["@openfhe//:pke"],
)

cc_binary(
    name = "sum_tensor_main",
    srcs = ["sum_tensor_main.cpp"],
    deps = [
        ":sum_tensor_codegen",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
