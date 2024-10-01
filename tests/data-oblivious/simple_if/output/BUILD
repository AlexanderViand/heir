cc_library(
    name = "simple_if_codegen",
    srcs = ["heir_output.cpp"],
    hdrs = ["heir_output.h"],
    deps = ["@openfhe//:pke"],
)

cc_binary(
    name = "simple_if_main",
    srcs = ["simple_if_main.cpp"],
    deps = [
        ":simple_if_codegen",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
