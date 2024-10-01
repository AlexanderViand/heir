cc_library(
    name = "set_intersect_codegen",
    srcs = ["heir_output.cpp"],
    hdrs = ["heir_output.h"],
    deps = ["@openfhe//:pke"],
)

cc_binary(
    name = "set_intersect_main",
    srcs = ["set_intersect_main.cpp"],
    deps = [
        ":set_intersect_codegen",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
