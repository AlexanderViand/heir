# BUILD file for a pre-built CHEDDAR library.
# Used during development when CHEDDAR is built outside of bazel.
# Expects libcheddar.so at the repo root (i.e., /tmp/cheddar-fhe/build/).

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cheddar",
    srcs = ["build/libcheddar.so"],
    hdrs = glob(["include/**/*.h"]),
    includes = ["."],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-ltommath",
    ],
)
