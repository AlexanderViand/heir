# BUILD file for CHEDDAR FHE library, used with rules_foreign_cc cmake rule.
# This is consumed by the new_local_repository or new_git_repository rule.

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "cheddar",
    build_args = ["-j8"],
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_UNITTEST": "OFF",
        "ENABLE_EXTENSION": "ON",
        "USE_GMP": "OFF",
        # CUDA paths — may need adjustment per machine
        "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
    },
    lib_source = ":all_srcs",
    out_shared_libs = ["libcheddar.so"],
    visibility = ["//visibility:public"],
)
