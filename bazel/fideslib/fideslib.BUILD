load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "fideslib_cmake",
    cache_entries = {
        "FIDESLIB_USE_BAZEL_OPENFHE": "ON",
        "OPENFHE_BAZEL_PREFIX": "$$EXT_BUILD_DEPS",
        "CMAKE_BUILD_TYPE": "Release",
        # Build only for the local GPU architecture to keep build times manageable.
        "FIDESLIB_ARCH": "native",
        "FIDESLIB_CUDA_STANDARD": "20",
        # Bazel action PATH does not include /usr/local/cuda/bin by default.
        "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        # Keep clang for host C/C++ compile, but use gcc as nvcc host compiler.
        "CMAKE_C_COMPILER:FILEPATH": "/usr/bin/clang",
        "CMAKE_CXX_COMPILER:FILEPATH": "/usr/bin/clang++",
        "CMAKE_CUDA_HOST_COMPILER:FILEPATH": "/usr/bin/g++",
        "CMAKE_CUDA_FLAGS": "-ccbin=/usr/bin/g++",
        # Wipe toolchain-provided linker flags that inject unsupported g++ options.
        "CMAKE_EXE_LINKER_FLAGS": "",
        "CMAKE_MODULE_LINKER_FLAGS": "",
        "CMAKE_SHARED_LINKER_FLAGS": "",
        "FIDESLIB_COMPILE_BENCHMARKS": "OFF",
        "FIDESLIB_COMPILE_TESTS": "OFF",
        # Keep OpenFHE ownership in HEIR's Bazel graph.
        "FIDESLIB_INSTALL_OPENFHE": "OFF",
    },
    generate_crosstool_file = False,
    lib_source = ":all_srcs",
    out_include_dir = "include",
    out_static_libs = [
        "fideslib.a",
    ],
    target_compatible_with = select({
        "@heir//:config_enable_fideslib": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    targets = ["fideslib"],
    deps = [
        "@openfhe//:pke",
    ],
)

alias(
    name = "fideslib",
    actual = ":fideslib_cmake",
)
