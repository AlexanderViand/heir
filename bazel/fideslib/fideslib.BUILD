load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "all_srcs",
    srcs = glob(
        ["**"],
        exclude = [
            "bazel-*/**",
            "build/**",
            "cmake-build*/**",
        ],
    ),
)

cmake(
    name = "fideslib_cmake",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        # Build for the local GPU architecture to keep build times manageable.
        "FIDESLIB_ARCH": "native",
        "FIDESLIB_CUDA_STANDARD": "20",
        "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        "CMAKE_CUDA_HOST_COMPILER:FILEPATH": "/usr/bin/g++",
        "CMAKE_CUDA_FLAGS": "-ccbin=/usr/bin/g++",
        # Wipe toolchain-provided linker flags that can inject unsupported host options.
        "CMAKE_EXE_LINKER_FLAGS": "",
        "CMAKE_MODULE_LINKER_FLAGS": "",
        "CMAKE_SHARED_LINKER_FLAGS": "",
        "FIDESLIB_COMPILE_BENCHMARKS": "OFF",
        "FIDESLIB_COMPILE_TESTS": "OFF",
        "FIDESLIB_INSTALL_OPENFHE": "OFF",
        "FIDESLIB_INSTALL_PREFIX": "$$INSTALLDIR",
        "FIDESLIB_USE_BAZEL_OPENFHE": "ON",
        "OPENFHE_BAZEL_PREFIX": "$$EXT_BUILD_DEPS",
    },
    generate_crosstool_file = False,
    lib_source = ":all_srcs",
    out_include_dir = "include",
    out_static_libs = [
        "fideslib.a",
    ],
    targets = ["fideslib"],
    deps = [
        "@openfhe//:pke",
    ],
)

alias(
    name = "fideslib",
    actual = ":fideslib_cmake",
)
