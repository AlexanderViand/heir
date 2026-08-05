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
    name = "cheddar_cmake",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_UNITTEST": "OFF",
        "ENABLE_EXTENSION": "ON",
        "USE_GMP": "OFF",
        # Build for the local GPU architecture to keep build times manageable.
        "CMAKE_CUDA_ARCHITECTURES": "native",
        # Assumes conventional CUDA toolkit location and configuration.
        "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        "CMAKE_CUDA_HOST_COMPILER:FILEPATH": "/usr/bin/g++",
        "CMAKE_CUDA_FLAGS": "-ccbin=/usr/bin/g++",
        # Compile the host-side C++ with the system toolchain too (matching
        # the nvcc host compiler above) instead of the bazel-provided hermetic
        # clang. rules_foreign_cc exports the hermetic CC/CXX and its
        # clang-specific flags through the environment and merges (rather than
        # replaces) user cache entries into them, so the only clean override
        # point is a toolchain file: it is read before the environment-derived
        # values seed the flag caches. The file is added to the CHEDDAR source
        # tree by patches/cheddar.patch; the relative path resolves against
        # the source tree.
        "CMAKE_TOOLCHAIN_FILE": "cmake/bazel_toolchain.cmake",
    },
    # Ninja instead of Unix Makefiles: rules_foreign_cc downloads prebuilt
    # cmake/ninja but builds GNU make from source, and that bootstrap breaks
    # under the hermetic clang toolchain (autoconf preprocessor probes run
    # without the toolchain's CFLAGS-borne header search paths, so configure
    # mis-detects uid_t/gid_t and the compile fails on the resulting macros).
    generate_args = ["-GNinja"],
    generate_crosstool_file = False,
    lib_source = ":all_srcs",
    out_include_dir = "include",
    out_shared_libs = [
        "libcheddar.so",
    ],
    targets = ["cheddar"],
)

# The full cmake install tree (include/ + lib/), for consumers that compile
# against the library with the system toolchain (bazel/cheddar/e2e.bzl).
filegroup(
    name = "install_dir",
    srcs = [":cheddar_cmake"],
    output_group = "gen_dir",
)
