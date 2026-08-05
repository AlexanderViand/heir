"""Compile-and-run rules for the CHEDDAR GPU end-to-end examples.

libcheddar.so is built by cmake with the system toolchain -- nvcc plus the
system g++/libstdc++ (see cheddar.BUILD) -- and its C++ API carries std::
types in exported symbol signatures. Binaries built by bazel's hermetic
clang/libc++ toolchain therefore cannot link against it. Instead, the e2e
harness is compiled with the same system g++ in a genrule and executed via
sh_test, mirroring how out-of-tree consumers (e.g. the medusa harness) build
against the installed library.
"""

load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")

_CUDA_ROOT = "/usr/local/cuda"

def cheddar_system_cc_binary(
        name,
        srcs,
        hdrs = [],
        visibility = None):
    """Compiles srcs with the system g++ against @cheddar's install tree.

    Args:
      name: name of the resulting binary artifact.
      srcs: sources compiled together into one binary (order preserved).
      hdrs: headers the sources include (made available, not compiled).
      visibility: standard attribute.
    """
    src_locs = " ".join(["$(locations {})".format(s) for s in srcs])
    native.genrule(
        name = name + "_compile",
        srcs = srcs + hdrs + ["@cheddar//:install_dir"],
        outs = [name],
        cmd = """
INSTALL_DIR="$(location @cheddar//:install_dir)"
/usr/bin/g++ -std=c++17 -O2 \\
  -isystem "$$INSTALL_DIR/include" \\
  -isystem {cuda_root}/include \\
  -iquote . -iquote $(GENDIR) \\
  {srcs} \\
  -L "$$INSTALL_DIR/lib" -lcheddar \\
  -L {cuda_root}/lib64 -lcudart \\
  -o $@
""".format(cuda_root = _CUDA_ROOT, srcs = src_locs),
        # The system g++/nvcc versions are not tracked as inputs, so results
        # must not travel between machines through a shared cache.
        tags = ["no-remote-cache"],
        target_compatible_with = requires_cheddar(),
        visibility = visibility,
    )

def cheddar_e2e_test(
        name,
        srcs,
        hdrs = [],
        data = [],
        tags = []):
    """A GPU e2e test: system-g++-compiled binary run under sh_test."""
    cheddar_system_cc_binary(
        name = name + "_bin",
        srcs = srcs,
        hdrs = hdrs,
    )
    native.sh_test(
        name = name,
        srcs = ["@heir//bazel/cheddar:run_e2e_test.sh"],
        args = [
            "$(location :{}_bin)".format(name),
            "$(location @cheddar//:install_dir)",
        ],
        data = [
            ":{}_bin".format(name),
            "@cheddar//:install_dir",
        ] + data,
        tags = [
            "exclusive",
            "no-remote-cache",
            "notap",
        ] + tags,
        target_compatible_with = requires_cheddar(),
    )
