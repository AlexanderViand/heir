"""A macro providing an end-to-end test for FIDESlib codegen."""

load("@heir//bazel:fideslib/config.bzl", "fideslib_deps", "requires_fideslib")
load("@heir//bazel/openfhe:copts.bzl", "OPENMP_COPTS", "OPENMP_LINKOPTS")
load("@heir//tools:heir-fideslib.bzl", "fideslib_lib")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

def fideslib_end_to_end_test(
        name,
        mlir_src,
        test_src,
        generated_lib_header,
        heir_opt_flags = [],
        heir_translate_flags = [],
        data = [],
        tags = [],
        deps = [],
        **kwargs):
    """Generate FIDESlib code and run a C++ end-to-end test.

    Args:
      name: The cc_test target name.
      mlir_src: The source mlir file.
      test_src: The C++ test harness source file.
      generated_lib_header: Name of generated header file.
      heir_opt_flags: Flags to pass to heir-opt before heir-translate.
      heir_translate_flags: Extra flags to pass to heir-translate.
      data: Data deps for generated targets.
      tags: Tags passed to generated targets.
      deps: Extra deps for generated targets.
      **kwargs: Keyword args forwarded to fideslib_lib/cc_test.
    """
    cc_lib_target_name = "%s_cc_lib" % name
    fideslib_lib(
        name = name,
        mlir_src = mlir_src,
        generated_lib_header = generated_lib_header,
        cc_lib_target_name = cc_lib_target_name,
        heir_opt_flags = heir_opt_flags,
        heir_translate_flags = heir_translate_flags,
        data = data,
        tags = tags,
        deps = deps,
        **kwargs
    )

    cc_test(
        name = name,
        srcs = [test_src],
        target_compatible_with = requires_fideslib(),
        deps = fideslib_deps(
            deps + [
                ":" + cc_lib_target_name,
                "@openfhe//:core",
                "@openfhe//:pke",
                "@googletest//:gtest_main",
            ],
        ),
        tags = tags,
        data = data,
        copts = OPENMP_COPTS,
        linkopts = OPENMP_LINKOPTS,
        **kwargs
    )
