"""A macro providing an end-to-end library for FIDESlib codegen."""

load("@heir//bazel/openfhe:copts.bzl", "OPENMP_COPTS", "OPENMP_LINKOPTS")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

_FIDESLIB_LINKOPTS = [
    "-L/usr/local/cuda/lib64",
    "-lcuda",
    "-lcudart",
]

def fideslib_lib(
        name,
        mlir_src,
        generated_lib_header,
        cc_lib_target_name = None,
        heir_opt_flags = [],
        heir_translate_flags = [],
        data = [],
        tags = [],
        deps = [],
        **kwargs):
    """Generate and compile C++ emitted for the FIDESlib backend.

    Args:
      name: Basename for generated targets/files.
      mlir_src: Source mlir file to run through heir-translate.
      generated_lib_header: Name of generated header file.
      cc_lib_target_name: Optional explicit name for generated cc_library target.
      heir_opt_flags: Flags to pass to heir-opt before heir-translate.
      heir_translate_flags: Extra flags to pass to heir-translate.
      data: Data deps for heir-opt and cc_library.
      tags: Tags for generated targets.
      deps: Extra deps for generated cc_library.
      **kwargs: Additional kwargs forwarded to cc_library.
    """
    cc_codegen_target = name + ".heir_translate_cc"
    h_codegen_target = name + ".heir_translate_h"
    generated_cc_filename = "%s_lib.inc.cc" % name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name
    heir_translate_cc_flags = heir_translate_flags + ["--emit-fideslib-pke"]
    heir_translate_h_flags = heir_translate_flags + ["--emit-fideslib-pke-header"]

    cc_lib_target = cc_lib_target_name
    if not cc_lib_target:
        cc_lib_target = "_heir_%s" % name

    if heir_opt_flags:
        heir_opt(
            name = heir_opt_name,
            src = mlir_src,
            pass_flags = heir_opt_flags,
            generated_filename = generated_heir_opt_name,
            data = data,
        )
    else:
        generated_heir_opt_name = mlir_src

    heir_translate(
        name = cc_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_cc_flags,
        generated_filename = generated_cc_filename,
    )
    heir_translate(
        name = h_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_h_flags,
        generated_filename = generated_lib_header,
    )

    cc_library(
        name = cc_lib_target,
        srcs = [generated_cc_filename],
        hdrs = [generated_lib_header],
        deps = deps + ["@fideslib//:fideslib"],
        tags = tags,
        data = data,
        copts = OPENMP_COPTS,
        linkopts = OPENMP_LINKOPTS + _FIDESLIB_LINKOPTS,
        **kwargs
    )
