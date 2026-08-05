package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Shared headers for the GPU e2e examples, compiled against @cheddar's
# install tree with the system toolchain (see bazel/cheddar/e2e.bzl):
# - cheddar_test_fixture.h: common CKKS parameter/context/encode boilerplate.
# - cheddar_emit_prelude.h: shims (RunLinearTransform / RunEvalPoly) that
#   cheddar-to-emitc lowers cheddar.linear_transform / cheddar.eval_poly
#   into; #include'd by each example's generated kernel.
# - e2e_check.h: googletest-shaped check macros (the hermetically built
#   googletest cannot link into these system-toolchain binaries).
exports_files([
    "cheddar_emit_prelude.h",
    "cheddar_test_fixture.h",
    "e2e_check.h",
])
