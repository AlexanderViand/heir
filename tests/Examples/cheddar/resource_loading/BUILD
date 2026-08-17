load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

heir_opt(
    name = "resource_emitc",
    src = "resource_loading.mlir",
    generated_filename = "resource_emitc.mlir",
    pass_flags = [
        "--ownership-based-buffer-deallocation",
        "--canonicalize",
        "--buffer-deallocation-simplification",
        "--bufferization-lower-deallocations",
        "--convert-to-emitc=filter-dialects=cheddar,arith,scf",
        "--cheddar-emitc-boundary",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "resource_cpp_raw",
    src = ":resource_emitc.mlir",
    generated_filename = "resource_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "resource_lib_src",
    srcs = [":resource_raw.cc"],
    outs = ["resource_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/resource_loading/BUILD.
#include <cstdint>
PRELUDE_EOF
cat $(location :resource_raw.cc) >> $@
""",
)

cc_library(
    name = "resource_lib",
    srcs = [":resource_lib_src"],
    deps = ["@heir//lib/Runtime:CleartextResource"],
)

cc_test(
    name = "resource_loading_test",
    srcs = ["resource_loading_test.cpp"],
    deps = [
        ":resource_lib",
        "@googletest//:gtest_main",
    ],
)
