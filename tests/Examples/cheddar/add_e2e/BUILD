load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# 1. cheddar dialect → emitc dialect
heir_opt(
    name = "add_e2e_emitc",
    src = "add_e2e.mlir",
    generated_filename = "add_e2e_emitc.mlir",
    pass_flags = ["--cheddar-to-emitc"],
)

# 2. emitc → raw C++ (just function bodies, no includes)
heir_translate(
    name = "add_e2e_cpp_raw",
    src = ":add_e2e_emitc.mlir",
    generated_filename = "add_e2e_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

# 3. Prepend a CHEDDAR prelude (includes + `using namespace cheddar; using
#    word = uint64_t;`) so the generated function bodies type-check.
genrule(
    name = "add_e2e_lib_src",
    srcs = [":add_e2e_raw.cc"],
    outs = ["add_e2e_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/add_e2e/BUILD.
#include <cstdint>
#include <utility>
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "UserInterface.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :add_e2e_raw.cc) >> $@
""",
)

cc_library(
    name = "add_e2e_lib",
    srcs = [":add_e2e_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "add_e2e_test",
    srcs = ["add_e2e_test.cpp"],
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":add_e2e_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
