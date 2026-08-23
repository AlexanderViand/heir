load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

heir_opt(
    name = "generated_bootstrap_setup_emitc",
    src = "@heir//tests/Pipelines/scheme_to_cheddar:bootstrap.mlir",
    externalize_constants = False,
    generated_filename = "generated_bootstrap_setup_emitc.mlir",
    pass_flags = [
        "--annotate-module=backend=cheddar scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8192 greedy-level-budget=6 greedy-bootstrap-waterline=3",
        "--scheme-to-cheddar=entry-function=bootstrap",
        "--cheddar-to-emitc",
    ],
)

heir_translate(
    name = "generated_bootstrap_setup_cpp_raw",
    src = ":generated_bootstrap_setup_emitc.mlir",
    generated_filename = "generated_bootstrap_setup_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "generated_bootstrap_setup_lib_src",
    srcs = [":generated_bootstrap_setup_raw.cc"],
    outs = ["generated_bootstrap_setup_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit.
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/EvkRequest.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :generated_bootstrap_setup_raw.cc) >> $@
""",
)

cc_library(
    name = "generated_bootstrap_setup_lib",
    srcs = [":generated_bootstrap_setup_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "generated_bootstrap_setup_test",
    timeout = "long",
    srcs = ["generated_bootstrap_setup_test.cpp"],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":generated_bootstrap_setup_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
