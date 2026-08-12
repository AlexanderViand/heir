load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Compile-only guard that `cheddar-to-emitc` emits C++ which honours CHEDDAR's
# move/const contract. Unlike the GPU end-to-end tests under
# tests/Examples/cheddar, this compiles the emitted code against a header-only
# stub (cheddar_stub.h) with no CUDA, so it runs in normal CI. The cheddar
# dialect and pass are registered unconditionally in heir-opt/-translate, so
# this needs neither a GPU nor --//:enable_cheddar.

heir_opt(
    name = "kernels_emitc",
    src = "kernels.mlir",
    generated_filename = "kernels_emitc.mlir",
    pass_flags = [
        "--cheddar-bufferize",
        "--fold-memref-alias-ops",
        "--cse",
        "--canonicalize",
        "--drop-equivalent-buffer-results",
        "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true",
        "--canonicalize",
        "--convert-to-emitc=filter-dialects=cheddar,arith,scf",
        "--cheddar-emitc-boundary",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "kernels_cpp_raw",
    src = ":kernels_emitc.mlir",
    generated_filename = "kernels_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

# Prepend the includes/usings the emitted body relies on.
genrule(
    name = "kernels_lib_src",
    srcs = [":kernels_raw.cc"],
    outs = ["kernels_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Conversions/CheddarToEmitC/compile/BUILD.
#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <vector>
#include "tests/Conversions/CheddarToEmitC/compile/cheddar_stub.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :kernels_raw.cc) >> $@
""",
)

cc_library(
    name = "cheddar_stub",
    hdrs = ["cheddar_stub.h"],
)

cc_library(
    name = "kernels_compiled",
    srcs = [":kernels_lib_src"],
    deps = [":cheddar_stub"],
)

genrule(
    name = "kernels_cyclops_lib_src",
    srcs = [":kernels_raw.cc"],
    outs = ["kernels_cyclops_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Conversions/CheddarToEmitC/compile/BUILD.
#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <vector>
#define HEIR_CYCLOPS_STUB
#include "tests/Conversions/CheddarToEmitC/compile/cheddar_stub.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :kernels_raw.cc) >> $@
""",
)

cc_library(
    name = "kernels_cyclops_compiled",
    srcs = [":kernels_cyclops_lib_src"],
    deps = [":cheddar_stub"],
)

# Compiling the generated C++ against the stub *is* the assertion: if the
# emitter produced C++ that violates CHEDDAR's move/const contract, this fails
# to build. build_test compiles without linking, so the stub only needs
# declarations (no method bodies), and no GPU/CUDA is involved.
build_test(
    name = "compile_test",
    targets = [
        ":kernels_compiled",
        ":kernels_cyclops_compiled",
    ],
)

# Compile the same generated kernels against the pinned scale-snu CHEDDAR API.
# The stub remains the fast CPU/CI contract check; this opt-in target detects
# drift between that contract and the actual library headers.
genrule(
    name = "kernels_real_lib_src",
    srcs = [":kernels_raw.cc"],
    outs = ["kernels_real_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Conversions/CheddarToEmitC/compile/BUILD.
#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"
#include "extension/EvalPoly.h"
#include "extension/LinearTransform.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :kernels_raw.cc) >> $@
""",
)

cc_library(
    name = "kernels_real_compiled",
    srcs = [":kernels_real_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

build_test(
    name = "real_compile_test",
    target_compatible_with = requires_cheddar(),
    targets = [":kernels_real_compiled"],
)

heir_opt(
    name = "setup_emitc",
    src = "setup.mlir",
    generated_filename = "setup_emitc.mlir",
    pass_flags = [
        "--cheddar-configure-crypto-context=entry-function=kernel",
        "--cheddar-bufferize",
        "--fold-memref-alias-ops",
        "--cse",
        "--canonicalize",
        "--drop-equivalent-buffer-results",
        "--canonicalize",
        "--convert-to-emitc=filter-dialects=cheddar,arith,scf",
        "--cheddar-emitc-boundary",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "setup_cpp_raw",
    src = ":setup_emitc.mlir",
    generated_filename = "setup_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "setup_real_lib_src",
    srcs = [":setup_raw.cc"],
    outs = ["setup_real_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Conversions/CheddarToEmitC/compile/BUILD.
#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"
#include "extension/EvalPoly.h"
#include "extension/LinearTransform.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :setup_raw.cc) >> $@
""",
)

cc_library(
    name = "setup_real_compiled",
    srcs = [":setup_real_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

build_test(
    name = "real_setup_compile_test",
    target_compatible_with = requires_cheddar(),
    targets = [":setup_real_compiled"],
)
