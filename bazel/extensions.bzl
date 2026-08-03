"""Module extensions for MLIR Tutorial dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _llvm_deps_impl(_):
    """Implementation of the llvm_deps module extension."""
    LLVM_COMMIT = "6d1ace547c9f9ecf7df9ab87e2db74b5bd3f199d"

    # Download LLVM/MLIR using a git repository
    new_git_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
        patches = [
            # This patch file contains changes that are fixed in upstream LLVM
            # that are (usually) required to build HEIR, but are not included
            # as of the LLVM_COMMIT hash above (the fixes are still progressing
            # through the automated integration process). The patch file is
            # automatically generated, and should not be removed even if empty.
            "@heir//patches:llvm.patch",
            # Hand-maintained (NOT auto-generated): lets `emitc.subscript` take
            # an `!emitc.lvalue` base, needed by the cheddar EmitC emitter.
            "@heir//patches:emitc_subscript_lvalue.patch",
        ],
        patch_args = ["-p1"],
    )

llvm_deps = module_extension(
    implementation = _llvm_deps_impl,
)

# CHEDDAR GPU FHE library
CHEDDAR_COMMIT = "307b49cbe03e7f8f14bf31485f716c1090c9ec9d"

def _cheddar_deps_impl(_):
    maybe(
        new_git_repository,
        name = "cheddar",
        build_file = "@heir//bazel/cheddar:cheddar.BUILD",
        commit = CHEDDAR_COMMIT,
        remote = "https://github.com/scale-snu/cheddar-fhe.git",
        patches = [
            "@heir//patches:cheddar.patch",
            # Raise the NTT's compile-time max ring degree from 2^16 to 2^17 so
            # wide models (e.g. CriteoHELRM, whose 65536-slot embeddings force
            # logN=17) fit. CHEDDAR's NTT kernels are constexpr-generated per
            # log_degree, so this just emits the extra logN=17 specialization;
            # logN=17 ciphertexts are ~2x memory (needs a >=48GB GPU).
            "@heir//patches:cheddar_max_log_degree.patch",
        ],
        patch_args = ["-p1"],
    )

cheddar_deps = module_extension(implementation = _cheddar_deps_impl)
