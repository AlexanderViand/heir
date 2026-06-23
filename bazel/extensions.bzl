"""Module extensions for MLIR Tutorial dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _llvm_deps_impl(_):
    """Implementation of the llvm_deps module extension."""
    LLVM_COMMIT = "93b1f81e03ef4c717a9d222b46f78c38bcff0a3f"

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
        patches = ["@heir//patches:cheddar.patch"],
        patch_args = ["-p1"],
    )

cheddar_deps = module_extension(implementation = _cheddar_deps_impl)
