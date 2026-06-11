import os
import shutil
from pathlib import Path

from lit.formats import ShTest

config.name = "heir"
config.test_format = ShTest()
config.suffixes = [".mlir", ".v"]

# lit executes relative to the directory
#
#   bazel-bin/tests/<test_target_name>.runfiles/_main/
#
# which contains tools/ and tests/ directories and the binary targets built
# within them, brought in via the `data` attribute in the BUILD file. To
# manually inspect the filesystem in situ, add the following to this script and
# run `bazel test //tests:<target>`
#
# import subprocess
#
# print(subprocess.run(["pwd",]).stdout)
# print(subprocess.run(["ls", "-l", os.environ["RUNFILES_DIR"]]).stdout)
# print(subprocess.run([ "env", ]).stdout)
#
# Hence, to get lit to see tools like `heir-opt`, we need to add the tools/
# subdirectory to the PATH environment variable.
#
# Bazel defines RUNFILES_DIR which includes _main/ and third party dependencies
# as their own directory. Generally, it seems that $PWD == $RUNFILES_DIR/_main/

runfiles_dir = Path(os.environ["RUNFILES_DIR"])

llvm_project_canonical_name = "+_repo_rules+llvm-project"
mlir_tools_relpath = llvm_project_canonical_name + "/mlir"
llvm_tools_relpath = llvm_project_canonical_name + "/llvm"
mlir_tools_path = runfiles_dir.joinpath(Path(mlir_tools_relpath))
tool_relpaths = [
    mlir_tools_relpath,
    llvm_tools_relpath,
    "_main/tools",
    "_main/tests/Emitter/verilog",
    "yosys+",
]

config.environment["PATH"] = (
    ":".join(str(runfiles_dir.joinpath(Path(path))) for path in tool_relpaths)
    + ":"
    + os.environ["PATH"]
)

abc_relpath = "abc+/abc_bin"
config.environment["HEIR_ABC_BINARY"] = str(
    runfiles_dir.joinpath(Path(abc_relpath))
)
yosys_libs = "_main/lib/Transforms/YosysOptimizer/yosys"
config.environment["HEIR_YOSYS_SCRIPTS_DIR"] = str(
    runfiles_dir.joinpath(Path(yosys_libs))
)

# veir-opt is an optional external tool (a formally verified Lean-based
# MLIR-compatible optimizer, built out-of-band with `lake build`) used by the
# --mod-arith-to-arith-veir pass. It is located, in order of precedence, via
# the HEIR_VEIR_OPT_PATH environment variable (forward it with
# --test_env=HEIR_VEIR_OPT_PATH), the @veir bazel repository (see
# MODULE.bazel), or the PATH. If found, the `veir` lit feature is set so that
# tests marked with `REQUIRES: veir` run; otherwise they are skipped.
veir_opt_path = os.environ.get("HEIR_VEIR_OPT_PATH")
if not veir_opt_path:
    bazel_veir_opt = runfiles_dir.joinpath(Path("+_repo_rules+veir/veir-opt-bin"))
    if bazel_veir_opt.exists():
        veir_opt_path = str(bazel_veir_opt)
if not veir_opt_path:
    veir_opt_path = shutil.which("veir-opt")
if veir_opt_path and os.path.exists(veir_opt_path):
    config.available_features.add("veir")
    config.environment["HEIR_VEIR_OPT_PATH"] = veir_opt_path

# Some tests that use mlir-runner need access to additional shared libs to
# link against functions like print. Substitutions replace magic strings in the
# test files with the needed paths.
substitutions = {
    "%mlir_lib_dir": str(mlir_tools_path),
}
config.substitutions.extend(substitutions.items())
