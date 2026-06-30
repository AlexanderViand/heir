"""CGGI Backend using tfhe-rs.

This backend lowers MLIR to tfhe-rs Rust code and compiles it into a
python module using maturin and pyo3. The resulting module exposes a
single function that encrypts its inputs, calls the homomorphic function
and decrypts the result.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
from functools import partial
from typing import Any

from heir.interfaces import BackendInterface, ClientInterface, CompilationResult
from heir.backends.util import common

Path = pathlib.Path


class CGGIClientInterface(ClientInterface):

  def __init__(self, compilation_result: CompilationResult):
    self.compilation_result = compilation_result

  def setup(self):
    pass

  def decrypt_result(self, result, **kwargs):
    return result

  def __getattr__(self, key: str) -> Any:
    try:
      return getattr(self.compilation_result.module, key)
    except AttributeError as e:
      raise AttributeError(f"Attribute {key} not found") from e

  def __call__(self, *args, **kwargs):
    return self.compilation_result.main_func(*args, **kwargs)


class CGGIRustBackend(BackendInterface):

  def run_backend(
      self,
      workspace_dir,
      heir_opt,
      heir_translate,
      func_name,
      arg_names,
      secret_args,
      heir_opt_output,
      debug,
  ):
    rust_filepath = Path(workspace_dir) / f"{func_name}.rs"
    heir_translate.run_binary(
        input=heir_opt_output,
        options=["--emit-tfhe-rust", "-o", rust_filepath],
    )

    cargo_toml = Path(workspace_dir) / "Cargo.toml"
    src_dir = Path(workspace_dir) / "src"
    src_dir.mkdir(exist_ok=True)
    generated_rs = src_dir / "generated.rs"
    rust_filepath.rename(generated_rs)

    lib_rs = src_dir / "lib.rs"
    lib_rs.write_text("""
use pyo3::prelude::*;
use tfhe::shortint::prelude::*;

include!("generated.rs");

#[pyfunction]
fn call(a: u8, b: u8) -> u8 {
    let params = get_parameters_from_message_and_carry(3, 2);
    let (ck, sk) = gen_keys(params);
    let ct_a = ck.encrypt(a.into());
    let ct_b = ck.encrypt(b.into());
    let res = fn_under_test(&sk, &ct_a, &ct_b);
    ck.decrypt(&res)
}

#[pymodule]
fn heir_generated(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(call, m)?)?;
    Ok(())
}
""")

    cargo_toml.write_text("""
[package]
name = "heir_generated"
version = "0.1.0"
edition = "2021"

[lib]
name = "heir_generated"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
tfhe = { version = "0.5.3", features = ["shortint", "x86_64-unix"] }
""")

    subprocess.run(
        [
            "maturin",
            "build",
            "--release",
            "--manifest-path",
            cargo_toml,
            "-o",
            workspace_dir,
            "-i",
            sys.executable,
        ],
        check=True,
    )

    ext = next(Path(workspace_dir).glob("heir_generated-*.whl"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", str(ext)], check=True
    )

    module = importlib.import_module("heir_generated")
    result = CompilationResult(
        module=module,
        func_name=func_name,
        arg_names=arg_names,
        secret_args=secret_args,
        main_func=module.call,
    )
    return CGGIClientInterface(result)
