"""Lattigo Backend."""

import os
import pathlib
import subprocess
import sys
import tempfile
import time
import importlib
from typing import Dict, List, Optional, Tuple, Any

import colorama
from colorama import Fore, Style

from heir.interfaces import BackendInterface, ClientInterface, CompilationResult, EncValue
from heir.backends.lattigo.config import LattigoConfig
from heir.backends.util.common import strip_and_verify_eval_arg_consistency

Path = pathlib.Path


class LattigoClientInterface(ClientInterface):
  """Client interface for the Lattigo backend."""

  def __init__(
      self, compilation_result: Dict[str, Any], go_code_path: pathlib.Path
  ):
    # Convert the dict to a CompilationResult object
    self.compilation_result = CompilationResult(
        module=None,
        func_name=compilation_result["func_name"],
        arg_names=compilation_result["arg_names"],
        secret_args=compilation_result["secret_args"],
    )
    self.go_code_path = go_code_path
    self.workspace_dir = go_code_path.parent
    self.func_name = compilation_result["func_name"]
    self.arg_names = compilation_result["arg_names"]
    self.secret_args = compilation_result["secret_args"]
    self.crypto_context = None
    self.keypair = None

  def setup(self):
    """Configure the initial cryptosystem and setup key generation."""
    # Execute Go setup code to generate keys and configuration
    config_cmd = ["go", "run", f"{self.func_name}_setup.go"]
    try:
      setup_output = subprocess.run(
          config_cmd,
          cwd=self.workspace_dir,
          capture_output=True,
          text=True,
          check=True,
      )
      if setup_output.returncode != 0:
        raise RuntimeError(
            f"Failed to setup Lattigo cryptosystem: {setup_output.stderr}"
        )

      # After setup, keys should be generated
      self.crypto_context = True  # Just a flag to indicate setup is complete

    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Failed to setup Lattigo cryptosystem: {e.stderr}")

    except FileNotFoundError:
      raise RuntimeError(
          "Go executable not found. Please make sure Go is installed and in"
          " your PATH."
      )

  def decrypt_result(self, result, **kwargs):
    """Decrypt the encrypted result using the Lattigo backend."""
    if not self.crypto_context:
      raise RuntimeError("Please call setup() before decrypting.")

    # Create a temp file to store the encrypted result
    with tempfile.NamedTemporaryFile(
        suffix=".bin", dir=self.workspace_dir, delete=False
    ) as result_file:
      result_path = result_file.name

    # Write the encrypted result to the file
    with open(result_path, "wb") as f:
      f.write(result)

    # Execute the Go decryption code
    decrypt_cmd = ["go", "run", f"{self.func_name}_decrypt.go", result_path]

    try:
      decrypt_output = subprocess.run(
          decrypt_cmd,
          cwd=self.workspace_dir,
          capture_output=True,
          text=True,
          check=True,
      )

      # Clean up the temp file
      try:
        os.remove(result_path)
      except OSError:
        pass

      # Parse the decrypted result from stdout
      return self._parse_decryption_output(decrypt_output.stdout)

    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Failed to decrypt result: {e.stderr}")

  def _parse_decryption_output(self, output: str):
    """Parse the decrypted output from the Go program."""
    # Basic parsing - can be enhanced based on output format
    output = output.strip()

    # Try to convert to appropriate type based on the output
    try:
      # Try to parse as int first
      return int(output)
    except ValueError:
      try:
        # Try to parse as float
        return float(output)
      except ValueError:
        # If it's a list or more complex structure, parse accordingly
        if output.startswith("[") and output.endswith("]"):
          # Parse as list
          content = output[1:-1].split(",")
          return [float(item.strip()) for item in content if item.strip()]

        # Return as string if no other parsing works
        return output

  def __getattr__(self, name: str):
    """Handle dynamic attribute access for encryption functions."""
    if name == "crypto_context" and not hasattr(self, "_crypto_context"):
      msg = (
          f"HEIR Error: Please call {self.func_name}.setup()"
          " before calling"
          f" {self.func_name}.encrypt/eval/decrypt"
      )
      colorama.init(autoreset=True)
      print(Fore.RED + Style.BRIGHT + msg)
      raise RuntimeError(msg)

    if name.startswith("encrypt_"):
      # Extract argument name
      arg_name = name[len("encrypt_") :]

      if arg_name not in self.arg_names:
        raise AttributeError(f"No encryption function for argument {arg_name}")

      # Create an encryption function for this argument
      def encrypt_func(arg, **kwargs):
        if not self.crypto_context:
          raise RuntimeError("Please call setup() before encrypting.")

        # Create a temp file to store the plaintext input
        with tempfile.NamedTemporaryFile(
            suffix=".txt", dir=self.workspace_dir, delete=False
        ) as input_file:
          input_path = input_file.name

        # Write the input to the file in the appropriate format
        with open(input_path, "w") as f:
          if isinstance(arg, list):
            f.write(",".join(map(str, arg)))
          else:
            f.write(str(arg))

        # Execute the Go encryption code
        encrypt_cmd = [
            "go",
            "run",
            f"{self.func_name}_encrypt_{arg_name}.go",
            input_path,
        ]

        try:
          encrypt_output = subprocess.run(
              encrypt_cmd,
              cwd=self.workspace_dir,
              capture_output=True,
              check=True,
          )

          # Output file with encrypted data
          output_path = f"{input_path}.enc"

          # Read encrypted data
          with open(output_path, "rb") as f:
            encrypted_data = f.read()

          # Clean up temp files
          try:
            os.remove(input_path)
            os.remove(output_path)
          except OSError:
            pass

          return EncValue(arg_name, encrypted_data)

        except subprocess.CalledProcessError as e:
          raise RuntimeError(f"Failed to encrypt {arg_name}: {e.stderr}")

      return encrypt_func

    raise AttributeError(
        f"'{self.__class__.__name__}' has no attribute '{name}'"
    )

  def eval(self, *args, **kwargs):
    """Evaluate the function on encrypted inputs."""
    if not self.crypto_context:
      raise RuntimeError("Please call setup() before evaluating.")

    # Verify argument consistency
    stripped_args, stripped_kwargs = strip_and_verify_eval_arg_consistency(
        self.compilation_result, *args, **kwargs
    )

    # Create temp files for each encrypted input
    input_files = []
    for i, arg in enumerate(stripped_args):
      with tempfile.NamedTemporaryFile(
          suffix=f".{i}.bin", dir=self.workspace_dir, delete=False
      ) as f:
        if isinstance(arg, bytes):
          f.write(arg)
        else:
          f.write(str(arg).encode())
        input_files.append(f.name)

    # Execute the Go evaluation code
    eval_cmd = ["go", "run", f"{self.func_name}_eval.go"] + input_files

    try:
      eval_output = subprocess.run(
          eval_cmd, cwd=self.workspace_dir, capture_output=True, check=True
      )

      # Output file with evaluation result
      output_path = os.path.join(
          self.workspace_dir, f"{self.func_name}_result.bin"
      )

      # Read result data
      with open(output_path, "rb") as f:
        result_data = f.read()

      # Clean up temp files
      for file_path in input_files:
        try:
          os.remove(file_path)
        except OSError:
          pass

      try:
        os.remove(output_path)
      except OSError:
        pass

      return result_data

    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Failed to evaluate function: {e.stderr}")

  def __call__(self, *args, **kwargs):
    """Invoke setup, encryption, eval and decryption seamlessly."""
    # Setup
    self.setup()

    # Encrypt arguments
    if len(self.arg_names) != len(args):
      raise ValueError(
          f"Expected {len(self.arg_names)} arguments, got {len(args)}"
      )

    new_args = []
    for i, arg in enumerate(args):
      if i in self.secret_args:
        # Get encryption function
        encrypt_func = getattr(self, f"encrypt_{self.arg_names[i]}")
        new_args.append(encrypt_func(arg))
      else:
        new_args.append(arg)

    new_kwargs = {}
    for arg_name, arg in kwargs.items():
      i = self.arg_names.index(arg_name)
      if i in self.secret_args:
        encrypt_func = getattr(self, f"encrypt_{arg_name}")
        new_kwargs[arg_name] = encrypt_func(arg)
      else:
        new_kwargs[arg_name] = arg

    # Evaluate
    result = self.eval(*new_args, **new_kwargs)

    # Decrypt
    return self.decrypt_result(result)


class LattigoBackend(BackendInterface):
  """Backend to emit Go code via the Lattigo API."""

  def __init__(self, config: LattigoConfig):
    self.config = config

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
    # Initialize colored output
    colorama.init(autoreset=True)

    # Lower to Lattigo dialect
    opt_flags = [
        f"--scheme-to-lattigo=entry-function={func_name}",
        "--mlir-print-debuginfo",
    ]
    if debug:
      print(
          "HEIRpy Debug (Lattigo Backend):",
          colorama.Style.BRIGHT + f"Running heir-opt {' '.join(opt_flags)}",
      )
    lattigo_mlir, graph = heir_opt.run_binary_stderr(
        input=heir_opt_output, options=opt_flags
    )
    if debug:
      mlir_path = Path(workspace_dir) / f"{func_name}.backend.mlir"
      dot_path = Path(workspace_dir) / f"{func_name}.backend.dot"
      print(
          f"HEIRpy Debug (Lattigo Backend): Writing backend MLIR to {mlir_path}"
      )
      mlir_path.write_text(lattigo_mlir)
      print(
          f"HEIRpy Debug (Lattigo Backend): Writing backend graph to {dot_path}"
      )
      dot_path.write_text(graph)

    # Emit Go code for the main function
    go_filepath = Path(workspace_dir) / f"{func_name}.go"
    translate_flags = [
        "--emit-lattigo",
        f"--package-name=main",
        "-o",
        str(go_filepath),
    ]
    if debug:
      print(
          "HEIRpy Debug (Lattigo Backend):",
          colorama.Style.BRIGHT
          + f"Running heir-translate {' '.join(translate_flags)}",
      )
    heir_translate.run_binary(input=lattigo_mlir, options=translate_flags)

    # Generate additional Go files for setup, encryption, evaluation, and decryption
    self._generate_go_helper_files(
        workspace_dir, func_name, arg_names, secret_args, debug
    )

    # Generate Go module files
    self._generate_go_module_files(workspace_dir, func_name, debug)

    # Install Go dependencies if needed
    self._install_go_dependencies(workspace_dir, debug)

    # Create a compilation result dict for the client interface
    compilation_result = {
        "func_name": func_name,
        "arg_names": arg_names,
        "secret_args": secret_args,
    }

    return LattigoClientInterface(compilation_result, go_filepath)

  def _generate_go_helper_files(
      self, workspace_dir, func_name, arg_names, secret_args, debug
  ):
    """Generate helper Go files for setup, encryption, evaluation, and decryption."""
    # Create setup.go for key generation and parameters
    setup_file_path = Path(workspace_dir) / f"{func_name}_setup.go"
    setup_code = self._generate_setup_code(func_name)
    with open(setup_file_path, "w") as f:
      f.write(setup_code)

    # Create encryption files for each secret argument
    for i, arg_name in enumerate(arg_names):
      if i in secret_args:
        encrypt_file_path = (
            Path(workspace_dir) / f"{func_name}_encrypt_{arg_name}.go"
        )
        encrypt_code = self._generate_encrypt_code(func_name, arg_name, i)
        with open(encrypt_file_path, "w") as f:
          f.write(encrypt_code)

    # Create evaluation file
    eval_file_path = Path(workspace_dir) / f"{func_name}_eval.go"
    eval_code = self._generate_eval_code(func_name, arg_names)
    with open(eval_file_path, "w") as f:
      f.write(eval_code)

    # Create decryption file
    decrypt_file_path = Path(workspace_dir) / f"{func_name}_decrypt.go"
    decrypt_code = self._generate_decrypt_code(func_name)
    with open(decrypt_file_path, "w") as f:
      f.write(decrypt_code)

    if debug:
      print(
          "HEIRpy Debug (Lattigo Backend): Generated helper Go files in"
          f" {workspace_dir}"
      )

  def _generate_go_module_files(self, workspace_dir, func_name, debug):
    """Generate necessary Go module files for the project."""
    # Create go.mod file
    go_mod_path = Path(workspace_dir) / "go.mod"
    go_mod_content = f"""module {func_name}

go 1.19

require (
    github.com/tuneinsight/lattigo/v4 v4.1.0
)
"""
    with open(go_mod_path, "w") as f:
      f.write(go_mod_content)

    # Create a README.md explaining the Go code
    readme_path = Path(workspace_dir) / "README.md"
    readme_content = f"""# {func_name} - Lattigo HE Implementation

This directory contains a homomorphic encryption implementation of the `{func_name}` function using the [Lattigo](https://github.com/tuneinsight/lattigo) library.

## Files

- `{func_name}.go`: Main implementation of the function
- `{func_name}_setup.go`: Sets up the cryptosystem parameters and keys
- `{func_name}_eval.go`: Evaluates the function on encrypted inputs
- `{func_name}_decrypt.go`: Decrypts the result

## Usage

1. Setup the cryptosystem:
   ```
   go run {func_name}_setup.go
   ```

2. Encrypt inputs:
   ```
   go run {func_name}_encrypt_<arg_name>.go <input_file>
   ```

3. Evaluate the function:
   ```
   go run {func_name}_eval.go <encrypted_input_files...>
   ```

4. Decrypt the result:
   ```
   go run {func_name}_decrypt.go <result_file>
   ```

Note: This code is automatically generated by the HEIR framework.
"""
    with open(readme_path, "w") as f:
      f.write(readme_content)

    if debug:
      print(
          "HEIRpy Debug (Lattigo Backend): Generated Go module files in"
          f" {workspace_dir}"
      )

  def _install_go_dependencies(self, workspace_dir, debug):
    """Install necessary Go dependencies for the Lattigo project."""
    try:
      if debug:
        print(
            "HEIRpy Debug (Lattigo Backend): Installing Go dependencies in"
            f" {workspace_dir}"
        )

      # Run go mod tidy to install dependencies
      go_tidy_cmd = ["go", "mod", "tidy"]
      tidy_process = subprocess.run(
          go_tidy_cmd, cwd=workspace_dir, capture_output=True, text=True
      )

      if tidy_process.returncode != 0 and debug:
        print(
            "HEIRpy Debug (Lattigo Backend): Warning - go mod tidy failed:"
            f" {tidy_process.stderr}"
        )

      # Download the dependencies
      go_download_cmd = ["go", "mod", "download"]
      download_process = subprocess.run(
          go_download_cmd, cwd=workspace_dir, capture_output=True, text=True
      )

      if download_process.returncode != 0 and debug:
        print(
            "HEIRpy Debug (Lattigo Backend): Warning - go mod download failed:"
            f" {download_process.stderr}"
        )

    except Exception as e:
      if debug:
        print(
            "HEIRpy Debug (Lattigo Backend): Warning - failed to install Go"
            f" dependencies: {str(e)}"
        )
      # Continue even if dependencies fail to install - they might be already installed

  def _generate_setup_code(self, func_name):
    """Generate Go code for cryptosystem setup."""
    return f"""package main

import (
    "fmt"
    "os"
    "encoding/json"
    "io/ioutil"

    "github.com/tuneinsight/lattigo/v4/bgv"
    "github.com/tuneinsight/lattigo/v4/rlwe"
)

func main() {{
    // Get parameters from the main function
    var params bgv.Parameters
    var err error

    // Use the configure function from the generated code
    params, err = {func_name}__configure()
    if err != nil {{
        fmt.Printf("Error configuring parameters: %v\\n", err)
        os.Exit(1)
    }}

    // Generate keys
    kgen := bgv.NewKeyGenerator(params)
    sk := kgen.GenSecretKey()
    pk := kgen.GenPublicKey(sk)

    // Generate evaluation keys (relinearization key)
    evk := kgen.GenRelinearizationKey(sk, 1)

    // Save keys and parameters to files
    paramsBytes, _ := json.Marshal(params)
    skBytes, _ := json.Marshal(sk)
    pkBytes, _ := json.Marshal(pk)
    evkBytes, _ := json.Marshal(evk)

    err = ioutil.WriteFile("params.json", paramsBytes, 0644)
    if err != nil {{
        fmt.Printf("Error writing params.json: %v\\n", err)
        os.Exit(1)
    }}

    err = ioutil.WriteFile("sk.json", skBytes, 0644)
    if err != nil {{
        fmt.Printf("Error writing sk.json: %v\\n", err)
        os.Exit(1)
    }}

    err = ioutil.WriteFile("pk.json", pkBytes, 0644)
    if err != nil {{
        fmt.Printf("Error writing pk.json: %v\\n", err)
        os.Exit(1)
    }}

    err = ioutil.WriteFile("evk.json", evkBytes, 0644)
    if err != nil {{
        fmt.Printf("Error writing evk.json: %v\\n", err)
        os.Exit(1)
    }}

    fmt.Println("Cryptosystem setup completed successfully")
}}
"""

  def _generate_encrypt_code(self, func_name, arg_name, arg_index):
    """Generate Go code for encrypting arguments."""
    return f"""package main

import (
    "fmt"
    "os"
    "io/ioutil"
    "encoding/json"
    "strings"
    "strconv"

    "github.com/tuneinsight/lattigo/v4/bgv"
    "github.com/tuneinsight/lattigo/v4/rlwe"
)

func main() {{
    if len(os.Args) < 2 {{
        fmt.Println("Usage: go run {func_name}_encrypt_{arg_name}.go <input_file>")
        os.Exit(1)
    }}

    inputFile := os.Args[1]

    // Read input data
    inputData, err := ioutil.ReadFile(inputFile)
    if err != nil {{
        fmt.Printf("Error reading input file: %v\\n", err)
        os.Exit(1)
    }}

    // Parse input
    inputStr := strings.TrimSpace(string(inputData))
    var values []int64

    // Check if it's a comma-separated list
    if strings.Contains(inputStr, ",") {{
        // Parse as array
        parts := strings.Split(inputStr, ",")
        for _, part := range parts {{
            val, err := strconv.ParseInt(strings.TrimSpace(part), 10, 64)
            if err != nil {{
                fmt.Printf("Error parsing input value: %v\\n", err)
                os.Exit(1)
            }}
            values = append(values, val)
        }}
    }} else {{
        // Parse as single value
        val, err := strconv.ParseInt(inputStr, 10, 64)
        if err != nil {{
            fmt.Printf("Error parsing input value: %v\\n", err)
            os.Exit(1)
        }}
        values = append(values, val)
    }}

    // Load parameters and keys
    paramsBytes, err := ioutil.ReadFile("params.json")
    if err != nil {{
        fmt.Printf("Error reading params.json: %v\\n", err)
        os.Exit(1)
    }}

    pkBytes, err := ioutil.ReadFile("pk.json")
    if err != nil {{
        fmt.Printf("Error reading pk.json: %v\\n", err)
        os.Exit(1)
    }}

    var params bgv.Parameters
    var pk *rlwe.PublicKey

    err = json.Unmarshal(paramsBytes, &params)
    if err != nil {{
        fmt.Printf("Error unmarshaling parameters: %v\\n", err)
        os.Exit(1)
    }}

    err = json.Unmarshal(pkBytes, &pk)
    if err != nil {{
        fmt.Printf("Error unmarshaling public key: %v\\n", err)
        os.Exit(1)
    }}

    // Create encoder and encryptor
    encoder := bgv.NewEncoder(params)
    encryptor := bgv.NewEncryptor(params, pk)

    // Encode plaintext
    plaintext := bgv.NewPlaintext(params, params.MaxLevel())

    // Use the specific encryption function from the generated code
    err = {func_name}__encrypt__arg{arg_index}(encoder, values, plaintext)
    if err != nil {{
        fmt.Printf("Error encoding values: %v\\n", err)
        os.Exit(1)
    }}

    // Encrypt
    ciphertext := encryptor.EncryptNew(plaintext)

    // Serialize ciphertext
    ctBytes, err := ciphertext.MarshalBinary()
    if err != nil {{
        fmt.Printf("Error serializing ciphertext: %v\\n", err)
        os.Exit(1)
    }}

    // Write to output file
    outputFile := inputFile + ".enc"
    err = ioutil.WriteFile(outputFile, ctBytes, 0644)
    if err != nil {{
        fmt.Printf("Error writing output file: %v\\n", err)
        os.Exit(1)
    }}

    fmt.Printf("Encrypted {arg_name} successfully\\n")
}}
"""

  def _generate_eval_code(self, func_name, arg_names):
    """Generate Go code for evaluating the function."""
    arg_loading = ""
    for i in range(len(arg_names)):
      arg_loading += f"""
    // Load input {i}
    if {i} < len(inputFiles) {{
        inputData, err := ioutil.ReadFile(inputFiles[{i}])
        if err != nil {{
            fmt.Printf("Error reading input file %s: %v\\n", inputFiles[{i}], err)
            os.Exit(1)
        }}

        // Deserialize ciphertext
        inputs[{i}] = bgv.NewCiphertext(params, 1, params.MaxLevel())
        err = inputs[{i}].UnmarshalBinary(inputData)
        if err != nil {{
            fmt.Printf("Error deserializing input %d: %v\\n", {i}, err)
            os.Exit(1)
        }}
    }}
"""

    return f"""package main

import (
    "fmt"
    "os"
    "io/ioutil"
    "encoding/json"

    "github.com/tuneinsight/lattigo/v4/bgv"
    "github.com/tuneinsight/lattigo/v4/rlwe"
)

func main() {{
    if len(os.Args) < 2 {{
        fmt.Println("Usage: go run {func_name}_eval.go <input_files...>")
        os.Exit(1)
    }}

    inputFiles := os.Args[1:]

    // Load parameters
    paramsBytes, err := ioutil.ReadFile("params.json")
    if err != nil {{
        fmt.Printf("Error reading params.json: %v\\n", err)
        os.Exit(1)
    }}

    // Load evaluation keys
    evkBytes, err := ioutil.ReadFile("evk.json")
    if err != nil {{
        fmt.Printf("Error reading evk.json: %v\\n", err)
        os.Exit(1)
    }}

    var params bgv.Parameters
    var evk *rlwe.RelinearizationKey

    err = json.Unmarshal(paramsBytes, &params)
    if err != nil {{
        fmt.Printf("Error unmarshaling parameters: %v\\n", err)
        os.Exit(1)
    }}

    err = json.Unmarshal(evkBytes, &evk)
    if err != nil {{
        fmt.Printf("Error unmarshaling evaluation key: %v\\n", err)
        os.Exit(1)
    }}

    // Create evaluator with relinearization key
    evaluator := bgv.NewEvaluator(params, rlwe.EvaluationKey{{Rlk: evk}})

    // Load inputs
    inputs := make([]*bgv.Ciphertext, {len(arg_names)})
    {arg_loading}

    // Evaluate function
    result, err := {func_name}(evaluator, params, inputs...)
    if err != nil {{
        fmt.Printf("Error evaluating function: %v\\n", err)
        os.Exit(1)
    }}

    // Serialize result
    resultBytes, err := result.MarshalBinary()
    if err != nil {{
        fmt.Printf("Error serializing result: %v\\n", err)
        os.Exit(1)
    }}

    // Write to output file
    outputFile := "{func_name}_result.bin"
    err = ioutil.WriteFile(outputFile, resultBytes, 0644)
    if err != nil {{
        fmt.Printf("Error writing result to file: %v\\n", err)
        os.Exit(1)
    }}

    fmt.Println("Evaluation completed successfully")
}}
"""

  def _generate_decrypt_code(self, func_name):
    """Generate Go code for decrypting results."""
    return f"""package main

import (
    "fmt"
    "os"
    "io/ioutil"
    "encoding/json"
    "strings"

    "github.com/tuneinsight/lattigo/v4/bgv"
    "github.com/tuneinsight/lattigo/v4/rlwe"
)

func main() {{
    if len(os.Args) < 2 {{
        fmt.Println("Usage: go run {func_name}_decrypt.go <result_file>")
        os.Exit(1)
    }}

    resultFile := os.Args[1]

    // Load parameters and keys
    paramsBytes, err := ioutil.ReadFile("params.json")
    if err != nil {{
        fmt.Printf("Error reading params.json: %v\\n", err)
        os.Exit(1)
    }}

    skBytes, err := ioutil.ReadFile("sk.json")
    if err != nil {{
        fmt.Printf("Error reading sk.json: %v\\n", err)
        os.Exit(1)
    }}

    var params bgv.Parameters
    var sk *rlwe.SecretKey

    err = json.Unmarshal(paramsBytes, &params)
    if err != nil {{
        fmt.Printf("Error unmarshaling parameters: %v\\n", err)
        os.Exit(1)
    }}

    err = json.Unmarshal(skBytes, &sk)
    if err != nil {{
        fmt.Printf("Error unmarshaling secret key: %v\\n", err)
        os.Exit(1)
    }}

    // Create decoder and decryptor
    encoder := bgv.NewEncoder(params)
    decryptor := bgv.NewDecryptor(params, sk)

    // Read result data
    resultData, err := ioutil.ReadFile(resultFile)
    if err != nil {{
        fmt.Printf("Error reading result file: %v\\n", err)
        os.Exit(1)
    }}

    // Deserialize ciphertext
    ct := bgv.NewCiphertext(params, 1, params.MaxLevel())
    err = ct.UnmarshalBinary(resultData)
    if err != nil {{
        fmt.Printf("Error deserializing ciphertext: %v\\n", err)
        os.Exit(1)
    }}

    // Decrypt
    plaintext := decryptor.DecryptNew(ct)

    // Use the specific decryption function from the generated code
    result, err := {func_name}__decrypt__result0(encoder, plaintext)
    if err != nil {{
        fmt.Printf("Error decoding result: %v\\n", err)
        os.Exit(1)
    }}

    // Print result as a string that can be parsed by Python
    if len(result) == 1 {{
        fmt.Printf("%d", result[0])
    }} else {{
        // Format as a list
        parts := make([]string, len(result))
        for i, val := range result {{
            parts[i] = fmt.Sprintf("%d", val)
        }}
        fmt.Printf("[%s]", strings.Join(parts, ", "))
    }}
}}
"""
