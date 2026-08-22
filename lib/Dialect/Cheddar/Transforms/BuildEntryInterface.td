#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_BUILDENTRYINTERFACE_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_BUILDENTRYINTERFACE_TD_

include "mlir/Pass/PassBase.td"

def CheddarBuildEntryInterface : Pass<"cheddar-build-entry-interface", "::mlir::ModuleOp"> {
  let summary = "Build the public entry facades over the client/server helpers";

  let description = [{
    Generates one public function per interface step of the entry point whose
    roles `--add-client-interface`, `--split-preprocessing` and
    `--cheddar-configure-crypto-context` recorded:

    * `<entry>__encrypt_inputs` encrypts the cleartext inputs (and the
      encrypted zeros) into the entry's ciphertext arguments,
    * `<entry>__preprocess` runs the server-side plaintext preprocessing,
    * `<entry>__evaluate` evaluates the ciphertext workload,
    * `<entry>__decrypt_outputs` decrypts the entry's results.

    A facade takes only the support values a caller owns, tagged
    `cheddar.support`: its context first,
    the key material (a user interface, or an evaluation-key map when no step
    needs the secret), a debug handler and the resource directory. The
    encoder, evaluation-key map and multiplication key the helpers need are
    derived inside the facade with `cheddar.get_encoder`,
    `cheddar.get_evk_map` and `cheddar.get_mult_key`. Data arguments fed from
    a cleartext entry argument carry `cheddar.entry_input`; preprocessed
    values carry `cheddar.prepared`.

    The facades are ordinary functions, so bufferization and the EmitC
    conversion lower them like the helpers, and a language binding only
    packs and unpacks its aggregates around them.
  }];

  let options = [
    Option<"entryFunction", "entry-function", "std::string", "\"\"",
           "Entry function to build facades for (auto-detected if empty)">,
  ];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_BUILDENTRYINTERFACE_TD_
