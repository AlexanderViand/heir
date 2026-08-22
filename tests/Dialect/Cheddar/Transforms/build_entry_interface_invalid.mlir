// RUN: heir-opt --cheddar-build-entry-interface --split-input-file --verify-diagnostics %s

!ct = !cheddar.ciphertext
!context = !cheddar.context
!ui = !cheddar.user_interface

func.func @entry__setup() -> tensor<!context>
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} {
  %ctx = tensor.empty() : tensor<!context>
  return %ctx : tensor<!context>
}
// A tensor that reaches the server unencrypted cannot be handed over; the
// program still lowers without its encrypt facade (and, lacking a decrypt
// helper, without its decrypt facade).
// expected-warning@+2 {{entry input 1 reaches the server as cleartext}}
// expected-warning@+1 {{is missing the decryption helper for result 0}}
func.func @entry(%ctx: !context {cheddar.support = "context"}, %arg0: tensor<1x!ct>, %arg1: tensor<4xf32>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, tensor<4xf32>], result_types = [tensor<4xf32>], roles = ["entry", "server.evaluate"]}} {
  return %arg0 : tensor<1x!ct>
}
func.func @entry__encrypt__arg0(%ui: !ui {cheddar.support = "user_interface"}, %input: tensor<4xf32>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} {
  %ct = tensor.empty() : tensor<1x!ct>
  return %ct : tensor<1x!ct>
}

// -----

!ct = !cheddar.ciphertext
!context = !cheddar.context
!ui = !cheddar.user_interface

func.func @entry__setup() -> tensor<!context>
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} {
  %ctx = tensor.empty() : tensor<!context>
  return %ctx : tensor<!context>
}
// expected-warning@+1 {{is missing the decryption helper for result 0}}
func.func @entry(%ctx: !context {cheddar.support = "context"}, %arg0: tensor<1x!ct>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>], result_types = [tensor<4xf32>], roles = ["entry", "server.evaluate"]}} {
  return %arg0 : tensor<1x!ct>
}
func.func @entry__encrypt__arg0(%ui: !ui {cheddar.support = "user_interface"}, %input: tensor<4xf32>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} {
  %ct = tensor.empty() : tensor<1x!ct>
  return %ct : tensor<1x!ct>
}
