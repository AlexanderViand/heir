// RUN: heir-opt --cheddar-build-entry-interface %s | FileCheck %s

// Facades over a CHEDDAR-runtime entry: the client owns a context and its
// UserInterface, the server a context and the evaluation-key map; the
// encoder and multiplication key are derived inside.

!ct = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!ui = !cheddar.user_interface
!evk = !cheddar.eval_key

func.func @entry__setup() -> tensor<!context>
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} {
  %ctx = tensor.empty() : tensor<!context>
  return %ctx : tensor<!context>
}
func.func @entry__keygen(%ctx: tensor<!context>) -> (tensor<!context>, tensor<!ui>)
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} {
  %ui = tensor.empty() : tensor<!ui>
  return %ctx, %ui : tensor<!context>, tensor<!ui>
}
func.func @entry(%ctx: !context {cheddar.support = "context"}, %encoder: !encoder {cheddar.support = "encoder"},
                 %evk: !evk {cheddar.support = "eval_key"},
                 %arg0: tensor<1x!ct>, %arg1: tensor<1x!ct>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, tensor<4xf32>], result_types = [tensor<4xf32>], roles = ["entry", "server.evaluate"]}} {
  return %arg0 : tensor<1x!ct>
}
func.func @entry__encrypt__arg0(%encoder: !encoder {cheddar.support = "encoder"}, %ui: !ui {cheddar.support = "user_interface"}, %input: tensor<4xf32>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} {
  %ct = tensor.empty() : tensor<1x!ct>
  return %ct : tensor<1x!ct>
}
func.func @entry__encrypt__arg1(%encoder: !encoder {cheddar.support = "encoder"}, %ui: !ui {cheddar.support = "user_interface"}, %input: tensor<4xf32>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", index = 1 : i64, roles = ["client.encrypt"]}} {
  %ct = tensor.empty() : tensor<1x!ct>
  return %ct : tensor<1x!ct>
}
func.func @entry__decrypt__result0(%encoder: !encoder {cheddar.support = "encoder"}, %ui: !ui {cheddar.support = "user_interface"}, %input: tensor<1x!ct>) -> tensor<4xf32>
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  return %cst : tensor<4xf32>
}

// CHECK: func.func @entry__encrypt_inputs(%[[CTX:[^:]+]]: !context {cheddar.support = "context"}, %[[UI:[^:]+]]: !user_interface {cheddar.support = "user_interface"}, %[[IN0:[^:]+]]: tensor<4xf32> {cheddar.entry_input = 0 : i64}, %[[IN1:[^:]+]]: tensor<4xf32> {cheddar.entry_input = 1 : i64}) -> (tensor<1x!ciphertext>, tensor<1x!ciphertext>)
// CHECK-SAME: roles = ["facade.encrypt"]
// CHECK: %[[ENC:.*]] = cheddar.get_encoder %[[CTX]]
// CHECK: %[[CT0:.*]] = call @entry__encrypt__arg0(%[[ENC]], %[[UI]], %[[IN0]])
// CHECK: %[[CT1:.*]] = call @entry__encrypt__arg1(%[[ENC]], %[[UI]], %[[IN1]])
// CHECK: return %[[CT0]], %[[CT1]]

// CHECK: func.func @entry__decrypt_outputs(%[[CTX:[^:]+]]: !context {cheddar.support = "context"}, %[[UI:[^:]+]]: !user_interface {cheddar.support = "user_interface"}, %[[CT:[^:]+]]: tensor<1x!ciphertext>) -> tensor<4xf32>
// CHECK-SAME: roles = ["facade.decrypt"]
// CHECK: %[[ENC:.*]] = cheddar.get_encoder %[[CTX]]
// CHECK: %[[OUT:.*]] = call @entry__decrypt__result0(%[[ENC]], %[[UI]], %[[CT]])
// CHECK: return %[[OUT]]

// The server never sees the secret: it owns the evaluation-key map and
// derives the multiplication key from it.
// CHECK: func.func @entry__evaluate(%[[CTX:[^:]+]]: !context {cheddar.support = "context"}, %[[MAP:[^:]+]]: !evk_map {cheddar.support = "evk_map"}, %[[A0:[^:]+]]: tensor<1x!ciphertext> {cheddar.entry_input = 0 : i64}, %[[A1:[^:]+]]: tensor<1x!ciphertext> {cheddar.entry_input = 1 : i64}) -> tensor<1x!ciphertext>
// CHECK-SAME: roles = ["facade.evaluate"]
// CHECK: %[[ENC:.*]] = cheddar.get_encoder %[[CTX]]
// CHECK: %[[KEY:.*]] = cheddar.get_mult_key %[[MAP]], %[[CTX]]
// CHECK: %[[OUT:.*]] = call @entry(%[[CTX]], %[[ENC]], %[[KEY]], %[[A0]], %[[A1]])
// CHECK: return %[[OUT]]
// CHECK-NOT: func.func @entry__preprocess
