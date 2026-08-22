// RUN: heir-opt %s --cheddar-emitc-entry-interface > %t
// RUN: heir-translate %t --mlir-to-cpp --file-id=source | FileCheck %s
// RUN: heir-translate %t --mlir-to-cpp --file-id=header | FileCheck %s --check-prefix=HEADER

// The facades own a plain Context, which names the public context type.
// HEADER: using Context = ::cheddar::Context<word>;
// HEADER: std::shared_ptr<Context> Setup();
// HEADER: EncryptedOutputs Evaluate(

!ctx = !emitc.ptr<!emitc.opaque<"Context<word>">>
!ctx_owner = !emitc.opaque<"std::shared_ptr<Context<word>>&">
!ctx_const = !emitc.opaque<"const std::shared_ptr<Context<word>>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!evk_map = !emitc.opaque<"const EvkMap<word>&">
!ct = !emitc.array<1x!emitc.opaque<"Ciphertext<word>">>
!ct_const = !emitc.array<1x!emitc.opaque<"const Ciphertext<word>">>
!float = !emitc.ptr<f32>

func.func @entry__setup(%out: !ctx_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
func.func @entry__keygen(%ctx: !ctx_const, %out: !ui_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
func.func @entry(%input: !ct_const, %flag: i1,
    %zero: !ct_const {client.enc_zero_arg = {func_name = "entry", index = 0 : i64}},
    %out0: !ct {bufferize.result}, %out1: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, i1], result_types = [tensor<4xf32>, tensor<4xf32>], roles = ["entry", "server.evaluate"]}} { return }
func.func @encrypt(%ui: !ui {cheddar.support = "user_interface"}, %input: !float, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} { return }
func.func @encrypt_zero(%ui: !ui {cheddar.support = "user_interface"}, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt_zero"]}} { return }
func.func @decrypt0(%ui: !ui {cheddar.support = "user_interface"}, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} { return }
func.func @decrypt1(%ui: !ui {cheddar.support = "user_interface"}, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 1 : i64, roles = ["client.decrypt"]}} { return }

// The scalar flag reaches the server as cleartext: the encrypt facade skips
// it and the wrapper copies it into its field.
func.func @entry__encrypt_inputs(%ctx: !ctx {cheddar.support = "context"}, %ui: !ui {cheddar.support = "user_interface"},
    %input: !float {cheddar.entry_input = 0 : i64},
    %out: !ct {bufferize.result}, %zero: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.encrypt"]}} {
  emitc.call_opaque "encrypt"() : () -> ()
  emitc.call_opaque "encrypt_zero"() : () -> ()
  return
}
func.func @entry__decrypt_outputs(%ctx: !ctx {cheddar.support = "context"}, %ui: !ui {cheddar.support = "user_interface"},
    %input0: !ct_const, %input1: !ct_const,
    %out0: !float {bufferize.result}, %out1: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.decrypt"]}} {
  emitc.call_opaque "decrypt0"() : () -> ()
  emitc.call_opaque "decrypt1"() : () -> ()
  return
}
func.func @entry__evaluate(%ctx: !ctx {cheddar.support = "context"}, %evk_map: !evk_map {cheddar.support = "evk_map"},
    %input: !ct_const {cheddar.entry_input = 0 : i64}, %flag: i1 {cheddar.entry_input = 1 : i64}, %zero: !ct_const,
    %out0: !ct {bufferize.result}, %out1: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.evaluate"]}} {
  emitc.call_opaque "entry"() : () -> ()
  return
}

// CHECK: EncryptedInputs Encrypt
// CHECK: std::get<1>
// CHECK: static_cast<bool>
// CHECK: std::get<0>
// CHECK: ::heir::generated::detail::entry__encrypt_inputs(
// CHECK: heir::pack<EncryptedInputs>(
// CHECK: Outputs Decrypt
// CHECK: ::heir::generated::detail::entry__decrypt_outputs(
// CHECK: heir::pack<Outputs>(
