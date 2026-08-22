// RUN: heir-opt --cheddar-emitc-entry-interface %s | FileCheck %s

!ctx = !emitc.ptr<!emitc.opaque<"Context<word>">>
!boot_ctx = !emitc.ptr<!emitc.opaque<"BootContext<word>">>
!ctx_owner = !emitc.opaque<"std::shared_ptr<BootContext<word>>&">
!ctx_owner_const = !emitc.opaque<"const std::shared_ptr<BootContext<word>>&">
!encoder = !emitc.opaque<"const Encoder<word>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!evk = !emitc.opaque<"const EvaluationKey<word>&">
!evk_map = !emitc.opaque<"const EvkMap<word>&">
!ct = !emitc.array<1x!emitc.opaque<"Ciphertext<word>">>
!ct_const = !emitc.array<1x!emitc.opaque<"const Ciphertext<word>">>
!pt = !emitc.array<2x!emitc.opaque<"Plaintext<word>">>
!pt_const = !emitc.array<2x!emitc.opaque<"const Plaintext<word>">>
!dir = !emitc.opaque<"std::string_view">

func.func @entry__setup(
    %out: !ctx_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} {
  return
}

func.func @entry__keygen(
    %ctx: !ctx_owner_const,
    %out: !ui_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} {
  return
}

func.func @entry(
    %input0: !ct_const, %input1: !ct_const, %prepared: !pt_const,
    %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, tensor<2xf32>], result_types = [tensor<2xf32>], roles = ["entry"]}} {
  return
}

func.func @entry__encrypt__arg0(
    %ctx: !ctx {cheddar.support = "context"}, %encoder: !encoder {cheddar.support = "encoder"},
    %ui: !ui {cheddar.support = "user_interface"}, %input: !emitc.ptr<f32>,
    %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} {
  return
}

func.func @entry__encrypt__arg1(
    %ctx: !ctx {cheddar.support = "context"}, %encoder: !encoder {cheddar.support = "encoder"},
    %ui: !ui {cheddar.support = "user_interface"}, %input: !emitc.ptr<f32>,
    %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 1 : i64, roles = ["client.encrypt"]}} {
  return
}

func.func @entry__preprocessing(
    %ctx: !boot_ctx {cheddar.support = "boot_context"}, %encoder: !encoder {cheddar.support = "encoder"},
    %out: !pt {bufferize.result}, %dir: !dir {cheddar.support = "resource_dir"})
    attributes {heir.interface = {func_name = "entry", roles = ["server.preprocessing"]}} {
  %data = emitc.literal "nullptr" : !emitc.ptr<f32>
  call @outlined_layout(%data, %dir) : (!emitc.ptr<f32>, !dir) -> ()
  return
}

func.func @entry__preprocessed(
    %ctx: !boot_ctx {cheddar.support = "boot_context"}, %encoder: !encoder {cheddar.support = "encoder"},
    %evk: !evk {cheddar.support = "eval_key"}, %evk_map: !evk_map {cheddar.support = "evk_map"},
    %input0: !ct_const, %input1: !ct_const, %prepared: !pt_const,
    %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["server.evaluate"]}} {
  return
}

func.func @entry__decrypt__result0(
    %ctx: !ctx {cheddar.support = "context"}, %encoder: !encoder {cheddar.support = "encoder"},
    %ui: !ui {cheddar.support = "user_interface"},
    %input: !ct_const, %out: !emitc.ptr<f32> {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} {
  return
}

func.func private @outlined_layout(%input: !emitc.ptr<f32>,
                                   %dir: !dir {cheddar.support = "resource_dir"})
    attributes {heir.interface = {func_name = "entry", roles = ["client.pack"]}} {
  emitc.call_opaque "heir::loadResource"(%dir, %input) <{
    args = [0 : index, #emitc.opaque<"\22data/weights.bin\22">, 1 : index,
            #emitc.opaque<"4">],
    template_args = [f32]
  }> : (!dir, !emitc.ptr<f32>) -> ()
  return
}

// The facades cheddar-build-entry-interface generated, after lowering.
func.func @entry__encrypt_inputs(
    %ctx: !boot_ctx {cheddar.support = "boot_context"}, %ui: !ui {cheddar.support = "user_interface"},
    %input0: !emitc.ptr<f32> {cheddar.entry_input = 0 : i64},
    %input1: !emitc.ptr<f32> {cheddar.entry_input = 1 : i64},
    %out0: !ct {bufferize.result}, %out1: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.encrypt"]}} {
  emitc.call_opaque "entry__encrypt__arg0"() : () -> ()
  emitc.call_opaque "entry__encrypt__arg1"() : () -> ()
  return
}

func.func @entry__decrypt_outputs(
    %ctx: !boot_ctx {cheddar.support = "boot_context"}, %ui: !ui {cheddar.support = "user_interface"},
    %input: !ct_const, %out: !emitc.ptr<f32> {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.decrypt"]}} {
  emitc.call_opaque "entry__decrypt__result0"() : () -> ()
  return
}

func.func @entry__evaluate(
    %ctx: !boot_ctx {cheddar.support = "boot_context"}, %evk_map: !evk_map {cheddar.support = "evk_map"},
    %input0: !ct_const {cheddar.entry_input = 0 : i64}, %input1: !ct_const {cheddar.entry_input = 1 : i64},
    %prepared: !pt_const {cheddar.prepared}, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.evaluate"]}} {
  emitc.call_opaque "entry__preprocessed"() : () -> ()
  return
}

func.func @entry__preprocess(
    %ctx: !boot_ctx {cheddar.support = "boot_context"}, %dir: !dir {cheddar.support = "resource_dir"},
    %out: !pt {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.preprocess"]}} {
  emitc.call_opaque "entry__preprocessing"() : () -> ()
  return
}

// CHECK: emitc.file "header"
// CHECK: verbatim "#pragma once"
// CHECK: include <"tuple">
// CHECK: verbatim "namespace heir::generated::entry {"
// CHECK: verbatim "using Context = ::cheddar::BootContext<word>;"
// CHECK: verbatim "using PublicKey = const ::cheddar::EvkMap<word>*;"
// CHECK: verbatim "using Input0 = std::array<float, 4>;"
// CHECK: verbatim "using Input1 = std::array<float, 2>;"
// CHECK: verbatim "using CleartextInputs = std::tuple<Input0, Input1>;"
// CHECK: verbatim "using Output0 = std::array<float, 2>;"
// CHECK: class @KeyPair
// CHECK: field @storage : !emitc.opaque<"std::unique_ptr<UserInterface<word>>">
// CHECK: verbatim "using PreparedInputs = std::tuple<std::array<Plaintext<word>, 2>>;"
// CHECK: verbatim "using EncryptedInputs = std::tuple<std::array<Ciphertext<word>, 1>, std::array<Ciphertext<word>, 1>>;"
// CHECK: verbatim "using EncryptedOutputs = std::tuple<std::array<Ciphertext<word>, 1>>;"
// CHECK: func @Setup() -> !emitc.opaque<"std::shared_ptr<Context>">

// CHECK: func @Encrypt(!emitc.opaque<"Context&">, !emitc.opaque<"SecretKey">, !emitc.opaque<"CleartextInputs&">) -> !emitc.opaque<"EncryptedInputs">
// CHECK: emitc.file "source"
// CHECK: include "entry.h"
// CHECK: verbatim "namespace heir::generated::detail {"
// CHECK: func.func private @entry__setup
// CHECK: func.func private @entry__encrypt__arg0
// CHECK: func.func private @entry__encrypt__arg1
// CHECK: func.func private @entry__preprocessing
// CHECK: func.func private @outlined_layout
// CHECK: func.func private @entry__encrypt_inputs
// CHECK-NOT: func.func private @entry(
// CHECK: func @Setup() -> !emitc.opaque<"std::shared_ptr<Context>">
// CHECK: !emitc.lvalue<!emitc.opaque<"std::shared_ptr<Context>">>
// CHECK: call_opaque "::heir::generated::detail::entry__setup"
// CHECK: call_opaque "heir::getPointer"
// CHECK: func @Encrypt
// CHECK: %[[CTX:.*]] = call_opaque "std::addressof"
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"Context">>
// CHECK: %[[IN0:.*]] = call_opaque "std::get<0>"
// CHECK: %[[DATA0:.*]] = call_opaque "heir::data"(%[[IN0]])
// CHECK: %[[IN1:.*]] = call_opaque "std::get<1>"
// CHECK: %[[DATA1:.*]] = call_opaque "heir::data"(%[[IN1]])
// CHECK: call_opaque "::heir::generated::detail::entry__encrypt_inputs"(%[[CTX]], %{{.*}}, %[[DATA0]], %[[DATA1]], %{{.*}}, %{{.*}})
// CHECK: call_opaque "heir::pack"
// CHECK: func @Preprocess(%{{.*}}: !emitc.opaque<"Context&">, %[[KEYS:.*]]: !emitc.opaque<"PublicKey">, %[[DIR:.*]]: !emitc.opaque<"std::string_view">)
// CHECK: call_opaque "::heir::generated::detail::entry__preprocess"(%{{.*}}, %[[DIR]], %{{.*}})
// CHECK: call_opaque "heir::pack"
// CHECK: func @Evaluate
// CHECK: call_opaque "heir::deref"
// CHECK: call_opaque "std::get<0>"
// CHECK: call_opaque "::heir::generated::detail::entry__evaluate"
// CHECK: call_opaque "heir::pack"
// CHECK: func @Decrypt
// CHECK: call_opaque "::heir::generated::detail::entry__decrypt_outputs"
// CHECK: call_opaque "std::move"
