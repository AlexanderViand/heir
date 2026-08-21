// RUN: heir-opt --cheddar-emitc-entry-interface %s | FileCheck %s

!ctx = !emitc.ptr<!emitc.opaque<"Context<word>">>
!ctx_owner = !emitc.opaque<"std::shared_ptr<Context<word>>&">
!ctx_owner_const = !emitc.opaque<"const std::shared_ptr<Context<word>>&">
!encoder = !emitc.opaque<"const Encoder<word>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!evk = !emitc.opaque<"const EvaluationKey<word>&">
!evk_map = !emitc.opaque<"const EvkMap<word>&">
!ct = !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
!ct_const = !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
!pt = !emitc.opaque<"std::array<Plaintext<word>, 2>&">
!pt_const = !emitc.opaque<"const std::array<Plaintext<word>, 2>&">

func.func @entry__setup(
    %out: !ctx_owner {bufferize.result})
    attributes {client.setup_func = {func_name = "entry"}} {
  return
}

func.func @entry__keygen(
    %ctx: !ctx_owner_const,
    %out: !ui_owner {bufferize.result})
    attributes {client.keygen_func = {func_name = "entry"}} {
  return
}

func.func @entry(
    %input: !ct_const, %prepared: !pt_const,
    %out: !ct {bufferize.result})
    attributes {
      heir.entry_func = {func_name = "entry"},
      heir.entry_input_types = [tensor<4xf32>],
      heir.entry_result_types = [tensor<2xf32>]
    } {
  return
}

func.func @entry__encrypt__arg0(
    %ctx: !ctx, %encoder: !encoder, %ui: !ui, %input: !emitc.ptr<f32>,
    %out: !ct {bufferize.result})
    attributes {
      client.enc_func = {func_name = "entry", index = 0 : i64}
    } {
  return
}

func.func @entry__preprocessing(
    %ctx: !ctx, %encoder: !encoder, %out: !pt {bufferize.result})
    attributes {server.preprocessing_func = {func_name = "entry"}} {
  return
}

func.func @entry__preprocessed(
    %ctx: !ctx, %encoder: !encoder, %ui: !ui, %evk: !evk,
    %evk_map: !evk_map, %input: !ct_const, %prepared: !pt_const,
    %out: !ct {bufferize.result})
    attributes {server.evaluate_func = {func_name = "entry"}} {
  return
}

func.func @entry__decrypt__result0(
    %ctx: !ctx, %encoder: !encoder, %ui: !ui, %evk: !evk,
    %input: !ct_const, %out: !emitc.ptr<f32> {bufferize.result})
    attributes {
      client.dec_func = {func_name = "entry", index = 0 : i64}
    } {
  return
}

// CHECK: emitc.file "header"
// CHECK: verbatim "#pragma once"
// CHECK: include <"tuple">
// CHECK: verbatim "namespace heir::generated::entry {"
// CHECK: verbatim "using Input0 = std::array<float, 4>;"
// CHECK: verbatim "using Output0 = std::array<float, 2>;"
// CHECK: class @KeyPair
// CHECK: field @storage : !emitc.opaque<"std::unique_ptr<UserInterface<word>>">
// CHECK: verbatim "using PreparedInputs = std::tuple<std::array<Plaintext<word>, 2>>;"
// CHECK: verbatim "using EncryptedInputs = std::tuple<std::array<Ciphertext<word>, 1>>;"
// CHECK: func @Setup() -> !emitc.opaque<"std::shared_ptr<Context>">
// CHECK: func @Encrypt(!emitc.opaque<"Context&">, !emitc.opaque<"SecretKey">, !emitc.opaque<"Input0&">) -> !emitc.opaque<"EncryptedInputs">
// CHECK: emitc.file "source"
// CHECK: include "entry.h"
// CHECK: verbatim "namespace heir::generated::detail {"
// CHECK: func.func private @entry__setup
// CHECK: func.func private @entry__encrypt__arg0
// CHECK: func @Setup() -> !emitc.opaque<"std::shared_ptr<Context>">
// CHECK: !emitc.lvalue<!emitc.opaque<"std::shared_ptr<Context>">>
// CHECK: call_opaque "::heir::generated::detail::entry__setup"
// CHECK: call_opaque "heir::getPointer"
// CHECK: func @Encrypt
// CHECK: !emitc.ptr<!emitc.opaque<"Context">>
// CHECK: call_opaque "heir::data"
// CHECK: call_opaque "::heir::generated::detail::entry__encrypt__arg0"
// CHECK: func @Evaluate
// CHECK: call_opaque "::heir::generated::detail::entry__preprocessed"
// CHECK: func @Decrypt
// CHECK: call_opaque "::heir::generated::detail::entry__decrypt__result0"
