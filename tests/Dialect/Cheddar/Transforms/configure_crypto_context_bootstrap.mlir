// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main %s | FileCheck %s --check-prefix=CONFIG
// RUN: heir-opt --cheddar-configure-crypto-context=entry-function=main --cheddar-bufferize --fold-memref-alias-ops --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s --check-prefix=EMITC

!boot_context = !cheddar.boot_context
!ciphertext = !cheddar.ciphertext
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

module attributes {
  cheddar.boot.num_cts = 4 : i64,
  cheddar.boot.num_stc = 3 : i64,
  ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329, 1099525128193, 1099526176769, 1099529060353, 1099535220737, 1099536138241, 1099537580033, 1099538104321, 1099540725761, 1099540856833, 1099543085057, 36028797019488257, 36028797023420417, 36028797024206849, 36028797025124353, 36028797032202241, 36028797033644033, 36028797037576193, 36028797048324097, 36028797048586241, 36028797049896961, 36028797051863041, 36028797053698049, 36028797054222337], P = [72057594038321153, 72057594040680449, 72057594042646529, 72057594047889409, 72057594057195521, 72057594058375169, 72057594058899457], logDefaultScale = 40>,
  scheme.actual_slot_count = 32768 : i64,
  scheme.requested_slot_count = 8 : i64
} {
  func.func @main(%ctx: !boot_context, %ui: !ui, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
    %rotDest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %rotated = cheddar.hrot %ctx, %evk, %ct, %rotDest {level = 13 : i64, static_distance = 7 : i64} : (!boot_context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.boot %ctx, %rotated, %evk, %dest : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}

// CONFIG: func.func @main__setup
// CONFIG: cheddar.make_parameter
// CONFIG-SAME: defaultEncryptionLevel = 13
// CONFIG-SAME: denseHammingWeight = 32768
// CONFIG-SAME: sparseHammingWeight = 32
// CONFIG: cheddar.create_boot_context
// CONFIG-SAME: logMessageRatio = 5
// CONFIG-SAME: numCtsLevels = 4
// CONFIG-SAME: numStcLevels = 3
// CONFIG: func.func @main__keygen
// CONFIG: cheddar.create_user_interface
// CONFIG: cheddar.prepare_rot_key
// CONFIG-SAME: distance = 7
// CONFIG-SAME: maxLevel = 13
// CONFIG: cheddar.prepare_bootstrap
// CONFIG-SAME: numSlots = 256
// CONFIG: func.func @main__configure
// CONFIG: call @main__setup
// CONFIG: call @main__keygen

// EMITC: func.func @main__setup
// EMITC-SAME: !emitc.opaque<"std::shared_ptr<BootContext<word>>&">
// EMITC: emitc.verbatim "static Parameter<word> cheddar_param
// EMITC: emitc.verbatim "cheddar_param.SetDenseHammingWeight(32768);"
// EMITC: emitc.verbatim "cheddar_param.SetSparseHammingWeight(32);"
// EMITC: emitc.verbatim "{} = BootContext<word>::Create({}, BootParameter({}.max_level_, 4, 3, 5));"
// EMITC: func.func @main__keygen
// EMITC-SAME: !emitc.opaque<"const std::shared_ptr<BootContext<word>>&">
// EMITC-SAME: !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
// EMITC: emitc.verbatim "{}->PrepareRotationKey(7, 13);"
// EMITC: emitc.verbatim "{}->PrepareEvalMod();"
// EMITC: emitc.verbatim "{}->PrepareEvalSpecialFFT(256, BootVariant::kImaginaryRemoving);"
// EMITC: emitc.verbatim "EvkRequest boot_evk_req;"
// EMITC: emitc.verbatim "{}->PrepareRotationKey(boot_evk_req);"
// EMITC: func.func @main__configure
// EMITC: emitc.call_opaque "main__setup"
// EMITC: emitc.call_opaque "main__keygen"
