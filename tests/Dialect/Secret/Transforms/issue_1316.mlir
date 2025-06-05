// RUN: heir-opt --secret-distribute-generic %s

// Smoke test to ensure the lowering can handle multiple results of the generic.
#alignment = #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [4294991873], P = [4295049217], plaintextModulus = 65537>, scheme.bgv} {
  func.func @foo(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>, tensor_ext.original_type = #original_type}, %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>, tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>, tensor_ext.original_type = #original_type}, !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>, tensor_ext.original_type = #original_type}) attributes {mgmt.openfhe_params = #mgmt.openfhe_params<evalAddCount = 2, keySwitchCount = 0>} {
    %0:2 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>}, %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>}) {
    ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
      %1 = arith.addi %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>} : tensor<1024xi16>
      secret.yield %1, %1 : tensor<1024xi16>, tensor<1024xi16>
    } -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>}, !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 1>})
    return %0#0, %0#1 : !secret.secret<tensor<1024xi16>>, !secret.secret<tensor<1024xi16>>
  }
}
