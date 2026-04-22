// RUN: heir-opt %s "--mlir-to-ckks=ciphertext-degree=4 modulus-switch-before-first-mul=true first-mod-bits=59 scaling-mod-bits=45 ckks-reconcile-policy=canonical-per-level" --scheme-to-cheddar | FileCheck %s
// RUN: heir-opt %s "--mlir-to-ckks=ciphertext-degree=4 modulus-switch-before-first-mul=true first-mod-bits=59 scaling-mod-bits=45 ckks-reconcile-policy=canonical-per-level ckks-scale-policy=precise" --scheme-to-cheddar | FileCheck %s

module attributes {backend.cheddar, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  func.func @cross_level(%base: tensor<4xf32> {secret.secret}, %add: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
    %base0 = arith.addf %base, %add : tensor<4xf32>
    %mul1 = arith.mulf %base0, %base0 : tensor<4xf32>
    %base1 = arith.addf %mul1, %add : tensor<4xf32>
    %mul2 = arith.mulf %base1, %base1 : tensor<4xf32>
    %base2 = arith.addf %mul2, %add : tensor<4xf32>
    %mul3 = arith.mulf %base2, %base2 : tensor<4xf32>
    %base3 = arith.addf %mul3, %add : tensor<4xf32>
    return %base3 : tensor<4xf32>
  }
}

// CHECK: func.func @cross_level(
// CHECK: %[[LHS:[^ ]+]] = tensor.extract %arg0[%{{.*}}] : tensor<1x!ct>
// CHECK: %[[RHS:[^ ]+]] = tensor.extract %arg1[%{{.*}}] : tensor<1x!ct>
// CHECK: %[[SUM0:.*]] = cheddar.add %ctx, %[[LHS]], %[[RHS]]
// CHECK-NEXT: %[[DOWN0:.*]] = cheddar.level_down %ctx, %[[SUM0]] {targetLevel = 3 : i64}
// CHECK-NEXT: %[[MUL1:.*]] = cheddar.mult %ctx, %[[DOWN0]], %[[DOWN0]]
// CHECK-NEXT: %[[KEY1:.*]] = cheddar.get_mult_key %ui
// CHECK-NEXT: %[[RELIN1:.*]] = cheddar.relinearize %ctx, %[[MUL1]], %[[KEY1]]
// CHECK-NEXT: %[[DOWN1:.*]] = cheddar.level_down %ctx, %[[RHS]] {targetLevel = 3 : i64}
// CHECK-NEXT: %[[PT1:.*]] = cheddar.encode %encoder, %{{.*}} {level = 3 : i64, scale = 45 : i64}
// CHECK-NEXT: %[[ADJ1:.*]] = cheddar.mult_plain %ctx, %[[DOWN1]], %[[PT1]]
// CHECK-NEXT: %[[ADD1:.*]] = cheddar.add %ctx, %[[RELIN1]], %[[ADJ1]]
// CHECK: %[[RED1:.*]] = cheddar.rescale %ctx, %[[ADD1]]
// CHECK-NEXT: %[[MUL2:.*]] = cheddar.mult %ctx, %[[RED1]], %[[RED1]]
// CHECK-NEXT: %[[KEY2:.*]] = cheddar.get_mult_key %ui
// CHECK-NEXT: %[[RELIN2:.*]] = cheddar.relinearize %ctx, %[[MUL2]], %[[KEY2]]
// CHECK-NEXT: %[[DOWN2:.*]] = cheddar.level_down %ctx, %[[RHS]] {targetLevel = 2 : i64}
// CHECK-NEXT: %[[PT2:.*]] = cheddar.encode %encoder, %{{.*}} {level = 2 : i64, scale = 45 : i64}
// CHECK-NEXT: %[[ADJ2:.*]] = cheddar.mult_plain %ctx, %[[DOWN2]], %[[PT2]]
// CHECK-NEXT: %[[ADD2:.*]] = cheddar.add %ctx, %[[RELIN2]], %[[ADJ2]]
// CHECK: %[[RED2:.*]] = cheddar.rescale %ctx, %[[ADD2]]
// CHECK-NEXT: %[[MUL3:.*]] = cheddar.mult %ctx, %[[RED2]], %[[RED2]]
// CHECK-NEXT: %[[KEY3:.*]] = cheddar.get_mult_key %ui
// CHECK-NEXT: %[[RELIN3:.*]] = cheddar.relinearize %ctx, %[[MUL3]], %[[KEY3]]
// CHECK-NEXT: %[[DOWN3:.*]] = cheddar.level_down %ctx, %[[RHS]] {targetLevel = 1 : i64}
// CHECK-NEXT: %[[PT3:.*]] = cheddar.encode %encoder, %{{.*}} {level = 1 : i64, scale = 45 : i64}
// CHECK-NEXT: %[[ADJ3:.*]] = cheddar.mult_plain %ctx, %[[DOWN3]], %[[PT3]]
// CHECK-NEXT: %[[ADD3:.*]] = cheddar.add %ctx, %[[RELIN3]], %[[ADJ3]]
