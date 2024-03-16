// RUN: stablehlo-quant-opt %s -stablehlo-merge-activation-transpose \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s

// Tests that an `add(transpose(arg0), arg1)` pattern is converted to
// `transpose(add(arg0, transpose(arg1)))`. The transpose in the activation is
// merged into `stablehlo.add` and extra transpose ops are inserted for the RHS
// and the result to match the shapes of the operand and users.

// CHECK-LABEL: add_with_activation_transpose
func.func @add_with_activation_transpose(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x3x3x4xf32>) -> tensor<1x4x3x3xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<1x4x3x3xf32>
  return %1 : tensor<1x4x3x3xf32>
}
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x3x4xf32>, %[[ARG_1:.+]]: tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>
// CHECK-DAG: %[[TRANSPOSE_0:.+]] = stablehlo.transpose %[[ARG_1]], dims = [0, 2, 3, 1] : (tensor<1x4x3x3xf32>) -> tensor<1x3x3x4xf32>

// Check that the shape of the add is changed to reflect the merged transpose.
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[ARG_0]], %[[TRANSPOSE_0]] : tensor<1x3x3x4xf32>
// CHECK: %[[TRANSPOSE_1:.+]] = stablehlo.transpose
// CHECK: return %[[TRANSPOSE_1]]

// -----

// [No change] Tests that the activation transpose whose permutation is not
// `[0, 3, 1, 2]` is not merged to `stablehlo.add`.

// CHECK-LABEL: add_with_activation_transpose_permutation_mismatch
func.func @add_with_activation_transpose_permutation_mismatch(
      %arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x3x2x4xf32>) -> tensor<1x3x2x4xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 2, 1, 3] : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<1x3x2x4xf32>
  return %1 : tensor<1x3x2x4xf32>
}
// CHECK: %[[TRANSPOSE_0:.+]] = stablehlo.transpose
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[TRANSPOSE_0]], {{.*}}
// CHECK: return %[[ADD_0]]

// -----

// [No change] Tests that the activation transpose whose rank is not 4 is not
// merged to `stablehlo.add`.

// CHECK-LABEL: add_with_activation_transpose_rank_two
func.func @add_with_activation_transpose_rank_two(
      %arg0: tensor<1x2xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x1xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
// CHECK: %[[TRANSPOSE_0:.+]] = stablehlo.transpose
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[TRANSPOSE_0]], {{.*}}
// CHECK: return %[[ADD_0]]
