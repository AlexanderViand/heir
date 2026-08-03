#include "lib/Utils/Layout/Hoisting.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

// ISL
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

using llvm::SmallVector;
using presburger::BoundType;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

presburger::IntegerRelation hoistConversionThroughMatvec(
    const IntegerRelation& matrixLayout, const IntegerRelation& fromVecLayout,
    const IntegerRelation& toVecLayout) {
  // The intuition for this function is that the conversion from fromVecLayout
  // to toVecLayout implies some transformation of the slot ordering of the
  // packed vector. The kernels we support have matrix layouts for which the
  // packed slots of a ciphertext track the columns of the packed vector.
  // So we need to apply the same transformation of the packed vector slots
  // to the packed matrix slots.
  //
  // This function works in two steps:
  //
  // 1. Compute the inferred re-packing relation of vector slots (i.e., a
  // relation (ct, slots) -> (ct, slots))
  //
  // 2. Compose (1) with the (ct, slot) dims of the matrix packing.

  IntegerRelation fromClone(fromVecLayout);
  IntegerRelation toClone(toVecLayout);

  // Project out the ciphertext dimension, though this will need to change when
  // we get a larger vector than can fit in one ciphertext. This is where the
  // assumption about the vector packing fitting into one ciphertext comes from.
  assert(fromClone.getConstantBoundOnDimSize64(1).value() == 1);
  assert(toClone.getConstantBoundOnDimSize64(1).value() == 1);
  fromClone.projectOut(1);
  toClone.projectOut(1);

  fromClone.inverse();
  fromClone.compose(toClone);
  fromClone.removeRedundantConstraints();
  fromClone.simplify();

  // At this stage, fromClone models the re-packing relation on the vector
  // (slots) -> (slots). Put the ct dim back in with bounds from the matrix
  // layout, and equality between the domain ct var and the range ct var (so
  // slots can only be rotated within a single ciphertext).
  std::optional<int64_t> ctUb = matrixLayout.getConstantBound64(
      BoundType::UB, matrixLayout.getVarKindOffset(VarKind::Range));
  std::optional<int64_t> ctLb = matrixLayout.getConstantBound64(
      BoundType::LB, matrixLayout.getVarKindOffset(VarKind::Range));
  prependPassthroughDim(fromClone, ctLb, ctUb);

  // At this point the re-packing relation should look something like
  //
  // {
  //   [i0, i1] -> [o0, o1] :     (ct, slot) -> (ct, slot)
  //   (3 - i1 + o1) mod 8 = 0    slot transformation from vec conversion
  //   and i0 = o0                ct equality constraint
  //   and 0 <= i0 <= 8           bounds on ct dim from matrixLayout
  //   and 0 <= o0 <= 8
  //   and 0 <= i1 <= 15          bounds on slot dim
  //   and 0 <= o1 <= 15
  // }
  //
  IntegerRelation result(matrixLayout);
  result.compose(fromClone);
  return result;
}

presburger::IntegerRelation absorbVectorLayoutIntoMatrix(
    const IntegerRelation& matrixLayout, const IntegerRelation& vecLayout) {
  assert(vecLayout.getNumDomainVars() == 1 &&
         "expected a vector (1-D domain) layout");
  assert(vecLayout.getNumRangeVars() == 2 &&
         "expected a (ct, slot) range layout");
  assert(matrixLayout.getNumDomainVars() == 2 &&
         "expected a matrix (2-D domain) layout");

  // Build the lifted relation (row, j) -> (row, slot_V(j)) from the vector
  // packing (j) -> (ct = 0, slot).
  IntegerRelation lift(vecLayout);
  // Drop the ct range var (the vector occupies a single ciphertext).
  lift.projectOut(lift.getVarKindOffset(VarKind::Range), 1);
  // Insert the row coordinate as a pass-through on both sides.
  lift.insertVar(VarKind::Domain, 0, 1);
  lift.insertVar(VarKind::Range, 0, 1);
  SmallVector<int64_t> rowEq(lift.getNumCols(), 0);
  rowEq[lift.getVarKindOffset(VarKind::Domain)] = 1;
  rowEq[lift.getVarKindOffset(VarKind::Range)] = -1;
  lift.addEquality(rowEq);

  // result = matrixLayout ∘ lift: (row, j) -> matrixLayout(row, slot_V(j)).
  IntegerRelation result(lift);
  result.compose(matrixLayout);
  result.removeRedundantConstraints();
  result.simplify();
  return result;
}

FailureOr<SmallVector<int64_t>> getMatvecInputSlots(
    RankedTensorType matrixType, const IntegerRelation& inputLayout,
    const IntegerRelation& outputLayout, int64_t ciphertextSize) {
  if (matrixType.getRank() != 2 || inputLayout.getNumDomainVars() != 1 ||
      outputLayout.getNumDomainVars() != 1 ||
      inputLayout.getNumRangeVars() != 2 ||
      outputLayout.getNumRangeVars() != 2) {
    return failure();
  }

  unsigned inputCt = inputLayout.getVarKindOffset(VarKind::Range);
  unsigned outputCt = outputLayout.getVarKindOffset(VarKind::Range);
  auto inputCtLb = inputLayout.getConstantBound64(BoundType::LB, inputCt);
  auto inputCtUb = inputLayout.getConstantBound64(BoundType::UB, inputCt);
  auto outputCtLb = outputLayout.getConstantBound64(BoundType::LB, outputCt);
  auto outputCtUb = outputLayout.getConstantBound64(BoundType::UB, outputCt);
  if (!inputCtLb.has_value() || !inputCtUb.has_value() ||
      inputCtLb.value() != 0 || inputCtUb.value() != 0 ||
      !outputCtLb.has_value() || !outputCtUb.has_value() ||
      outputCtLb.value() != 0 || outputCtUb.value() != 0) {
    return failure();
  }

  int64_t rows = matrixType.getDimSize(0);
  int64_t cols = matrixType.getDimSize(1);
  if (rows <= 0 || cols <= 0 || ciphertextSize <= 0) return failure();

  int64_t paddedRows = nextPowerOfTwo(rows);
  int64_t paddedCols = nextPowerOfTwo(cols);
  int64_t numDiagonals = std::min(paddedRows, paddedCols);

  std::vector<std::vector<int64_t>> inputSlots(cols);
  PointPairCollector inputPoints(/*domainDims=*/1, /*rangeDims=*/2);
  enumeratePoints(inputLayout, inputPoints);
  for (const auto& [domain, range] : inputPoints.points) {
    if (domain[0] < 0 || domain[0] >= cols || range[0] != 0 || range[1] < 0 ||
        range[1] >= ciphertextSize) {
      return failure();
    }
    inputSlots[domain[0]].push_back(range[1]);
  }

  std::vector<std::vector<int64_t>> outputSlots(rows);
  PointPairCollector outputPoints(/*domainDims=*/1, /*rangeDims=*/2);
  enumeratePoints(outputLayout, outputPoints);
  for (const auto& [domain, range] : outputPoints.points) {
    if (domain[0] < 0 || domain[0] >= rows || range[0] != 0 || range[1] < 0 ||
        range[1] >= ciphertextSize) {
      return failure();
    }
    outputSlots[domain[0]].push_back(range[1]);
  }

  if (paddedRows > paddedCols || ciphertextSize % paddedCols != 0) {
    return failure();
  }

  SmallVector<int64_t> inputSlotMap(cols + 1, 0);
  for (int64_t col = 0; col < cols; ++col) {
    llvm::sort(inputSlots[col]);
    inputSlots[col].erase(
        std::unique(inputSlots[col].begin(), inputSlots[col].end()),
        inputSlots[col].end());
    if (inputSlots[col].empty()) return failure();
    inputSlotMap[col + 1] =
        inputSlotMap[col] + static_cast<int64_t>(inputSlots[col].size());
  }

  for (const auto& slots : inputSlots) llvm::append_range(inputSlotMap, slots);

  std::vector<int64_t> destinationOwner(numDiagonals * ciphertextSize, -1);
  for (int64_t row = 0; row < rows; ++row) {
    llvm::sort(outputSlots[row]);
    outputSlots[row].erase(
        std::unique(outputSlots[row].begin(), outputSlots[row].end()),
        outputSlots[row].end());
    SmallVector<int64_t> expectedOutputSlots;
    for (int64_t slot = row; slot < ciphertextSize; slot += paddedRows) {
      expectedOutputSlots.push_back(slot);
    }
    if (ArrayRef<int64_t>(outputSlots[row]) !=
        ArrayRef<int64_t>(expectedOutputSlots))
      return failure();

    for (int64_t col = 0; col < cols; ++col) {
      std::vector<int64_t> outputCoverage(outputSlots[row].size(), 0);
      for (int64_t inputSlot : inputSlots[col]) {
        int64_t destinationCount = 0;
        for (int64_t outputIndex = 0;
             outputIndex < static_cast<int64_t>(outputSlots[row].size());
             ++outputIndex) {
          int64_t outputSlot = outputSlots[row][outputIndex];
          int64_t diagonal = (inputSlot - outputSlot) % ciphertextSize;
          if (diagonal < 0) diagonal += ciphertextSize;
          if (diagonal >= numDiagonals) continue;

          int64_t destination = diagonal * ciphertextSize + outputSlot;
          int64_t logicalElement = row * cols + col;
          if (destinationOwner[destination] != -1 &&
              destinationOwner[destination] != logicalElement) {
            return failure();
          }
          destinationOwner[destination] = logicalElement;
          ++destinationCount;

          // The squat Halevi-Shoup post-shifts add precursor slots spaced by
          // paddedRows. Record every final output slot reached by this packed
          // matrix entry and require an exact cover below.
          for (int64_t offset = 0; offset < paddedCols; offset += paddedRows) {
            int64_t finalSlot = (outputSlot - offset) % ciphertextSize;
            if (finalSlot < 0) finalSlot += ciphertextSize;
            if (finalSlot % paddedRows != row) return failure();
            int64_t finalIndex = finalSlot / paddedRows;
            if (finalIndex < 0 ||
                finalIndex >= static_cast<int64_t>(outputCoverage.size())) {
              return failure();
            }
            ++outputCoverage[finalIndex];
          }
        }
        if (destinationCount != 1) return failure();
      }
      if (llvm::any_of(outputCoverage,
                       [](int64_t count) { return count != 1; }))
        return failure();
    }
  }
  return inputSlotMap;
}

FailureOr<presburger::IntegerRelation> pushSliceLayoutThroughInsertSlice(
    SmallVector<int64_t> insertSliceSizes, ArrayRef<int64_t> resultShape,
    const presburger::IntegerRelation& sliceLayout) {
  // Check if the slice we insert fills up a full slice of the output tensor.
  SmallVector<int64_t> unitDims;
  for (auto [idx, size] : llvm::enumerate(insertSliceSizes)) {
    if (size != 1) {
      // Non-unit sizes must equal the input slice.
      assert(size == resultShape[idx]);
      continue;
    }
    // We have a unit dimension. If the unit dimension matches with an output
    // size 1 dimension, then we can drop the dimension. Otherwise, we are
    // inserting into a single slice of a larger dimension, so collect the
    // dimension for later.
    unitDims.push_back(idx);
  }

  // Get the number of ct that each slice takes up.
  auto numCt = sliceLayout.getConstantBound64(
      BoundType::UB, sliceLayout.getVarKindOffset(VarKind::Range));
  if (!numCt.has_value()) {
    return failure();
  }
  auto numSlots = sliceLayout.getConstantBound64(
      BoundType::UB, sliceLayout.getVarKindOffset(VarKind::Range) + 1);
  if (!numSlots.has_value()) {
    return failure();
  }

  // Now the slice takes up a full subtensor of the output tensor. For the
  // result layout, let this subtensor be preserved.
  auto resultLayout = sliceLayout.clone();
  // Add domain variables in the positions of the unit dimensions only if the
  // slice layout has fewer dimensions than the insert slice sizes (indicating
  // the layout doesn't describe all dimensions of the slice).
  if (sliceLayout.getNumDomainVars() < insertSliceSizes.size()) {
    for (int i = 0; i < resultShape.size(); ++i) {
      if (llvm::is_contained(unitDims, i)) {
        auto domainVar = resultLayout->insertVar(VarKind::Domain, i, 1);
        addBounds(*resultLayout, domainVar, 0, resultShape[i] - 1);
      }
    }
  } else {
    // When the slice layout already has all domain dimensions, we need to
    // relax the bounds on domain variables corresponding to unit dimensions
    // to match the result shape. We do this by removing equality and inequality
    // constraints that involve these dimensions.
    for (auto dim : unitDims) {
      // Find and remove equalities involving this dimension
      for (int i = resultLayout->getNumEqualities() - 1; i >= 0; --i) {
        if (resultLayout->atEq64(i, dim) != 0) {
          resultLayout->removeEquality(i);
        }
      }
      // Find and remove inequalities involving this dimension
      for (int i = resultLayout->getNumInequalities() - 1; i >= 0; --i) {
        if (resultLayout->atIneq64(i, dim) != 0) {
          resultLayout->removeInequality(i);
        }
      }
      // Add new bounds
      addBounds(*resultLayout, dim, 0, resultShape[dim] - 1);
    }
  }

  // Add a range var to indicate the index of the slice.
  auto newRangeVar = resultLayout->insertVar(VarKind::Range, 0, 1);

  // Create a relation mapping the new unit dimensions to a new range variable r
  // that represents the index into each (ct, slot) per slice.
  SmallVector<int64_t> newRangeCoeffs(resultLayout->getNumCols(), 0);
  unsigned product = 1;
  for (auto dim : unitDims) {
    newRangeCoeffs[dim] = product;
    product *= resultShape[dim];
  }
  newRangeCoeffs[newRangeVar] = -1;
  resultLayout->addEquality(newRangeCoeffs);

  addBounds(*resultLayout, newRangeVar, 0, product - 1);

  // Now compose the relation with a new relation mapping (r, ct, slot) -> (r *
  // num_ct + ct, slot).
  IntegerRelation newRangeRelation(PresburgerSpace::getRelationSpace(
      /*numDomain=*/3, /*numRange=*/2, /*numSymbol=*/0, /*numLocals=*/0));
  auto domainOffset = newRangeRelation.getVarKindOffset(VarKind::Domain);
  auto rangeOffset = newRangeRelation.getVarKindOffset(VarKind::Range);
  addBounds(newRangeRelation, domainOffset, 0);
  addBounds(newRangeRelation, domainOffset + 1, 0, numCt.value());
  addBounds(newRangeRelation, domainOffset + 2, 0, numSlots.value());
  addBounds(newRangeRelation, rangeOffset + 1, 0, numSlots.value());
  // slot = slot
  addConstraint(newRangeRelation,
                {{rangeOffset + 1, 1}, {domainOffset + 2, -1}},
                /*equality=*/true);
  // r * num_ct + ct = ct'
  addConstraint(newRangeRelation,
                {{rangeOffset, -1},
                 {domainOffset, numCt.value() + 1},
                 {domainOffset + 1, 1}},
                /*equality=*/true);
  resultLayout->compose(newRangeRelation);

  return *resultLayout;
}

}  // namespace heir
}  // namespace mlir
