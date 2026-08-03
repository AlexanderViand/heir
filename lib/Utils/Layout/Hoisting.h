#ifndef LIB_UTILS_LAYOUT_HOISTING_H_
#define LIB_UTILS_LAYOUT_HOISTING_H_

#include <cstdint>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

// Hoist the conversion of a vector layout through a matrix-vector multiply
// operation.
//
// Returns a new layout for the matrix argument of the matvec op.
//
// Note: this function requires the assumption that the chosen packing for the
// vector (and the corresponding matvec kernel) packs the vector into a single
// ciphertext.
presburger::IntegerRelation hoistConversionThroughMatvec(
    const presburger::IntegerRelation& matrixLayout,
    const presburger::IntegerRelation& fromVecLayout,
    const presburger::IntegerRelation& toVecLayout);

// Returns a compact CSR encoding of the packed input slots for each matrix
// column when a diagonal matrix packing can consume `inputLayout` and produce
// the standard row-major `outputLayout` without an online ciphertext
// conversion. The first `numColumns + 1` entries are offsets into the remaining
// slot entries. Both vector layouts must fit in one ciphertext; unsupported
// layouts fail so callers can retain the explicit fallback.
FailureOr<SmallVector<int64_t>> getMatvecInputSlots(
    RankedTensorType matrixType, const presburger::IntegerRelation& inputLayout,
    const presburger::IntegerRelation& outputLayout, int64_t ciphertextSize);

// Infers a layout relation for the result of an insert_slice operation that
// preserves the slice as a single continuous block in the destination
// ciphertext semantics type.
FailureOr<presburger::IntegerRelation> pushSliceLayoutThroughInsertSlice(
    SmallVector<int64_t> insertSliceSizes, ArrayRef<int64_t> resultShape,
    const presburger::IntegerRelation& sliceLayout);

// Absorb a matvec input vector's packing into the matrix layout.
//
// `matrixLayout` must be a (squat) diagonal matrix layout built under the
// convention that the matrix column coordinate equals the slot position of
// the packed input vector (i.e., a row-major packed vector). `vecLayout` is
// the actual packing of the input vector (index -> (ct, slot)) with all
// elements in a single ciphertext (ct = 0).
//
// Returns the matrix layout with its column coordinate re-indexed by the
// vector's slot positions, D' = D ∘ (id_row × slot(V)), so the same
// Halevi-Shoup diagonal kernel consumes the vector in its actual packing
// without a ciphertext-side layout conversion. Unlike
// hoistConversionThroughMatvec (which re-packs the matrix's own (ct, slot)
// coordinates and is only sound for rotation-like re-packings), this
// substitutes the *column space* and is sound for any packing where each
// vector element occupies exactly one slot (see
// isSingleCiphertextPermutation).
presburger::IntegerRelation absorbVectorLayoutIntoMatrix(
    const presburger::IntegerRelation& matrixLayout,
    const presburger::IntegerRelation& vecLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_HOISTING_H_
