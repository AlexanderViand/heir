// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --split-input-file --verify-diagnostics %s

// A dynamic-shape memref of a move-only cheddar payload can't be represented as
// a fixed-size `std::array`, so the cheddar type converter refuses it and the
// conversion fails -- rather than falling through to something stock
// MemRefToEmitC would lower with copies of the move-only payloads.
//
// (Static multi-dimensional payload memrefs ARE supported now: they map to a
// nested `std::array<std::array<...>>`, subscripted as `m[i][j]`. And a
// move-only value carried through an scf.for iter_arg is likewise supported via
// the destination-passing loop lowering -- see loop.mlir.)
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @dynamic(%m: memref<?x!cheddar.ciphertext>) {
  return
}

// -----

// Non-unit payload strides cannot be represented by a nested std::array.
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @strided_payload(
    %m: memref<4x!cheddar.ciphertext, strided<[2]>>) {
  return
}

// -----

// Likewise, a raw float pointer would lose this stride and miscompile users.
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @strided_float(%ctx: !cheddar.context,
                         %m: memref<4xf32, strided<[2]>>) {
  return
}

// -----

// A nested std::array subscript can remove only leading dimensions. Silently
// treating a middle rank reduction as a leading one would select the wrong
// ciphertext.
func.func @drop_middle_dimension(
    %m: memref<2x1x4x!cheddar.ciphertext>) {
  // expected-error @below {{failed to legalize operation 'memref.subview'}}
  %slice = memref.subview %m[0, 0, 0] [2, 1, 4] [1, 1, 1]
      : memref<2x1x4x!cheddar.ciphertext> to memref<2x4x!cheddar.ciphertext>
  return
}
