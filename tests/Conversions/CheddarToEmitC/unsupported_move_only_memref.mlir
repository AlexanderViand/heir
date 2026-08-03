// RUN: heir-opt --convert-to-emitc --split-input-file --verify-diagnostics %s

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
func.func @dynamic(%ctx: !cheddar.context, %m: memref<?x!cheddar.ciphertext>,
                   %i: index, %out: memref<!cheddar.ciphertext>) {
  %s = memref.subview %m[%i] [1] [1]
      : memref<?x!cheddar.ciphertext> to memref<!cheddar.ciphertext, strided<[], offset: ?>>
  cheddar.add %ctx, %s, %s, %out
      : (!cheddar.context, memref<!cheddar.ciphertext, strided<[], offset: ?>>, memref<!cheddar.ciphertext, strided<[], offset: ?>>, memref<!cheddar.ciphertext>) -> ()
  return
}
