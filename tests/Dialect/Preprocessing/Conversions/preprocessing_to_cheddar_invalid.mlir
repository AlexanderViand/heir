// RUN: heir-opt --preprocessing-to-cheddar --cheddar-bufferize --fold-memref-alias-ops --convert-to-emitc=filter-dialects=cheddar,arith --verify-diagnostics %s

// A borrowed payload cannot be redirected into preprocessing storage. One-Shot
// therefore preserves the required copy. It survives alias folding and the
// EmitC ownership check rejects it before it can silently consume the caller's
// plaintext.
func.func @borrowed_payload(%arg: tensor<!cheddar.plaintext>) {
  %storage = preprocessing.empty : !preprocessing.storage<tensor<!cheddar.plaintext>>
  // expected-error @below {{copying a move-only Cheddar value with memref.copy is invalid}}
  preprocessing.store %arg, %storage[] site 0 <tensor<!cheddar.plaintext>> : tensor<!cheddar.plaintext>, !preprocessing.storage<tensor<!cheddar.plaintext>>
  return
}
