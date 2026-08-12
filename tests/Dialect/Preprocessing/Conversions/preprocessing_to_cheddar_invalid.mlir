// RUN: heir-opt --preprocessing-to-cheddar --cheddar-bufferize --fold-memref-alias-ops --convert-to-emitc=filter-dialects=cheddar,arith --verify-diagnostics %s

// A borrowed payload cannot be redirected into preprocessing storage, so
// One-Shot keeps a copy of it, and a plaintext has no deep copy.
func.func @borrowed_payload(%arg: tensor<!cheddar.plaintext>) {
  %storage = preprocessing.empty : !preprocessing.storage<tensor<!cheddar.plaintext>>
  // expected-error @below {{no deep copy for '!cheddar.plaintext'}}
  // expected-error @below {{failed to legalize operation 'memref.copy'}}
  preprocessing.store %arg, %storage[] site 0 <tensor<!cheddar.plaintext>> : tensor<!cheddar.plaintext>, !preprocessing.storage<tensor<!cheddar.plaintext>>
  return
}
