// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar,arith --split-input-file --verify-diagnostics %s

// memref.store has copy semantics. It cannot be reinterpreted as an ownership
// transfer merely because a move-only payload happens to be dead or local.
func.func @borrowed_source(
    %src: memref<!cheddar.plaintext>,
    %dst: memref<!cheddar.plaintext>) {
  %value = memref.load %src[] : memref<!cheddar.plaintext>
  // expected-error @below {{copying a move-only Cheddar payload with memref.store is invalid}}
  memref.store %value, %dst[] : memref<!cheddar.plaintext>
  return
}

// -----

func.func @local_source(%dst: memref<!cheddar.plaintext>) {
  %src = memref.alloc() : memref<!cheddar.plaintext>
  %value = memref.load %src[] : memref<!cheddar.plaintext>
  // expected-error @below {{copying a move-only Cheddar payload with memref.store is invalid}}
  memref.store %value, %dst[] : memref<!cheddar.plaintext>
  return
}

// -----

func.func @source_used_twice(%dst: memref<2x!cheddar.plaintext>) {
  %src = memref.alloc() : memref<!cheddar.plaintext>
  %value = memref.load %src[] : memref<!cheddar.plaintext>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{copying a move-only Cheddar payload with memref.store is invalid}}
  memref.store %value, %dst[%c0] : memref<2x!cheddar.plaintext>
  // expected-error @below {{copying a move-only Cheddar payload with memref.store is invalid}}
  memref.store %value, %dst[%c1] : memref<2x!cheddar.plaintext>
  return
}
