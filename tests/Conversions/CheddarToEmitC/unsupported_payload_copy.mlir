// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --split-input-file --verify-diagnostics %s

// A memref.copy is not an ownership-transfer operation. Moving from a borrowed
// function argument would silently consume the caller's ciphertext.
func.func @borrowed_source(
    %src: memref<!cheddar.ciphertext>,
    %dst: memref<!cheddar.ciphertext>) {
  // expected-error @below {{copying a move-only Cheddar value with memref.copy is invalid}}
  memref.copy %src, %dst
      : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// -----

// A self-copy is a no-op, not a self-move.
func.func @self_copy(%value: memref<!cheddar.ciphertext>) {
  memref.copy %value, %value
      : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// -----

// Local ownership does not turn memref.copy into an ownership transfer.
func.func @local_source(%dst: memref<!cheddar.ciphertext>) {
  %src = memref.alloc() : memref<!cheddar.ciphertext>
  // expected-error @below {{copying a move-only Cheddar value with memref.copy is invalid}}
  memref.copy %src, %dst
      : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// -----

// The setup UI is represented by std::unique_ptr at an owning buffer boundary
// and is subject to the same no-copy rule as ciphertext payloads.
func.func @user_interface_copy(
    %src: memref<!cheddar.user_interface>,
    %dst: memref<!cheddar.user_interface>) {
  // expected-error @below {{copying a move-only Cheddar value with memref.copy is invalid}}
  memref.copy %src, %dst
      : memref<!cheddar.user_interface> to memref<!cheddar.user_interface>
  return
}
