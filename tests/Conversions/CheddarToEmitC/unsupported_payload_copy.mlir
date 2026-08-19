// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --split-input-file --verify-diagnostics %s

// A ciphertext memref.copy is a real deep copy, not an ownership transfer.
func.func @borrowed_source(
    %ctx: !cheddar.context,
    %src: memref<!cheddar.ciphertext>,
    %dst: memref<!cheddar.ciphertext>) {
  memref.copy %src, %dst
      : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// -----

// Rank-0 subviews use an explicit strided layout. They have the same deep-copy
// semantics as identity-layout ciphertext buffers.
func.func @strided_scalar(
    %ctx: !cheddar.context,
    %src: memref<!cheddar.ciphertext, strided<[]>>,
    %dst: memref<!cheddar.ciphertext, strided<[]>>) {
  memref.copy %src, %dst
      : memref<!cheddar.ciphertext, strided<[]>>
        to memref<!cheddar.ciphertext, strided<[]>>
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

// Local ownership does not change the copy semantics.
func.func @local_source(%ctx: !cheddar.context,
                        %dst: memref<!cheddar.ciphertext>) {
  %src = memref.alloc() : memref<!cheddar.ciphertext>
  memref.copy %src, %dst
      : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// -----

// CHEDDAR needs a context to perform a ciphertext deep copy.
func.func @missing_context(
    %src: memref<!cheddar.ciphertext>,
    %dst: memref<!cheddar.ciphertext>) {
  // expected-error @below {{cannot deep-copy a Cheddar ciphertext without a context argument}}
  memref.copy %src, %dst
      : memref<!cheddar.ciphertext> to memref<!cheddar.ciphertext>
  return
}

// -----

// The setup UI is represented by std::unique_ptr at an owning buffer boundary
// and has no deep-copy operation.
func.func @user_interface_copy(
    %src: memref<!cheddar.user_interface>,
    %dst: memref<!cheddar.user_interface>) {
  // expected-error @below {{copying a move-only Cheddar value with memref.copy is invalid}}
  memref.copy %src, %dst
      : memref<!cheddar.user_interface> to memref<!cheddar.user_interface>
  return
}
