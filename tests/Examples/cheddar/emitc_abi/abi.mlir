func.func @abi_inner(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext> {bufferize.result}) {
  return
}

func.func @abi_outer(
    %input: memref<1x!cheddar.ciphertext>,
    %output: memref<1x!cheddar.ciphertext> {bufferize.result}) {
  func.call @abi_inner(%input, %output)
      : (memref<1x!cheddar.ciphertext>, memref<1x!cheddar.ciphertext>) -> ()
  return
}
