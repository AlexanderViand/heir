func.func @bad_char_heuristic(%str: tensor<256xi8> {secret.secret}, %size: index, %badCharArg: tensor<256xi32> {secret.secret}) -> (tensor<256xi32>) {
    %neg1 = arith.constant -1 : i32
    %initBadChar = affine.for %i = 0 to 256 iter_args(%temp = %badCharArg) -> (tensor<256xi32>){
        %tempBadChar = tensor.insert %neg1 into %temp[%i] : tensor<256xi32>
        affine.yield %tempBadChar : tensor<256xi32>
    }

    %badChar = affine.for %i = 0 to %size iter_args(%iBadChar = %initBadChar) -> (tensor<256xi32>){
        %strI = tensor.extract %str[%i] : tensor<256xi8>
        %badCharIndex = index.casts %strI: i8 to index

        %value = index.casts %i : index to i32
        %tempBadChar = affine.for %j = 0 to 256 iter_args(%jBadChar = %iBadChar) -> (tensor<256xi32>){
            %cond = arith.cmpi eq, %j, %badCharIndex : index
            %oldValue = tensor.extract %jBadChar[%j] : tensor<256xi32>
            %insertedChar = arith.select %cond, %value, %oldValue : i32
            %jTempBadChar = tensor.insert %value into %jBadChar[%badCharIndex] : tensor<256xi32>
            affine.yield %jTempBadChar : tensor<256xi32>
        }

        affine.yield %tempBadChar : tensor<256xi32>

    }

    return %badChar : tensor<256xi32>
}
