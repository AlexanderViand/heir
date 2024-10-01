// func.func @sumTensorInRange(%input: tensor<16xi32>, %lower: index, %upper:index) -> i32{
 func.func @sumTensorInRange(%input: tensor<16xi32>) -> i32{
    // add range check 
    %lower = arith.constant 0 : index
    %upper = arith.constant 16 : index
    %c_0 = arith.constant 0 : i32
    %step = arith.constant 1 : index
    %sum = affine.for %i = 1 to 16 step 1 iter_args(%sum_iter = %c_0) -> (i32) {
        %extracted = tensor.extract %input[%i] : tensor<16xi32>
        %temp = arith.addi %extracted, %sum_iter : i32
        affine.yield %temp : i32
    }

    return %sum : i32
}