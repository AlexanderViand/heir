// void set_intersect(vector<VIP_ENCINT>& setA, vector<VIP_ENCINT>& setB, vector<VIP_ENCBOOL>& setA_match)
// {
//   for (unsigned i=0; i < setA.size(); i++)
//   {
//     VIP_ENCBOOL match = false;
//     for (unsigned j=0; j < setB.size(); j++)
//     {

// #ifdef VIP_DO_MODE
//       /**** IISWC DO Transformation: <IF> [MZD] ****/
//       /**** Description: Using CMOV in place of if-statement ****/
//       match = VIP_CMOV(setA[i] == setB[j], (VIP_ENCBOOL)true, match);
// #else /* !VIP_DO_MODE */
//       if (setA[i] == setB[j])
//         match = true;
// #endif /* VIP_DO_MODE */
//     }
//     setA_match[i] = match;
//   }
// }

// func.func @set_intersect(%setA: tensor<16xi16>, %setB: tensor<16xi16>) -> tensor<16xi1>{
//     %setMatch = tensor.empty() : tensor<16xi1>
//     %resultSet = affine.for %i = 0 to 16 iter_args(%argSetMatch = %setMatch) -> (tensor<16xi1>){
//         %match = arith.constant 0 : i1
//         %for_j = affine.for %j = 0 to 16 iter_args(%arg = %match) -> (i1){
//             %setA_i = tensor.extract %setA[%i] : tensor<16xi16>
//             %setB_j = tensor.extract %setB[%j] : tensor<16xi16>
//             %cond = arith.cmpi eq, %setA_i, %setB_j : i16
//             // [DO-TRANSFORM] If-Transformation
//             %result = scf.if %cond -> (i1){
//                 %found = arith.constant 1 : i1
//                 scf.yield %found : i1
//             }
//             else{
//                 scf.yield %arg : i1
//             }
//             affine.yield %result : i1
//         }
//         %newSetMatch = tensor.insert %for_j into %argSetMatch[%i] : tensor<16xi1>
//         affine.yield %newSetMatch : tensor<16xi1>

//     }
//     return %resultSet : tensor<16xi1>

// }

func.func @set_intersect(%setA: tensor<16xi16>, %setB: tensor<16xi16>) -> tensor<16xi1>{
    %setMatch = tensor.empty() : tensor<16xi1>
    %resultSet = affine.for %i = 0 to 16 iter_args(%argSetMatch = %setMatch) -> (tensor<16xi1>){
        %match = arith.constant 0 : i1
        %for_j = affine.for %j = 0 to 16 iter_args(%arg = %match) -> (i1){
            %setA_i = tensor.extract %setA[%i] : tensor<16xi16>
            %setB_j = tensor.extract %setB[%j] : tensor<16xi16>
            %cond = arith.cmpi eq, %setA_i, %setB_j : i16
            %found = arith.constant 1 : i1
            %result = arith.select %cond, %found, %arg : i1
            affine.yield %result : i1
        }
        %newSetMatch = tensor.insert %for_j into %argSetMatch[%i] : tensor<16xi1>
        affine.yield %newSetMatch : tensor<16xi1>

    }
    return %resultSet : tensor<16xi1>

}
