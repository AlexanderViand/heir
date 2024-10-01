// void set_intersect(vector<VIP_ENCINT>& setA, vector<VIP_ENCINT>& setB, vector<VIP_ENCBOOL>& setA_match)
// {
//   for (unsigned i=0; i < setA.size(); i++)
//   {
//     VIP_ENCBOOL match = false;
//     for (unsigned j=0; j < setB.size(); j++)
//     {
//       /**** IISWC DO Transformation: <IF> Using CMOV in place of if-statement ****/
//       match = VIP_CMOV(setA[i] == setB[j], (VIP_ENCBOOL)true, match);
//     }
//     setA_match[i] = match;
//   }
// }



func.func @set_intersect(%setA: tensor<16xi16>, %setB: tensor<16xi16>) -> tensor<16xi1>{
    %temp = tensor.empty() : tensor<16xi1>
    %result = affine.for %i = 0 to 16 iter_args(%arg_i = %temp) -> (tensor<16xi1>){
        %initial = arith.constant 0 : i1
        %match = affine.for %j = 0 to 16 iter_args(%arg_j = %initial) -> (i1){
            %setA_i = tensor.extract %setA[%i] : tensor<16xi16>
            %setB_j = tensor.extract %setB[%j] : tensor<16xi16>
            %cond = arith.cmpi eq, %setA_i, %setB_j : i16
            %found = arith.constant 1 : i1
            %match = arith.select %cond, %found, %arg_j : i1
            affine.yield %match : i1
        }
        %inserted = tensor.insert %match into %arg_i[%i] : tensor<16xi1>
        affine.yield %inserted : tensor<16xi1>

    }
    return %result : tensor<16xi1>

}

func.func @main() -> (tensor<16xi1>){
    %size = arith.constant 16 : index
    %setA = tensor.generate{
        ^bb1(%i : index):
            %two = arith.constant 2 : i16
            tensor.yield %two : i16
    } : tensor<16xi16>
    %result = call @set_intersect(%setA, %setA) : (tensor<16xi16>, tensor<16xi16>) -> tensor<16xi1>
    return %result : tensor<16xi1>
}
