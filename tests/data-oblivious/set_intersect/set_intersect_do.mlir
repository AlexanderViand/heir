// **** Original Data-oblivious Set Intersection Code from VIP-Bench ****
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


func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func private @printTensor16(%tensor: tensor<16xi1>) attributes {llvm.emit_c_interface} {
  %1 = arith.extui %tensor : tensor<16xi1> to tensor<16xi32>
  %memRef = bufferization.to_memref %1 : memref<16xi32>
  %cast = memref.cast %memRef : memref<16xi32> to memref<*xi32>
  call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}

func.func @set_intersect(%setA: tensor<16xi16>, %setB: tensor<16xi16>) -> tensor<16xi1>{
    %temp = tensor.empty() : tensor<16xi1>
    %notFound = arith.constant 0 : i1
    %found = arith.constant 1 : i1
    %result = affine.for %i = 0 to 16 iter_args(%tensor = %temp) -> (tensor<16xi1>){
        %result = affine.for %j = 0 to 16 iter_args(%val = %notFound) -> (i1){
            %A = tensor.extract %setA[%i] : tensor<16xi16>
            %B = tensor.extract %setB[%j] : tensor<16xi16>
            %cond = arith.cmpi eq, %A, %B : i16
            %match = arith.select %cond, %found, %val : i1
            affine.yield %match : i1
        }
        %inserted = tensor.insert %result into %tensor[%i] : tensor<16xi1>
        affine.yield %inserted : tensor<16xi1>

    }
    return %result : tensor<16xi1>

}

func.func @main() {
    %c2 = arith.constant 2 : i16
    %c3 = arith.constant 3 : i16
    %c4 = arith.constant 4 : i16

    %setA = tensor.generate{
        ^bb1(%i : index):
            tensor.yield %c2 : i16
    } : tensor<16xi16>
    %setB = tensor.generate{
        ^bb1(%i : index):
            tensor.yield %c3 : i16
    } : tensor<16xi16>

    %setC = tensor.from_elements %c2, %c3, %c2, %c3, %c2, %c3, %c2, %c3, %c2, %c3, %c2, %c3, %c2, %c3, %c2, %c3 : tensor<16xi16>
    %setD = tensor.from_elements %c2, %c4, %c2, %c4, %c2, %c4, %c2, %c4, %c2, %c4, %c2, %c4, %c2, %c4, %c2, %c4 : tensor<16xi16>

    %same = func.call @set_intersect(%setA, %setA) : (tensor<16xi16>, tensor<16xi16>) -> tensor<16xi1>
    %different = func.call @set_intersect(%setA, %setB) : (tensor<16xi16>, tensor<16xi16>) -> tensor<16xi1>
    %random = func.call @set_intersect(%setC, %setD) : (tensor<16xi16>, tensor<16xi16>) -> tensor<16xi1>

    func.call @printTensor16(%same) : (tensor<16xi1>) -> ()
    func.call @printTensor16(%different) : (tensor<16xi1>) -> ()
    func.call @printTensor16(%random) : (tensor<16xi1>) -> ()


    return
}
