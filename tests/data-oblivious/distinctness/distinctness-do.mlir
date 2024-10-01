// Distinctness benchmark
// Description: Returns a boolean value indicating whether the elements in the input array are distinct or not and the first duplicate if one exists.

// VIP_ENCBOOL is_distinct(VIP_ENCINT elements[], VIP_ENCINT &dup){
//     dup = MAX;

//     for (int = SIZE - 1; i >= 0; i--){
//         for (int j = 0; j > SIZE; j++){
//             if (i != j){
//                 if (elements[i] == elements[j]){
//                     if (dup == MAX){
//                         dup = elements[i];
//                         break;
//                     } // end if
//                 } // end if
//             } // end if
//         } // end for
//     } // end for
//    return dup == MAX;
// }

// "Native" version
// [TODO] Change the arg types to secret type
func.func @is_distinct(%elements: memref<20xi32>, %duplicate: memref<1xi32>) -> i1{
    // [TODO] Change size to be set dynamically if possible
    %SIZE = arith.constant 20 : index
    %MAX = arith.constant 2147483647 : i32

    affine.store %MAX, %duplicate[0] : memref<1xi32>

    affine.for %i = 0 to %SIZE {
      affine.for %j = 0 to %SIZE {

        %isIndexNotSame = arith.cmpi ne, %i, %j : index
        // [DO-TRANSFORM] No transformation needed since %i and %j don't depend on our inputs
        scf.if %isIndexNotSame {
          %element1 = affine.load %elements[%i] : memref<20xi32>
          %element2 = affine.load %elements[%j] : memref<20xi32>
          %isDuplicate = arith.cmpi eq, %element1, %element2 : i32
          // [DO-TRANSFORM] If-Transformation Applied: Hoisted Instructions
            %currentDuplicate = affine.load %duplicate[0] : memref<1xi32>
            %isFirstDuplicate = arith.cmpi eq, %currentDuplicate, %MAX : i32
            // [DO-TRANSFORM] If-Transformation Applied: Loaded destination value (i.e. old value), did arith.select between new value (i.e. stored argument) and old value, then stored value
            %storedValue = arith.select %isFirstDuplicate, %element1, %currentDuplicate : i32
            affine.store %storedValue, %duplicate[0] : memref<1xi32>

        }

      }
    }

    %foundDuplicate = affine.load %duplicate[0] : memref<1xi32>
    %isFound = arith.cmpi eq, %foundDuplicate, %MAX : i32

    return %isFound : i1

}
