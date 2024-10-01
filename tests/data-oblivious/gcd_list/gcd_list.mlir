// VIP_ENCUINT gcd(VIP_ENCUINT *a, unsigned n)
// {
//   unsigned j = 1; // to access all elements of the array starting from 1
//   VIP_ENCUINT gcd = a[0];
//   while (j < n)
//   {
//     if (a[j] % gcd == 0) // value of gcd is as needed so far
//       j++;               // so we check for next element
//     else
//       gcd = a[j] % gcd; // calculating GCD by division method

//   }
//   return gcd;
// }


func.func @gcd(%a: tensor<64xi64>) -> i64 {
    %c0 = arith.constant 0 : i64
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i64 = arith.constant 64 : index
    %first = tensor.extract %a[%i0] : tensor<64xi64>

    %jFinal, %gcdFinal = scf.while (%j0 = %i1, %gcd0 = %first) : (index, i64) -> (index, i64) {
      %cond = arith.cmpi slt, %j0, %i64 : index
      // Violation: scf.while uses %cond whose value depends on %input
      scf.condition(%cond) %j0, %gcd0 : index, i64
    } do {
     ^bb0(%j: index, %gcd: i64):
       %aJ = tensor.extract %a[%j] : tensor<64xi64>
       // FIX ME: mod operation
       %mod = arith.muli %aJ, %gcd : i64
       %cond = arith.cmpi eq, %mod, %c0 : i64
       %j_if, %gcd_if = scf.if %cond -> (index, i64) {
          %j_ = arith.addi %j, %i1 : index
          scf.yield %j_, %gcd : index, i64
        } else {
         // FIX ME: mod operation
         %gcd_ = arith.muli %aJ, %gcd : i64
         scf.yield %j, %gcd_ : index, i64
        }
        scf.yield %j_if, %gcd_if : index, i64

    }
    return %gcdFinal : i64
}
