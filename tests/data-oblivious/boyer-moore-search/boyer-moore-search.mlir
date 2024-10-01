// Boyer-Moore Benchmark
// Description: A C Program for Bad Character Heuristic of Boyer-Moore String Matching Algorithm */
// SOURCE: https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/

// void badCharHeuristic(vector<VIP_ENCCHAR> str, int size, VIP_ENCINT badchar[NO_OF_CHARS])
// {
//   // Initialize all occurrences as -1
//   for (int i = 0; i < NO_OF_CHARS; i++)
//   {
//     badchar[i] = -1;
//   }

//   // Fill the actual value of last occurrence of a character
//   for (int i = 0; i < size; i++)
//   {
// #ifndef VIP_DO_MEM
//     badchar[(int)str[i]] = i;
// #else
//     /**** IISWC DO Transformation: <ACCESS> <B.M.T> ****/
//     // True data-obliviousness requires us to write to all indexes
//     VIP_ENCINT badchar_index = (VIP_ENCINT)str[i];
//     for (int j = 0; j < NO_OF_CHARS; j++)
//     {
//       badchar[j] = VIP_CMOV(j == badchar_index, j, badchar[j]);
//     }
// #endif
//   }
// }

// /* A pattern searching function that uses Bad
// Character Heuristic of Boyer Moore Algorithm */
// void search(vector<VIP_ENCCHAR> txt, int n, vector<VIP_ENCCHAR> pat, int m, VIP_ENCBOOL *ret)
// {
//   VIP_ENCINT badchar[NO_OF_CHARS];

//   /* Fill the bad character array by calling
//   the preprocessing function badCharHeuristic()
//   for given pattern */
//   badCharHeuristic(pat, m, badchar);

//   VIP_ENCINT s = 0; // s is shift of the pattern with
//                     // respect to text

//   for (int l = 0; l <= (n - m); l++)
//   {
// #ifndef VIP_DO_IF
//     /**** IISWC DO Transformation: <LOOP> <B.M.T> ****/
//     // Early exit if not data-oblivious...
//     if (VIP_DEC(s) > (n - m))
//       break;
// #endif

//     VIP_ENCINT idx = m - 1;

//     /* Keep reducing index idx of pattern while
//     characters of pattern and text are
//     matching at this shift s */
// #ifndef VIP_DO_MODE
//     while (idx >= 0 && pat[VIP_DEC(idx)] == txt[VIP_DEC(s + idx)])
//     {
//       idx--;
//     }
// #else
//     /**** IISWC DO Transformation: <LOOP> <B.M.T> ****/
//     for (int i = 0; i < m; i++)
//     {
//       /**** IISWC DO Transformation: <ACCESS> <B.M.T> ****/
//       // Access: pat[VIP_DEC(idx)]
//       VIP_ENCCHAR pat_idx = pat[0];
//       for (int j = 0; j < m; j++)
//         pat_idx = VIP_CMOV(j == idx, pat[j], pat_idx);

//       /**** IISWC DO Transformation: <ACCESS> <B.M.T> ****/
//       // Access: txt[VIP_DEC(s+idx)]
//       VIP_ENCCHAR txt_idx = txt[0];
//       for (int j = 0; j < n; j++)
//         txt_idx = VIP_CMOV(j == s + idx, txt[j], txt_idx);

//       /* Comment from Reviewer: [MZD]
//       * 
//       * For the for-loops above, to improve performance I would do the following: 
//       * Identify if n or m is greater: let's call it x. Description [x: (n > m) ? n : m]
//       *  
//       * VIP_ENCCHAR pat_idx = pat[0];
//       * VIP_ENCCHAR txt_idx = txt[0];
//       * 
//       * for(int j = 0; j < x; j++){
//       *     pat_idx = VIP_CMOV(j == idx , pat[j%m], pat_idx);
//       *     txt_idx = VIP_CMOV(j == s + idx, txt[j%n], txt_idx);
//       *  }
//       * 
//       */

//       /**** IISWC DO Transformation: <IF> <B.M.T> ****/
//       idx = idx - VIP_CMOV(idx >= 0 && pat_idx == txt_idx, (VIP_ENCINT)1, (VIP_ENCINT)0);

//       /* 
//       * Comment from Reviewer: [MZD] 
//       * I wonder for the code above, if this version would be optimal in terms 
//       * of performance: idx = idx - (idx >= 0 && pat_idx == txt_idx);
//       * 
//       */
//     }
// #endif

//     /* If the pattern is present at current
//     shift, then index idx will become -1 after
//     the above loop */
//     VIP_ENCBOOL cond = (idx < 0);
// #ifndef VIP_DO_MODE
//     if (cond)
//     {
//       ret[s] = true;
//       s += (s + m < n) ? m - badchar[(int)txt[s + m]] : 1;
//     }
//     else
//     {
//       VIP_ENCINT s_shift = idx - badchar[(int)txt[VIP_DEC(s + idx)]];
//       s += 1 > s_shift ? 1 : s_shift;
//     }
// #else
//     /**** IISWC DO Transformation: <ACCESS/IF> <B.M.T> ****/
//     // True data-obliviousness requires us to write to all indexes
//     for (int i = 0; i < n; i++)
//     {
//       ret[i] = VIP_CMOV(cond && i == s, true, ret[i]);
//     }
//     /**** IISWC DO Transformation: <ACCESS> <B.M.T> ****/
//     // Access: badchar[VIP_DEC(txt[VIP_DEC(s+m)])]
//     VIP_ENCINT txt_sm = (VIP_ENCINT)txt[0];
//     for (int j = 0; j < n; j++)
//       txt_sm = VIP_CMOV(j == s + m, txt[j], txt_sm);

//     /**** IISWC DO Transformation: <ACCESS> <B.M.T> ****/
//     VIP_ENCINT badchar_txt_sm = badchar[0];
//     for (int j = 0; j < NO_OF_CHARS; j++)
//       badchar_txt_sm = VIP_CMOV(j == txt_sm, badchar[j], badchar_txt_sm);

//     /**** IISWC DO Transformation: <ACCESS> <B.M.T> ****/
//     // Access: badchar[VIP_DEC(txt[VIP_DEC(s+idx)])]
//     VIP_ENCINT txt_sidx = (VIP_ENCINT)txt[0];
//     for (int j = 0; j < n; j++)
//       txt_sidx = VIP_CMOV(j == s + idx, txt[j], txt_sidx);

//     /**** IISWC DO Transformation: <ACCESS> <B.M.T> ****/
//     VIP_ENCINT badchar_txt_sidx = badchar[0];
//     for (int j = 0; j < NO_OF_CHARS; j++)
//       badchar_txt_sidx = VIP_CMOV(j == txt_sidx, badchar[j], badchar_txt_sidx);

    
//     /* Comment from Reviewer: [MZD]
//       * 
//       * For the for-loops above, to improve performance I would do the following: 
//       *  
//       * VIP_ENCINT txt_sm = (VIP_ENCINT)txt[0];
//       * VIP_ENCINT badchar_txt_sm = badchar[0];
//       * VIP_ENCINT txt_sidx = (VIP_ENCINT)txt[0];
//       * VIP_ENCINT badchar_txt_sidx = badchar[0];
//       * 
//       * for(int j = 0; j < n; j++){
//       *     ret[j] = VIP_CMOV(cond && j == s, true, ret[j]);
//       *     txt_sm = VIP_CMOV(j == s + m, txt[j], txt_sm);
//       *     txt_sidx = VIP_CMOV(j == s + idx, txt[j], txt_sidx);
//       *  }
//       * 
//       * for(int j = 0; j< NO_OF_CHARS; j++){
//       *     badchar_txt_sm = VIP_CMOV(j == txt_sm , badchar[j], badchar_txt_sm);
//       *     badchar_txt_sidx = VIP_CMOV(j == txt_sidx, badchar[j], badchar_txt_sidx);
//       * }
//       * 
//       */


//     /**** IISWC DO Transformation: <IF> <B.M.T> ****/
//     VIP_ENCINT s_if = VIP_CMOV(s + m < n, m - badchar_txt_sm, 1);
//     VIP_ENCINT s_shift = idx - badchar_txt_sidx;

//     /**** IISWC DO Transformation: <IF> <B.M.T> ****/
//     VIP_ENCINT s_else = VIP_CMOV(1 > s_shift, 1, s_shift);

//     /**** IISWC DO Transformation: <IF> <B.M.T> ****/
//     s = s + VIP_CMOV(cond, s_if, s_else);
// #endif
//   }
// }

// %size = size of %str so it should be kept to 256 if we plan to hardcode the size of %str to be <256xi8>
func.func @bad_char_heuristic(%str: tensor<3xi8> {secret.secret}, %size: index, %badCharArg: tensor<256xi32> {secret.secret}) -> (tensor<256xi32>) {
    %neg1 = arith.constant -1 : i32
    %initBadChar = affine.for %i = 0 to 256 iter_args(%temp = %badCharArg) -> (tensor<256xi32>){
        %tempBadChar = tensor.insert %neg1 into %temp[%i] : tensor<256xi32>
        affine.yield %tempBadChar : tensor<256xi32>
    }
    
    %badChar = affine.for %i = 0 to %size iter_args(%iBadChar = %initBadChar) -> (tensor<256xi32>){
        %strI = tensor.extract %str[%i] : tensor<3xi8>
        %badCharIndex = index.casts %strI: i8 to index

        %value = index.casts %i : index to i32
        %tempBadChar = tensor.insert %value into %iBadChar[%badCharIndex] : tensor<256xi32>
        affine.yield %tempBadChar : tensor<256xi32>
    }

    return %badChar : tensor<256xi32>    
}


// pattern is a subset of the text so should be less than 256
// n (size of text): 256  and m (size of pattern): any number <= 256
// Here we set n: 256, m: 3 [TODO]: Find a way to set this globally instead of hardcoding
 func.func @search(%txt: tensor<256xi8> {secret.secret}, %n: i32, %pat: tensor<3xi8> {secret.secret}, %m: i32, %ret: tensor<256xi1> {secret.secret}) {
    %badChar = tensor.empty() : tensor<256xi32>
    func.call @bad_char_heuristic(%pat, %m, %badchar) : (tensor<3xi8>, i32, tensor<256xi32>) -> ()

    // s is shift of the pattern with respect to text
    %s = arith.constant 0 : i32
    %c_0 = arith.constant 0 : i32
    %c_1 = arith.constant 1 : i32
    %n_m = arith.subi %n, %m : i32
    %upperBound = arith.addi %n_m, %c_1 : i32
    %upperBoundIdx = index.casts %upperBound : i32 to index

    %newRet, %newS = affine.for %l = 0 to %upperBoundIdx iter_args(%whileRet = %ret, %whileS = %s) -> (tensor<256xi1>, i32){
        %cond1 = arith.cmpi sge, %whileS, %n_m : i32
        // [MLIR] Early exit if %cond1 is false

        %idx = arith.subi %m, %c_1 : i32
        %idxIndex = index.casts %idx : i32 to index
        %whileCond1 = arith.cmpi sge, %idx, %c_0 : i32

        %pat_idx = tensor.extract %pat[%idxIndex] : tensor<3xi8>
        %s_index = index.casts %whileS : i32 to index
        %s_idx = arith.addi %s_index, %idxIndex : index
        %txt_idx = tensor.extract %txt[%s_idx] : tensor<256xi8>
        %whileCond2 = arith.cmpi eq, %pat_idx, %txt_idx : i8

        %resultIdx = scf.while (%whileIdx = %idx) : (i32) -> i32 {
            %whileCondtemp = arith.andi  %whileCond1, %whileCond2 : i1
            %whileCond = arith.andi %whileCondtemp, %cond1 : i1

            // Violation: scf.while uses %cond whose value depends on %input
            scf.condition(%whileCond) %arg1 : i16
        } do {
            ^bb0(%arg2: i16):
                %tempIdx = arith.subi %while_idx, %c_1 : i32
                scf.yield %tempIdx : i32
        } // end scf.while
        
        
        %cond = arith.cmpi slt, %resultIdx, %c_0 : i32
        %forRet, %forS = scf.if %cond -> (tensor<256xi1>, i32) {
            %true = arith.constant 1 : i1
            %sIdx = index.casts %whileS : i32 to index
            %ifRet = tensor.insert %true into %whileRet[%sIdx] : tensor<256xi1>
            %s_m = arith.addi %whileS, %m : i32
            %s_mIdx = index.casts %s_m : i32 to index
            %txt_s_m = tensor.extract %txt[%s_mIdx] : tensor<256xi8>
            %txt_s_mIdx = index.casts %txt_s_m : i8 to index
            %ifCond1 = arith.cmpi slt, %s_m, %n : i32
            %s_if = scf.if %ifCond1 -> (i32) {
                %badChar_txt_s_m = tensor.extract %badchar[%txt_s_mIdx] : tensor<256xi32>
                %m_badChar = arith.subi %m, %badChar_txt_s_m : i32
                scf.yield %m_badChar : i32
            } else {
                scf.yield %c_1 : i32
            } // end scf.if %ifCond1

            // [Left off here] Trying to figure out how to return 2 things from scf.if
            scf.yield %ifRet, %s_if : (i1) -> (tensor<256xi1>, i32) 
        } else{
            %txt_s_idx= tensor.extract %txt[%s_idx] : tensor<256xi8>
            %txt_s_idxIndex = index.casts %txt_s_idx : i8 to index
            %badChar_txt_s_idx = tensor.extract %badchar[%txt_s_idxIndex] : tensor<256xi32>
            %s_shift = arith.subi %resultIdx, %badChar_txt_s_idx : i32
            %ifCond1 = arith.cmpi slt, %c_1, %s_shift : i32
            %s_else = scf.if %ifCond1 -> i32 {
                scf.yield %s_shift : i32
            } else {
                scf.yield %c_1 : i32
            } // end scf.if %cond1
            scf.yield %whileRet, %s_else : (tensor<256xi1>,i32)
        } // end scf.if %cond
        
        %s_add = arith.addi %whileS, %forS : i32
        affine.yield %forRet, %forS : tensor<256xi1>, i32
    }

    return %newRet : tensor<256xi1>

 }