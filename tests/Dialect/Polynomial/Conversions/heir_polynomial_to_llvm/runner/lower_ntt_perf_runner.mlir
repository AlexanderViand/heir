// RUN: heir-opt %s --heir-polynomial-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_NTT < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

#cycl = #polynomial.int_polynomial<1 + x**65536>
!coeff_ty = !mod_arith.int<786433:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl>
#root = #polynomial.primitive_root<value=283965:i32, degree=131072:i32>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func @test_poly_ntt() {
  %rand_coeffs = arith.constant dense<[82466, 284102, 230668, 726464, 689117, 138714, 365947, 689485, 446187, 553091, 780346, 411551, 281642, 459186, 220603, 654511, 59260, 764961, 16666, 685313, 185494, 496060, 115841, 252683, 240479, 594524, 416523, 15494, 599774, 602830, 747394, 26308, 114781, 332978, 68665, 68778, 284638, 391464, 520202, 457892, 311791, 592237, 431, 355444, 570435, 281966, 455422, 734625, 410275, 585604, 342610, 252937, 500915, 185942, 350739, 503473, 503734, 299480, 274387, 731481, 221528, 248143, 42922, 376637, 375877, 7330, 52495, 392501, 506041, 585691, 419166, 572553, 443650, 195220, 778694, 189969, 70437, 767512, 88039, 102191, 533715, 71537, 299910, 421484, 778028, 331670, 49663, 476567, 770219, 142409, 17515, 334750, 191359, 756857, 138776, 189860, 310377, 269922, 123155, 322425, 196530, 572988, 387881, 693640, 649776, 180305, 190535, 671521, 500677, 572629, 639529, 7917, 496775, 754601, 342810, 352620, 566306, 192070, 411491, 314939, 65466, 746207, 74857, 323561, 684125, 410365, 403750, 86735, 609582, 589955, 168067, 182598, 372487, 426906, 280440, 93888, 459279, 340251, 61319, 425866, 96770, 757022, 599341, 550851, 150353, 279359, 514129, 470867, 634484, 71518, 359067, 663842, 358108, 255984, 743819, 498698, 338391, 618044, 205702, 536924, 551736, 198363, 87452, 46470, 585248, 53857, 443489, 387573, 549213, 461919, 341020, 557268, 104295, 510410, 422064, 41391, 399312, 356900, 82705, 43467, 289311, 403719, 348615, 325627, 401499, 501031, 41324, 743151, 558730, 487703, 692004, 691852, 215177, 212269, 251863, 170361, 768210, 250989, 195699, 531621, 255600, 370065, 26502, 201868, 125163, 363038, 700427, 30487, 391928, 52856, 452124, 759967, 531073, 116689, 362042, 255716, 315098, 391743, 38415, 367529, 504177, 343228, 496991, 442943, 362269, 661169, 394207, 476331, 361445, 67053, 730805, 597631, 3379, 238251, 159057, 287526, 671488, 618631, 525038, 695869, 91737, 47054, 509029, 227973, 127210, 689448, 370642, 79893, 400573, 206914, 428506, 778137, 563541, 644030, 376996, 612443]> : tensor<256xi32>
  %c42 = arith.constant 42 : i32
  %full = tensor.splat %c42 : tensor<65536xi32>
  %insert_rand0 = tensor.insert_slice %rand_coeffs into %full[0] [256] [1] : tensor<256xi32> into tensor<65536xi32>
  %insert_rand1 = tensor.insert_slice %rand_coeffs into %insert_rand0[65280] [256] [1] : tensor<256xi32> into tensor<65536xi32>
  %rand1_enc = mod_arith.encapsulate %insert_rand1 : tensor<65536xi32> -> tensor<65536x!coeff_ty>
  %poly = polynomial.from_tensor %rand1_enc : tensor<65536x!coeff_ty> -> !poly_ty
  %0 = polynomial.ntt %poly {root=#root} : !poly_ty -> tensor<65536x!coeff_ty, #ring>

  // Insert casts so that intt(ntt()) does not get folded away during polynomial
  // canonicalization
  %cast = tensor.cast %0 : tensor<65536x!coeff_ty, #ring> to tensor<65536x!coeff_ty>
  %cast_back = tensor.cast %cast : tensor<65536x!coeff_ty> to tensor<65536x!coeff_ty, #ring>

  %1 = polynomial.intt %cast_back {root=#root} : tensor<65536x!coeff_ty, #ring> -> !poly_ty

  %2 = polynomial.to_tensor %1 : !poly_ty -> tensor<65536x!coeff_ty>
  %ext2 = mod_arith.extract %2 : tensor<65536x!coeff_ty> -> tensor<65536xi32>
  %4 = bufferization.to_buffer %ext2 : tensor<65536xi32> to memref<65536xi32>
  %U = memref.cast %4 : memref<65536xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// Checking the first and last 8 values:
// CHECK_TEST_POLY_NTT: [82466, 284102, 230668, 726464, 689117, 138714, 365947, 689485,
// CHECK_TEST_POLY_NTT: 79893, 400573, 206914, 428506, 778137, 563541, 644030, 376996, 612443]
