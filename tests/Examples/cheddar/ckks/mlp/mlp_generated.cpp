
#include <cheddar/include/UserInterface.h>
#include <cheddar/include/core/Container.h>
#include <cheddar/include/core/Context.h>
#include <cheddar/include/core/Encode.h>
#include <cheddar/include/core/EvkMap.h>
#include <cheddar/include/core/EvkRequest.h>
#include <cheddar/include/core/Parameter.h>
#include <cheddar/include/extension/BootContext.h>
#include <cheddar/include/extension/EvalPoly.h>
#include <cheddar/include/extension/Hoist.h>
#include <cheddar/include/extension/LinearTransform.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

using namespace cheddar;
using word = uint64_t;
using Ct = Ciphertext<word>;
using Pt = Plaintext<word>;
using Const = Constant<word>;
using Evk = EvaluationKey<word>;
using EvkMapT = EvkMap<word>;
using CtxPtr = std::shared_ptr<Context<word>>;
using Param = Parameter<word>;
using UI = UserInterface<word>;
using Enc = Encoder<word>;
using Complex = std::complex<double>;

Ct mlp(CtxPtr ctx, Enc& encoder, UI ui, Ct ct, std::vector<Complex> v0,
       std::vector<Complex> v1, std::vector<Complex> v2,
       std::vector<Complex> v3, std::vector<Complex> v4,
       std::vector<Complex> v5) {
  const auto& evk_map = ui.GetEvkMap();
  // LinearTransform: diag_indices=array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  // 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
  // 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  // 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
  // 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
  // 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
  // 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
  // 119, 120, 121, 122, 123, 124, 125, 126, 127>, level=5, logBSGS=2
  Ct ct1;  // TODO: implement LinearTransform emission
  const auto& evk = ui.GetRotationKey(2048);
  Ct ct2;
  ctx->HRotAdd(ct2, ct1, ct1, evk, 2048);
  const auto& evk1 = ui.GetRotationKey(1024);
  Ct ct3;
  ctx->HRotAdd(ct3, ct2, ct2, evk1, 1024);
  const auto& evk2 = ui.GetRotationKey(512);
  Ct ct4;
  ctx->HRotAdd(ct4, ct3, ct3, evk2, 512);
  const auto& evk3 = ui.GetRotationKey(256);
  Ct ct5;
  ctx->HRotAdd(ct5, ct4, ct4, evk3, 256);
  const auto& evk4 = ui.GetRotationKey(128);
  Ct ct6;
  ctx->HRotAdd(ct6, ct5, ct5, evk4, 128);
  Pt pt;
  encoder.Encode(pt, 0, 1099511627776, v1);
  Ct ct7;
  ctx->Add(ct7, ct6, pt);
  const auto& evk5 = ui.GetMultiplicationKey();
  Ct ct8;
  ctx->HMult(ct8, ct7, ct7, evk5, true);
  const auto& evk_map1 = ui.GetEvkMap();
  // LinearTransform: diag_indices=array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  // 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
  // 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  // 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
  // 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
  // 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
  // 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
  // 119, 120, 121, 122, 123, 124, 125, 126, 127>, level=3, logBSGS=2
  Ct ct9;  // TODO: implement LinearTransform emission
  const auto& evk6 = ui.GetRotationKey(2048);
  Ct ct10;
  ctx->HRotAdd(ct10, ct9, ct9, evk6, 2048);
  const auto& evk7 = ui.GetRotationKey(1024);
  Ct ct11;
  ctx->HRotAdd(ct11, ct10, ct10, evk7, 1024);
  const auto& evk8 = ui.GetRotationKey(512);
  Ct ct12;
  ctx->HRotAdd(ct12, ct11, ct11, evk8, 512);
  const auto& evk9 = ui.GetRotationKey(256);
  Ct ct13;
  ctx->HRotAdd(ct13, ct12, ct12, evk9, 256);
  const auto& evk10 = ui.GetRotationKey(128);
  Ct ct14;
  ctx->HRotAdd(ct14, ct13, ct13, evk10, 128);
  Pt pt1;
  encoder.Encode(pt1, 0, 1099511627776, v3);
  Ct ct15;
  ctx->Add(ct15, ct14, pt1);
  const auto& evk11 = ui.GetMultiplicationKey();
  Ct ct16;
  ctx->HMult(ct16, ct15, ct15, evk11, true);
  const auto& evk_map2 = ui.GetEvkMap();
  // LinearTransform: diag_indices=array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  // 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
  // 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  // 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
  // 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
  // 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
  // 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
  // 119, 120, 121, 122, 123, 124, 125, 126, 127, 4087, 4088, 4089, 4090, 4091,
  // 4092, 4093, 4094, 4095>, level=1, logBSGS=2
  Ct ct17;  // TODO: implement LinearTransform emission
  Pt pt2;
  encoder.Encode(pt2, 0, 1099511627776, v5);
  Ct ct18;
  ctx->Add(ct18, ct17, pt2);
  return ct18;
}
