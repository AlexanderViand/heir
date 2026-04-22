
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "UserInterface.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "core/Parameter.h"
#include "extension/BootContext.h"
#include "extension/EvalPoly.h"
#include "extension/Hoist.h"
#include "extension/LinearTransform.h"

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

Ct mlp(CtxPtr ctx, Enc& encoder, UI& ui, const Ct& ct,
       const std::vector<double>& v0, const std::vector<double>& v1,
       const std::vector<double>& v2, const std::vector<double>& v3,
       const std::vector<double>& v4, const std::vector<double>& v5) {
  int64_t v6 = 2048;
  int64_t v7 = 1024;
  int64_t v8 = 512;
  int64_t v9 = 256;
  int64_t v10 = 128;
  const auto& evk_map = ui.GetEvkMap();
  StripedMatrix ct1_mat(4096, 4096);
  {
    auto* data = v0.data();
    ct1_mat[0] = std::vector<Complex>(data + 0, data + 4096);
    ct1_mat[1] = std::vector<Complex>(data + 4096, data + 8192);
    ct1_mat[2] = std::vector<Complex>(data + 8192, data + 12288);
    ct1_mat[3] = std::vector<Complex>(data + 12288, data + 16384);
    ct1_mat[4] = std::vector<Complex>(data + 16384, data + 20480);
    ct1_mat[5] = std::vector<Complex>(data + 20480, data + 24576);
    ct1_mat[6] = std::vector<Complex>(data + 24576, data + 28672);
    ct1_mat[7] = std::vector<Complex>(data + 28672, data + 32768);
    ct1_mat[8] = std::vector<Complex>(data + 32768, data + 36864);
    ct1_mat[9] = std::vector<Complex>(data + 36864, data + 40960);
    ct1_mat[10] = std::vector<Complex>(data + 40960, data + 45056);
    ct1_mat[11] = std::vector<Complex>(data + 45056, data + 49152);
    ct1_mat[12] = std::vector<Complex>(data + 49152, data + 53248);
    ct1_mat[13] = std::vector<Complex>(data + 53248, data + 57344);
    ct1_mat[14] = std::vector<Complex>(data + 57344, data + 61440);
    ct1_mat[15] = std::vector<Complex>(data + 61440, data + 65536);
    ct1_mat[16] = std::vector<Complex>(data + 65536, data + 69632);
    ct1_mat[17] = std::vector<Complex>(data + 69632, data + 73728);
    ct1_mat[18] = std::vector<Complex>(data + 73728, data + 77824);
    ct1_mat[19] = std::vector<Complex>(data + 77824, data + 81920);
    ct1_mat[20] = std::vector<Complex>(data + 81920, data + 86016);
    ct1_mat[21] = std::vector<Complex>(data + 86016, data + 90112);
    ct1_mat[22] = std::vector<Complex>(data + 90112, data + 94208);
    ct1_mat[23] = std::vector<Complex>(data + 94208, data + 98304);
    ct1_mat[24] = std::vector<Complex>(data + 98304, data + 102400);
    ct1_mat[25] = std::vector<Complex>(data + 102400, data + 106496);
    ct1_mat[26] = std::vector<Complex>(data + 106496, data + 110592);
    ct1_mat[27] = std::vector<Complex>(data + 110592, data + 114688);
    ct1_mat[28] = std::vector<Complex>(data + 114688, data + 118784);
    ct1_mat[29] = std::vector<Complex>(data + 118784, data + 122880);
    ct1_mat[30] = std::vector<Complex>(data + 122880, data + 126976);
    ct1_mat[31] = std::vector<Complex>(data + 126976, data + 131072);
    ct1_mat[32] = std::vector<Complex>(data + 131072, data + 135168);
    ct1_mat[33] = std::vector<Complex>(data + 135168, data + 139264);
    ct1_mat[34] = std::vector<Complex>(data + 139264, data + 143360);
    ct1_mat[35] = std::vector<Complex>(data + 143360, data + 147456);
    ct1_mat[36] = std::vector<Complex>(data + 147456, data + 151552);
    ct1_mat[37] = std::vector<Complex>(data + 151552, data + 155648);
    ct1_mat[38] = std::vector<Complex>(data + 155648, data + 159744);
    ct1_mat[39] = std::vector<Complex>(data + 159744, data + 163840);
    ct1_mat[40] = std::vector<Complex>(data + 163840, data + 167936);
    ct1_mat[41] = std::vector<Complex>(data + 167936, data + 172032);
    ct1_mat[42] = std::vector<Complex>(data + 172032, data + 176128);
    ct1_mat[43] = std::vector<Complex>(data + 176128, data + 180224);
    ct1_mat[44] = std::vector<Complex>(data + 180224, data + 184320);
    ct1_mat[45] = std::vector<Complex>(data + 184320, data + 188416);
    ct1_mat[46] = std::vector<Complex>(data + 188416, data + 192512);
    ct1_mat[47] = std::vector<Complex>(data + 192512, data + 196608);
    ct1_mat[48] = std::vector<Complex>(data + 196608, data + 200704);
    ct1_mat[49] = std::vector<Complex>(data + 200704, data + 204800);
    ct1_mat[50] = std::vector<Complex>(data + 204800, data + 208896);
    ct1_mat[51] = std::vector<Complex>(data + 208896, data + 212992);
    ct1_mat[52] = std::vector<Complex>(data + 212992, data + 217088);
    ct1_mat[53] = std::vector<Complex>(data + 217088, data + 221184);
    ct1_mat[54] = std::vector<Complex>(data + 221184, data + 225280);
    ct1_mat[55] = std::vector<Complex>(data + 225280, data + 229376);
    ct1_mat[56] = std::vector<Complex>(data + 229376, data + 233472);
    ct1_mat[57] = std::vector<Complex>(data + 233472, data + 237568);
    ct1_mat[58] = std::vector<Complex>(data + 237568, data + 241664);
    ct1_mat[59] = std::vector<Complex>(data + 241664, data + 245760);
    ct1_mat[60] = std::vector<Complex>(data + 245760, data + 249856);
    ct1_mat[61] = std::vector<Complex>(data + 249856, data + 253952);
    ct1_mat[62] = std::vector<Complex>(data + 253952, data + 258048);
    ct1_mat[63] = std::vector<Complex>(data + 258048, data + 262144);
    ct1_mat[64] = std::vector<Complex>(data + 262144, data + 266240);
    ct1_mat[65] = std::vector<Complex>(data + 266240, data + 270336);
    ct1_mat[66] = std::vector<Complex>(data + 270336, data + 274432);
    ct1_mat[67] = std::vector<Complex>(data + 274432, data + 278528);
    ct1_mat[68] = std::vector<Complex>(data + 278528, data + 282624);
    ct1_mat[69] = std::vector<Complex>(data + 282624, data + 286720);
    ct1_mat[70] = std::vector<Complex>(data + 286720, data + 290816);
    ct1_mat[71] = std::vector<Complex>(data + 290816, data + 294912);
    ct1_mat[72] = std::vector<Complex>(data + 294912, data + 299008);
    ct1_mat[73] = std::vector<Complex>(data + 299008, data + 303104);
    ct1_mat[74] = std::vector<Complex>(data + 303104, data + 307200);
    ct1_mat[75] = std::vector<Complex>(data + 307200, data + 311296);
    ct1_mat[76] = std::vector<Complex>(data + 311296, data + 315392);
    ct1_mat[77] = std::vector<Complex>(data + 315392, data + 319488);
    ct1_mat[78] = std::vector<Complex>(data + 319488, data + 323584);
    ct1_mat[79] = std::vector<Complex>(data + 323584, data + 327680);
    ct1_mat[80] = std::vector<Complex>(data + 327680, data + 331776);
    ct1_mat[81] = std::vector<Complex>(data + 331776, data + 335872);
    ct1_mat[82] = std::vector<Complex>(data + 335872, data + 339968);
    ct1_mat[83] = std::vector<Complex>(data + 339968, data + 344064);
    ct1_mat[84] = std::vector<Complex>(data + 344064, data + 348160);
    ct1_mat[85] = std::vector<Complex>(data + 348160, data + 352256);
    ct1_mat[86] = std::vector<Complex>(data + 352256, data + 356352);
    ct1_mat[87] = std::vector<Complex>(data + 356352, data + 360448);
    ct1_mat[88] = std::vector<Complex>(data + 360448, data + 364544);
    ct1_mat[89] = std::vector<Complex>(data + 364544, data + 368640);
    ct1_mat[90] = std::vector<Complex>(data + 368640, data + 372736);
    ct1_mat[91] = std::vector<Complex>(data + 372736, data + 376832);
    ct1_mat[92] = std::vector<Complex>(data + 376832, data + 380928);
    ct1_mat[93] = std::vector<Complex>(data + 380928, data + 385024);
    ct1_mat[94] = std::vector<Complex>(data + 385024, data + 389120);
    ct1_mat[95] = std::vector<Complex>(data + 389120, data + 393216);
    ct1_mat[96] = std::vector<Complex>(data + 393216, data + 397312);
    ct1_mat[97] = std::vector<Complex>(data + 397312, data + 401408);
    ct1_mat[98] = std::vector<Complex>(data + 401408, data + 405504);
    ct1_mat[99] = std::vector<Complex>(data + 405504, data + 409600);
    ct1_mat[100] = std::vector<Complex>(data + 409600, data + 413696);
    ct1_mat[101] = std::vector<Complex>(data + 413696, data + 417792);
    ct1_mat[102] = std::vector<Complex>(data + 417792, data + 421888);
    ct1_mat[103] = std::vector<Complex>(data + 421888, data + 425984);
    ct1_mat[104] = std::vector<Complex>(data + 425984, data + 430080);
    ct1_mat[105] = std::vector<Complex>(data + 430080, data + 434176);
    ct1_mat[106] = std::vector<Complex>(data + 434176, data + 438272);
    ct1_mat[107] = std::vector<Complex>(data + 438272, data + 442368);
    ct1_mat[108] = std::vector<Complex>(data + 442368, data + 446464);
    ct1_mat[109] = std::vector<Complex>(data + 446464, data + 450560);
    ct1_mat[110] = std::vector<Complex>(data + 450560, data + 454656);
    ct1_mat[111] = std::vector<Complex>(data + 454656, data + 458752);
    ct1_mat[112] = std::vector<Complex>(data + 458752, data + 462848);
    ct1_mat[113] = std::vector<Complex>(data + 462848, data + 466944);
    ct1_mat[114] = std::vector<Complex>(data + 466944, data + 471040);
    ct1_mat[115] = std::vector<Complex>(data + 471040, data + 475136);
    ct1_mat[116] = std::vector<Complex>(data + 475136, data + 479232);
    ct1_mat[117] = std::vector<Complex>(data + 479232, data + 483328);
    ct1_mat[118] = std::vector<Complex>(data + 483328, data + 487424);
    ct1_mat[119] = std::vector<Complex>(data + 487424, data + 491520);
    ct1_mat[120] = std::vector<Complex>(data + 491520, data + 495616);
    ct1_mat[121] = std::vector<Complex>(data + 495616, data + 499712);
    ct1_mat[122] = std::vector<Complex>(data + 499712, data + 503808);
    ct1_mat[123] = std::vector<Complex>(data + 503808, data + 507904);
    ct1_mat[124] = std::vector<Complex>(data + 507904, data + 512000);
    ct1_mat[125] = std::vector<Complex>(data + 512000, data + 516096);
    ct1_mat[126] = std::vector<Complex>(data + 516096, data + 520192);
    ct1_mat[127] = std::vector<Complex>(data + 520192, data + 524288);
  }
  LinearTransform<word> ct1_lt(ctx, ct1_mat, 5, ctx->param_.GetScale(5 - 1), 8,
                               16, 0);
  Ct ct1;
  ct1_lt.Evaluate(ctx, ct1, ct, evk_map);
  Ct ct2;
  ctx->HRot(ct2, ct1, ui.GetRotationKey(v6), v6);
  Ct ct3;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct2.GetScale();
    double rhs_scale = ct1.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct3 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct3, ct2, ct1);
  Ct ct4;
  ctx->HRot(ct4, ct3, ui.GetRotationKey(v7), v7);
  Ct ct5;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct4.GetScale();
    double rhs_scale = ct3.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct5 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct5, ct4, ct3);
  Ct ct6;
  ctx->HRot(ct6, ct5, ui.GetRotationKey(v8), v8);
  Ct ct7;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct6.GetScale();
    double rhs_scale = ct5.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct7 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct7, ct6, ct5);
  Ct ct8;
  ctx->HRot(ct8, ct7, ui.GetRotationKey(v9), v9);
  Ct ct9;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct8.GetScale();
    double rhs_scale = ct7.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct9 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct9, ct8, ct7);
  Ct ct10;
  ctx->HRot(ct10, ct9, ui.GetRotationKey(v10), v10);
  Ct ct11;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct10.GetScale();
    double rhs_scale = ct9.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct11 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct11, ct10, ct9);
  Pt pt;
  std::vector<Complex> pt_complex(v1.begin(), v1.end());
  encoder.Encode(pt, 4, ctx->param_.GetScale(4), pt_complex);
  Ct ct12;
  pt.SetScale(ct11.GetScale());
  ctx->Add(ct12, ct11, pt);
  Ct ct13;
  ctx->Mult(ct13, ct12, ct12);
  const auto& evk5 = ui.GetMultiplicationKey();
  Ct ct14;
  ctx->Relinearize(ct14, ct13, evk5);
  const auto& evk_map1 = ui.GetEvkMap();
  StripedMatrix ct15_mat(4096, 4096);
  {
    auto* data = v2.data();
    ct15_mat[0] = std::vector<Complex>(data + 0, data + 4096);
    ct15_mat[1] = std::vector<Complex>(data + 4096, data + 8192);
    ct15_mat[2] = std::vector<Complex>(data + 8192, data + 12288);
    ct15_mat[3] = std::vector<Complex>(data + 12288, data + 16384);
    ct15_mat[4] = std::vector<Complex>(data + 16384, data + 20480);
    ct15_mat[5] = std::vector<Complex>(data + 20480, data + 24576);
    ct15_mat[6] = std::vector<Complex>(data + 24576, data + 28672);
    ct15_mat[7] = std::vector<Complex>(data + 28672, data + 32768);
    ct15_mat[8] = std::vector<Complex>(data + 32768, data + 36864);
    ct15_mat[9] = std::vector<Complex>(data + 36864, data + 40960);
    ct15_mat[10] = std::vector<Complex>(data + 40960, data + 45056);
    ct15_mat[11] = std::vector<Complex>(data + 45056, data + 49152);
    ct15_mat[12] = std::vector<Complex>(data + 49152, data + 53248);
    ct15_mat[13] = std::vector<Complex>(data + 53248, data + 57344);
    ct15_mat[14] = std::vector<Complex>(data + 57344, data + 61440);
    ct15_mat[15] = std::vector<Complex>(data + 61440, data + 65536);
    ct15_mat[16] = std::vector<Complex>(data + 65536, data + 69632);
    ct15_mat[17] = std::vector<Complex>(data + 69632, data + 73728);
    ct15_mat[18] = std::vector<Complex>(data + 73728, data + 77824);
    ct15_mat[19] = std::vector<Complex>(data + 77824, data + 81920);
    ct15_mat[20] = std::vector<Complex>(data + 81920, data + 86016);
    ct15_mat[21] = std::vector<Complex>(data + 86016, data + 90112);
    ct15_mat[22] = std::vector<Complex>(data + 90112, data + 94208);
    ct15_mat[23] = std::vector<Complex>(data + 94208, data + 98304);
    ct15_mat[24] = std::vector<Complex>(data + 98304, data + 102400);
    ct15_mat[25] = std::vector<Complex>(data + 102400, data + 106496);
    ct15_mat[26] = std::vector<Complex>(data + 106496, data + 110592);
    ct15_mat[27] = std::vector<Complex>(data + 110592, data + 114688);
    ct15_mat[28] = std::vector<Complex>(data + 114688, data + 118784);
    ct15_mat[29] = std::vector<Complex>(data + 118784, data + 122880);
    ct15_mat[30] = std::vector<Complex>(data + 122880, data + 126976);
    ct15_mat[31] = std::vector<Complex>(data + 126976, data + 131072);
    ct15_mat[32] = std::vector<Complex>(data + 131072, data + 135168);
    ct15_mat[33] = std::vector<Complex>(data + 135168, data + 139264);
    ct15_mat[34] = std::vector<Complex>(data + 139264, data + 143360);
    ct15_mat[35] = std::vector<Complex>(data + 143360, data + 147456);
    ct15_mat[36] = std::vector<Complex>(data + 147456, data + 151552);
    ct15_mat[37] = std::vector<Complex>(data + 151552, data + 155648);
    ct15_mat[38] = std::vector<Complex>(data + 155648, data + 159744);
    ct15_mat[39] = std::vector<Complex>(data + 159744, data + 163840);
    ct15_mat[40] = std::vector<Complex>(data + 163840, data + 167936);
    ct15_mat[41] = std::vector<Complex>(data + 167936, data + 172032);
    ct15_mat[42] = std::vector<Complex>(data + 172032, data + 176128);
    ct15_mat[43] = std::vector<Complex>(data + 176128, data + 180224);
    ct15_mat[44] = std::vector<Complex>(data + 180224, data + 184320);
    ct15_mat[45] = std::vector<Complex>(data + 184320, data + 188416);
    ct15_mat[46] = std::vector<Complex>(data + 188416, data + 192512);
    ct15_mat[47] = std::vector<Complex>(data + 192512, data + 196608);
    ct15_mat[48] = std::vector<Complex>(data + 196608, data + 200704);
    ct15_mat[49] = std::vector<Complex>(data + 200704, data + 204800);
    ct15_mat[50] = std::vector<Complex>(data + 204800, data + 208896);
    ct15_mat[51] = std::vector<Complex>(data + 208896, data + 212992);
    ct15_mat[52] = std::vector<Complex>(data + 212992, data + 217088);
    ct15_mat[53] = std::vector<Complex>(data + 217088, data + 221184);
    ct15_mat[54] = std::vector<Complex>(data + 221184, data + 225280);
    ct15_mat[55] = std::vector<Complex>(data + 225280, data + 229376);
    ct15_mat[56] = std::vector<Complex>(data + 229376, data + 233472);
    ct15_mat[57] = std::vector<Complex>(data + 233472, data + 237568);
    ct15_mat[58] = std::vector<Complex>(data + 237568, data + 241664);
    ct15_mat[59] = std::vector<Complex>(data + 241664, data + 245760);
    ct15_mat[60] = std::vector<Complex>(data + 245760, data + 249856);
    ct15_mat[61] = std::vector<Complex>(data + 249856, data + 253952);
    ct15_mat[62] = std::vector<Complex>(data + 253952, data + 258048);
    ct15_mat[63] = std::vector<Complex>(data + 258048, data + 262144);
    ct15_mat[64] = std::vector<Complex>(data + 262144, data + 266240);
    ct15_mat[65] = std::vector<Complex>(data + 266240, data + 270336);
    ct15_mat[66] = std::vector<Complex>(data + 270336, data + 274432);
    ct15_mat[67] = std::vector<Complex>(data + 274432, data + 278528);
    ct15_mat[68] = std::vector<Complex>(data + 278528, data + 282624);
    ct15_mat[69] = std::vector<Complex>(data + 282624, data + 286720);
    ct15_mat[70] = std::vector<Complex>(data + 286720, data + 290816);
    ct15_mat[71] = std::vector<Complex>(data + 290816, data + 294912);
    ct15_mat[72] = std::vector<Complex>(data + 294912, data + 299008);
    ct15_mat[73] = std::vector<Complex>(data + 299008, data + 303104);
    ct15_mat[74] = std::vector<Complex>(data + 303104, data + 307200);
    ct15_mat[75] = std::vector<Complex>(data + 307200, data + 311296);
    ct15_mat[76] = std::vector<Complex>(data + 311296, data + 315392);
    ct15_mat[77] = std::vector<Complex>(data + 315392, data + 319488);
    ct15_mat[78] = std::vector<Complex>(data + 319488, data + 323584);
    ct15_mat[79] = std::vector<Complex>(data + 323584, data + 327680);
    ct15_mat[80] = std::vector<Complex>(data + 327680, data + 331776);
    ct15_mat[81] = std::vector<Complex>(data + 331776, data + 335872);
    ct15_mat[82] = std::vector<Complex>(data + 335872, data + 339968);
    ct15_mat[83] = std::vector<Complex>(data + 339968, data + 344064);
    ct15_mat[84] = std::vector<Complex>(data + 344064, data + 348160);
    ct15_mat[85] = std::vector<Complex>(data + 348160, data + 352256);
    ct15_mat[86] = std::vector<Complex>(data + 352256, data + 356352);
    ct15_mat[87] = std::vector<Complex>(data + 356352, data + 360448);
    ct15_mat[88] = std::vector<Complex>(data + 360448, data + 364544);
    ct15_mat[89] = std::vector<Complex>(data + 364544, data + 368640);
    ct15_mat[90] = std::vector<Complex>(data + 368640, data + 372736);
    ct15_mat[91] = std::vector<Complex>(data + 372736, data + 376832);
    ct15_mat[92] = std::vector<Complex>(data + 376832, data + 380928);
    ct15_mat[93] = std::vector<Complex>(data + 380928, data + 385024);
    ct15_mat[94] = std::vector<Complex>(data + 385024, data + 389120);
    ct15_mat[95] = std::vector<Complex>(data + 389120, data + 393216);
    ct15_mat[96] = std::vector<Complex>(data + 393216, data + 397312);
    ct15_mat[97] = std::vector<Complex>(data + 397312, data + 401408);
    ct15_mat[98] = std::vector<Complex>(data + 401408, data + 405504);
    ct15_mat[99] = std::vector<Complex>(data + 405504, data + 409600);
    ct15_mat[100] = std::vector<Complex>(data + 409600, data + 413696);
    ct15_mat[101] = std::vector<Complex>(data + 413696, data + 417792);
    ct15_mat[102] = std::vector<Complex>(data + 417792, data + 421888);
    ct15_mat[103] = std::vector<Complex>(data + 421888, data + 425984);
    ct15_mat[104] = std::vector<Complex>(data + 425984, data + 430080);
    ct15_mat[105] = std::vector<Complex>(data + 430080, data + 434176);
    ct15_mat[106] = std::vector<Complex>(data + 434176, data + 438272);
    ct15_mat[107] = std::vector<Complex>(data + 438272, data + 442368);
    ct15_mat[108] = std::vector<Complex>(data + 442368, data + 446464);
    ct15_mat[109] = std::vector<Complex>(data + 446464, data + 450560);
    ct15_mat[110] = std::vector<Complex>(data + 450560, data + 454656);
    ct15_mat[111] = std::vector<Complex>(data + 454656, data + 458752);
    ct15_mat[112] = std::vector<Complex>(data + 458752, data + 462848);
    ct15_mat[113] = std::vector<Complex>(data + 462848, data + 466944);
    ct15_mat[114] = std::vector<Complex>(data + 466944, data + 471040);
    ct15_mat[115] = std::vector<Complex>(data + 471040, data + 475136);
    ct15_mat[116] = std::vector<Complex>(data + 475136, data + 479232);
    ct15_mat[117] = std::vector<Complex>(data + 479232, data + 483328);
    ct15_mat[118] = std::vector<Complex>(data + 483328, data + 487424);
    ct15_mat[119] = std::vector<Complex>(data + 487424, data + 491520);
    ct15_mat[120] = std::vector<Complex>(data + 491520, data + 495616);
    ct15_mat[121] = std::vector<Complex>(data + 495616, data + 499712);
    ct15_mat[122] = std::vector<Complex>(data + 499712, data + 503808);
    ct15_mat[123] = std::vector<Complex>(data + 503808, data + 507904);
    ct15_mat[124] = std::vector<Complex>(data + 507904, data + 512000);
    ct15_mat[125] = std::vector<Complex>(data + 512000, data + 516096);
    ct15_mat[126] = std::vector<Complex>(data + 516096, data + 520192);
    ct15_mat[127] = std::vector<Complex>(data + 520192, data + 524288);
  }
  LinearTransform<word> ct15_lt(ctx, ct15_mat, 4, ctx->param_.GetScale(4 - 1),
                                8, 16, 0);
  Ct ct15;
  ct15_lt.Evaluate(ctx, ct15, ct14, evk_map1);
  Ct ct16;
  ctx->HRot(ct16, ct15, ui.GetRotationKey(v6), v6);
  Ct ct17;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct16.GetScale();
    double rhs_scale = ct15.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct17 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct17, ct16, ct15);
  Ct ct18;
  ctx->HRot(ct18, ct17, ui.GetRotationKey(v7), v7);
  Ct ct19;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct18.GetScale();
    double rhs_scale = ct17.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct19 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct19, ct18, ct17);
  Ct ct20;
  ctx->HRot(ct20, ct19, ui.GetRotationKey(v8), v8);
  Ct ct21;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct20.GetScale();
    double rhs_scale = ct19.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct21 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct21, ct20, ct19);
  Ct ct22;
  ctx->HRot(ct22, ct21, ui.GetRotationKey(v9), v9);
  Ct ct23;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct22.GetScale();
    double rhs_scale = ct21.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct23 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct23, ct22, ct21);
  Ct ct24;
  ctx->HRot(ct24, ct23, ui.GetRotationKey(v10), v10);
  Ct ct25;
  if (std::getenv("HEIR_CHEDDAR_DEBUG_SCALES")) {
    double lhs_scale = ct24.GetScale();
    double rhs_scale = ct23.GetScale();
    if (std::abs(lhs_scale - rhs_scale) >= 1e-12 * lhs_scale) {
      std::cerr << "[heir-cheddar] add ct25 scale mismatch lhs=" << lhs_scale
                << " rhs=" << rhs_scale << std::endl;
    }
  }
  ctx->Add(ct25, ct24, ct23);
  Pt pt1;
  std::vector<Complex> pt1_complex(v3.begin(), v3.end());
  encoder.Encode(pt1, 3, ctx->param_.GetScale(3) * ctx->param_.GetScale(3),
                 pt1_complex);
  Ct ct26;
  pt1.SetScale(ct25.GetScale());
  ctx->Add(ct26, ct25, pt1);
  Ct ct27;
  ctx->Rescale(ct27, ct26);
  Ct ct28;
  ctx->Mult(ct28, ct27, ct27);
  const auto& evk11 = ui.GetMultiplicationKey();
  Ct ct29;
  ctx->Relinearize(ct29, ct28, evk11);
  const auto& evk_map2 = ui.GetEvkMap();
  StripedMatrix ct30_mat(4096, 4096);
  {
    auto* data = v4.data();
    ct30_mat[0] = std::vector<Complex>(data + 0, data + 4096);
    ct30_mat[1] = std::vector<Complex>(data + 4096, data + 8192);
    ct30_mat[2] = std::vector<Complex>(data + 8192, data + 12288);
    ct30_mat[3] = std::vector<Complex>(data + 12288, data + 16384);
    ct30_mat[4] = std::vector<Complex>(data + 16384, data + 20480);
    ct30_mat[5] = std::vector<Complex>(data + 20480, data + 24576);
    ct30_mat[6] = std::vector<Complex>(data + 24576, data + 28672);
    ct30_mat[7] = std::vector<Complex>(data + 28672, data + 32768);
    ct30_mat[8] = std::vector<Complex>(data + 32768, data + 36864);
    ct30_mat[9] = std::vector<Complex>(data + 36864, data + 40960);
    ct30_mat[10] = std::vector<Complex>(data + 40960, data + 45056);
    ct30_mat[11] = std::vector<Complex>(data + 45056, data + 49152);
    ct30_mat[12] = std::vector<Complex>(data + 49152, data + 53248);
    ct30_mat[13] = std::vector<Complex>(data + 53248, data + 57344);
    ct30_mat[14] = std::vector<Complex>(data + 57344, data + 61440);
    ct30_mat[15] = std::vector<Complex>(data + 61440, data + 65536);
    ct30_mat[16] = std::vector<Complex>(data + 65536, data + 69632);
    ct30_mat[17] = std::vector<Complex>(data + 69632, data + 73728);
    ct30_mat[18] = std::vector<Complex>(data + 73728, data + 77824);
    ct30_mat[19] = std::vector<Complex>(data + 77824, data + 81920);
    ct30_mat[20] = std::vector<Complex>(data + 81920, data + 86016);
    ct30_mat[21] = std::vector<Complex>(data + 86016, data + 90112);
    ct30_mat[22] = std::vector<Complex>(data + 90112, data + 94208);
    ct30_mat[23] = std::vector<Complex>(data + 94208, data + 98304);
    ct30_mat[24] = std::vector<Complex>(data + 98304, data + 102400);
    ct30_mat[25] = std::vector<Complex>(data + 102400, data + 106496);
    ct30_mat[26] = std::vector<Complex>(data + 106496, data + 110592);
    ct30_mat[27] = std::vector<Complex>(data + 110592, data + 114688);
    ct30_mat[28] = std::vector<Complex>(data + 114688, data + 118784);
    ct30_mat[29] = std::vector<Complex>(data + 118784, data + 122880);
    ct30_mat[30] = std::vector<Complex>(data + 122880, data + 126976);
    ct30_mat[31] = std::vector<Complex>(data + 126976, data + 131072);
    ct30_mat[32] = std::vector<Complex>(data + 131072, data + 135168);
    ct30_mat[33] = std::vector<Complex>(data + 135168, data + 139264);
    ct30_mat[34] = std::vector<Complex>(data + 139264, data + 143360);
    ct30_mat[35] = std::vector<Complex>(data + 143360, data + 147456);
    ct30_mat[36] = std::vector<Complex>(data + 147456, data + 151552);
    ct30_mat[37] = std::vector<Complex>(data + 151552, data + 155648);
    ct30_mat[38] = std::vector<Complex>(data + 155648, data + 159744);
    ct30_mat[39] = std::vector<Complex>(data + 159744, data + 163840);
    ct30_mat[40] = std::vector<Complex>(data + 163840, data + 167936);
    ct30_mat[41] = std::vector<Complex>(data + 167936, data + 172032);
    ct30_mat[42] = std::vector<Complex>(data + 172032, data + 176128);
    ct30_mat[43] = std::vector<Complex>(data + 176128, data + 180224);
    ct30_mat[44] = std::vector<Complex>(data + 180224, data + 184320);
    ct30_mat[45] = std::vector<Complex>(data + 184320, data + 188416);
    ct30_mat[46] = std::vector<Complex>(data + 188416, data + 192512);
    ct30_mat[47] = std::vector<Complex>(data + 192512, data + 196608);
    ct30_mat[48] = std::vector<Complex>(data + 196608, data + 200704);
    ct30_mat[49] = std::vector<Complex>(data + 200704, data + 204800);
    ct30_mat[50] = std::vector<Complex>(data + 204800, data + 208896);
    ct30_mat[51] = std::vector<Complex>(data + 208896, data + 212992);
    ct30_mat[52] = std::vector<Complex>(data + 212992, data + 217088);
    ct30_mat[53] = std::vector<Complex>(data + 217088, data + 221184);
    ct30_mat[54] = std::vector<Complex>(data + 221184, data + 225280);
    ct30_mat[55] = std::vector<Complex>(data + 225280, data + 229376);
    ct30_mat[56] = std::vector<Complex>(data + 229376, data + 233472);
    ct30_mat[57] = std::vector<Complex>(data + 233472, data + 237568);
    ct30_mat[58] = std::vector<Complex>(data + 237568, data + 241664);
    ct30_mat[59] = std::vector<Complex>(data + 241664, data + 245760);
    ct30_mat[60] = std::vector<Complex>(data + 245760, data + 249856);
    ct30_mat[61] = std::vector<Complex>(data + 249856, data + 253952);
    ct30_mat[62] = std::vector<Complex>(data + 253952, data + 258048);
    ct30_mat[63] = std::vector<Complex>(data + 258048, data + 262144);
    ct30_mat[64] = std::vector<Complex>(data + 262144, data + 266240);
    ct30_mat[65] = std::vector<Complex>(data + 266240, data + 270336);
    ct30_mat[66] = std::vector<Complex>(data + 270336, data + 274432);
    ct30_mat[67] = std::vector<Complex>(data + 274432, data + 278528);
    ct30_mat[68] = std::vector<Complex>(data + 278528, data + 282624);
    ct30_mat[69] = std::vector<Complex>(data + 282624, data + 286720);
    ct30_mat[70] = std::vector<Complex>(data + 286720, data + 290816);
    ct30_mat[71] = std::vector<Complex>(data + 290816, data + 294912);
    ct30_mat[72] = std::vector<Complex>(data + 294912, data + 299008);
    ct30_mat[73] = std::vector<Complex>(data + 299008, data + 303104);
    ct30_mat[74] = std::vector<Complex>(data + 303104, data + 307200);
    ct30_mat[75] = std::vector<Complex>(data + 307200, data + 311296);
    ct30_mat[76] = std::vector<Complex>(data + 311296, data + 315392);
    ct30_mat[77] = std::vector<Complex>(data + 315392, data + 319488);
    ct30_mat[78] = std::vector<Complex>(data + 319488, data + 323584);
    ct30_mat[79] = std::vector<Complex>(data + 323584, data + 327680);
    ct30_mat[80] = std::vector<Complex>(data + 327680, data + 331776);
    ct30_mat[81] = std::vector<Complex>(data + 331776, data + 335872);
    ct30_mat[82] = std::vector<Complex>(data + 335872, data + 339968);
    ct30_mat[83] = std::vector<Complex>(data + 339968, data + 344064);
    ct30_mat[84] = std::vector<Complex>(data + 344064, data + 348160);
    ct30_mat[85] = std::vector<Complex>(data + 348160, data + 352256);
    ct30_mat[86] = std::vector<Complex>(data + 352256, data + 356352);
    ct30_mat[87] = std::vector<Complex>(data + 356352, data + 360448);
    ct30_mat[88] = std::vector<Complex>(data + 360448, data + 364544);
    ct30_mat[89] = std::vector<Complex>(data + 364544, data + 368640);
    ct30_mat[90] = std::vector<Complex>(data + 368640, data + 372736);
    ct30_mat[91] = std::vector<Complex>(data + 372736, data + 376832);
    ct30_mat[92] = std::vector<Complex>(data + 376832, data + 380928);
    ct30_mat[93] = std::vector<Complex>(data + 380928, data + 385024);
    ct30_mat[94] = std::vector<Complex>(data + 385024, data + 389120);
    ct30_mat[95] = std::vector<Complex>(data + 389120, data + 393216);
    ct30_mat[96] = std::vector<Complex>(data + 393216, data + 397312);
    ct30_mat[97] = std::vector<Complex>(data + 397312, data + 401408);
    ct30_mat[98] = std::vector<Complex>(data + 401408, data + 405504);
    ct30_mat[99] = std::vector<Complex>(data + 405504, data + 409600);
    ct30_mat[100] = std::vector<Complex>(data + 409600, data + 413696);
    ct30_mat[101] = std::vector<Complex>(data + 413696, data + 417792);
    ct30_mat[102] = std::vector<Complex>(data + 417792, data + 421888);
    ct30_mat[103] = std::vector<Complex>(data + 421888, data + 425984);
    ct30_mat[104] = std::vector<Complex>(data + 425984, data + 430080);
    ct30_mat[105] = std::vector<Complex>(data + 430080, data + 434176);
    ct30_mat[106] = std::vector<Complex>(data + 434176, data + 438272);
    ct30_mat[107] = std::vector<Complex>(data + 438272, data + 442368);
    ct30_mat[108] = std::vector<Complex>(data + 442368, data + 446464);
    ct30_mat[109] = std::vector<Complex>(data + 446464, data + 450560);
    ct30_mat[110] = std::vector<Complex>(data + 450560, data + 454656);
    ct30_mat[111] = std::vector<Complex>(data + 454656, data + 458752);
    ct30_mat[112] = std::vector<Complex>(data + 458752, data + 462848);
    ct30_mat[113] = std::vector<Complex>(data + 462848, data + 466944);
    ct30_mat[114] = std::vector<Complex>(data + 466944, data + 471040);
    ct30_mat[115] = std::vector<Complex>(data + 471040, data + 475136);
    ct30_mat[116] = std::vector<Complex>(data + 475136, data + 479232);
    ct30_mat[117] = std::vector<Complex>(data + 479232, data + 483328);
    ct30_mat[118] = std::vector<Complex>(data + 483328, data + 487424);
    ct30_mat[119] = std::vector<Complex>(data + 487424, data + 491520);
    ct30_mat[120] = std::vector<Complex>(data + 491520, data + 495616);
    ct30_mat[121] = std::vector<Complex>(data + 495616, data + 499712);
    ct30_mat[122] = std::vector<Complex>(data + 499712, data + 503808);
    ct30_mat[123] = std::vector<Complex>(data + 503808, data + 507904);
    ct30_mat[124] = std::vector<Complex>(data + 507904, data + 512000);
    ct30_mat[125] = std::vector<Complex>(data + 512000, data + 516096);
    ct30_mat[126] = std::vector<Complex>(data + 516096, data + 520192);
    ct30_mat[127] = std::vector<Complex>(data + 520192, data + 524288);
    ct30_mat[4087] = std::vector<Complex>(data + 524288, data + 528384);
    ct30_mat[4088] = std::vector<Complex>(data + 528384, data + 532480);
    ct30_mat[4089] = std::vector<Complex>(data + 532480, data + 536576);
    ct30_mat[4090] = std::vector<Complex>(data + 536576, data + 540672);
    ct30_mat[4091] = std::vector<Complex>(data + 540672, data + 544768);
    ct30_mat[4092] = std::vector<Complex>(data + 544768, data + 548864);
    ct30_mat[4093] = std::vector<Complex>(data + 548864, data + 552960);
    ct30_mat[4094] = std::vector<Complex>(data + 552960, data + 557056);
    ct30_mat[4095] = std::vector<Complex>(data + 557056, data + 561152);
  }
  LinearTransform<word> ct30_lt(ctx, ct30_mat, 2, ctx->param_.GetScale(2 - 1),
                                16, 9, 4087);
  Ct ct30;
  ct30_lt.Evaluate(ctx, ct30, ct29, evk_map2);
  Pt pt2;
  std::vector<Complex> pt2_complex(v5.begin(), v5.end());
  encoder.Encode(pt2, 1, ctx->param_.GetScale(1) * ctx->param_.GetScale(1),
                 pt2_complex);
  Ct ct31;
  pt2.SetScale(ct30.GetScale());
  ctx->Add(ct31, ct30, pt2);
  Ct ct32;
  ctx->Rescale(ct32, ct31);
  return ct32;
}

Ct mlp__encrypt__arg0(CtxPtr ctx, Enc& encoder, UI& ui,
                      const std::vector<double>& v0, UI& ui1) {
  Pt pt;
  std::vector<Complex> pt_complex(v0.begin(), v0.end());
  encoder.Encode(pt, 5, ctx->param_.GetScale(5), pt_complex);
  Ct ct;
  ui.Encrypt(ct, pt);
  return ct;
}

std::vector<double> mlp__decrypt__result0(CtxPtr ctx, Enc& encoder, UI& ui,
                                          const Ct& ct, UI& ui1) {
  Pt pt;
  ui.Decrypt(pt, ct);
  std::vector<Complex> v0_complex;
  encoder.Decode(v0_complex, pt);
  std::vector<double> v0(v0_complex.size());
  for (size_t i = 0; i < v0_complex.size(); ++i) v0[i] = v0_complex[i].real();
  return v0;
}

std::tuple<CtxPtr, UI> __configure() {
  static std::vector<word> main_primes = {1073643521ULL, 67731457ULL,
                                          66813953ULL,   67502081ULL,
                                          67043329ULL,   67239937ULL};
  static std::vector<word> aux_primes = {1152921504607338497ULL};
  static std::vector<std::pair<int, int>> level_config = []() {
    std::vector<std::pair<int, int>> lc;
    for (int i = 1; i <= static_cast<int>(main_primes.size()); ++i)
      lc.push_back({i, 0});
    return lc;
  }();
  static Param param(14, static_cast<double>(1ULL << 26),
                     static_cast<int>(main_primes.size()) - 1, level_config,
                     main_primes, aux_primes);
  auto ctx = Context<word>::Create(param);
  UI ui(ctx);
  ui.PrepareRotationKey(1, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(2, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(3, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(4, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(5, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(6, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(7, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(8, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(9, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(10, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(11, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(12, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(13, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(14, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(15, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(16, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(24, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(32, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(40, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(48, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(56, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(64, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(72, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(80, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(88, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(96, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(104, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(112, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(120, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(128, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(256, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(512, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(1024, static_cast<int>(main_primes.size()) - 1);
  ui.PrepareRotationKey(2048, static_cast<int>(main_primes.size()) - 1);
  return {ctx, std::move(ui)};
}
