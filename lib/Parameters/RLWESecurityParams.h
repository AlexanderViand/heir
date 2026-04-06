#ifndef LIB_PARAMETERS_RLWESECURITYPARAMS_H_
#define LIB_PARAMETERS_RLWESECURITYPARAMS_H_

namespace mlir {
namespace heir {

// struct for recording the maximal Q for each ring dim
// under certain security condition.
struct RLWESecurityParam {
  int ringDim;
  int logMaxQ;
};

// compute ringDim given logPQ under 128-bit classic security.
// When useOpenFHEBounds is true, uses OpenFHE's ternary secret key
// distribution bounds (slightly more generous) to ensure generated
// parameters are accepted by OpenFHE's runtime validation.
int computeRingDim(int logPQ, int minRingDim, bool useOpenFHEBounds = false);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_RLWESECURITYPARAMS_H_
