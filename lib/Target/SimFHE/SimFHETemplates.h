#ifndef LIB_TARGET_SIMFHE_SIMFHETEMPLATES_H_
#define LIB_TARGET_SIMFHE_SIMFHETEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace simfhe {

constexpr std::string_view kModulePrelude = R"python(
import params
import evaluator
from perf_counter import PerfCounter
)python";

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_SIMFHE_SIMFHETEMPLATES_H_
