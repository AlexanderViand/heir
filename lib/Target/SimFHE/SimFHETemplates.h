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
import tabulate


def run_workload(fn):
    schemes = params.get_alg_params()
    headers = ["params", "total_ops", "mult", "dram_total"]
    rows = []
    for sp in schemes:
        args = [sp.fresh_ctxt] * (fn.__code__.co_argcount - 1)
        args.append(sp)
        stats = fn(*args)
        rows.append(
            [
                sp,
                stats.sw.total_ops,
                stats.sw.mult,
                stats.arch.dram_total_rdwr_small,
            ]
        )
    print(tabulate.tabulate(rows, headers=headers))
)python";

}  // namespace simfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_SIMFHE_SIMFHETEMPLATES_H_
