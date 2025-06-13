// RUN: heir-translate
// %S/../../CKKS/Conversions/ckks_to_openfhe/ckks_to_openfhe.mlir --emit-simfhe
// | FileCheck %s

// CHECK: import params
// CHECK-NEXT: import evaluator
// CHECK-NEXT: from perf_counter import PerfCounter
// CHECK-NEXT: import tabulate
// CHECK: def run_workload(

// CHECK: def test_ops(
// CHECK: stats = PerfCounter()
// CHECK: stats += evaluator.negate([[ARG0:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES0:[A-Za-z0-9_]+]] = [[ARG0]]
// CHECK: stats += evaluator.add([[ARG1:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES1:[A-Za-z0-9_]+]] = [[ARG1]]
// CHECK: stats += evaluator.subtract([[ARG2:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES2:[A-Za-z0-9_]+]] = [[ARG2]]
// CHECK: stats += evaluator.multiply([[ARG3:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES3:[A-Za-z0-9_]+]] = [[ARG3]]
// CHECK: stats += evaluator.rotate([[ARG4:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES4:[A-Za-z0-9_]+]] = [[ARG4]]
// CHECK: stats += evaluator.add_plain([[ARG5:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES5:[A-Za-z0-9_]+]] = [[ARG5]]
// CHECK: stats += evaluator.subtract_plain([[ARG6:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES6:[A-Za-z0-9_]+]] = [[ARG6]]
// CHECK: stats += evaluator.multiply_plain([[ARG7:[A-Za-z0-9_]+]],
// scheme_params.arch_param) CHECK-NEXT: [[RES7:[A-Za-z0-9_]+]] = [[ARG7]]
// CHECK: return stats

// CHECK: def test_relin(
// CHECK: stats = PerfCounter()
// CHECK: stats += evaluator.key_switch([[ARG:[A-Za-z0-9_]+]], [[ARG]],
// scheme_params.arch_param) CHECK-NEXT: [[RES:[A-Za-z0-9_]+]] = [[ARG]] CHECK:
// return stats

// CHECK: if __name__ == "__main__":
// CHECK: run_workload(test_ops)
// CHECK: run_workload(test_relin)
