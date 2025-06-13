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
// CHECK: stats += evaluator.negate(
// CHECK-NEXT: v
// CHECK: stats += evaluator.add(
// CHECK-NEXT: v
// CHECK: stats += evaluator.subtract(
// CHECK-NEXT: v
// CHECK: stats += evaluator.multiply(
// CHECK-NEXT: v
// CHECK: stats += evaluator.rotate(
// CHECK-NEXT: v
// CHECK: stats += evaluator.add_plain(
// CHECK-NEXT: v
// CHECK: stats += evaluator.subtract_plain(
// CHECK-NEXT: v
// CHECK: stats += evaluator.multiply_plain(
// CHECK-NEXT: v
// CHECK: return stats

// CHECK: def test_relin(
// CHECK: stats = PerfCounter()
// CHECK: stats += evaluator.key_switch({{.*}}, {{.*}}, arch_params)
// CHECK-NEXT: v
// CHECK: return stats

// CHECK: if __name__ == "__main__":
// CHECK: run_workload(test_ops)
// CHECK: run_workload(test_relin)
