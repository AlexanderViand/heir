package relu2compositedbg

import (
	"testing"
)

// Not an assertion test: drives the debug-instrumented circuit so the
// __heir_debug hook traces every op's decrypted values to stderr (visible in
// the bazel test log with --test_output=all).
func TestRelu2CompositeDebugTrace(t *testing.T) {
	btp, evaluator, params, ecd, enc, dec := Relu2_composite__configure()

	arg0 := make([]float32, 16)
	for i := 0; i < 16; i++ {
		arg0[i] = float32(i)*0.25 - 2.0
	}

	ct0 := Relu2_composite__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Relu2_composite(btp, evaluator, params, ecd, dec, ct0)
	result := Relu2_composite__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	t.Logf("final result: %v", result)
}
