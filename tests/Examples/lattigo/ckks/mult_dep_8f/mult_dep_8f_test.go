package multdep8f

import (
	"math"
	"testing"
)

func TestMultDep(t *testing.T) {
	evaluator, params, ecd, enc, dec := mult_dep__configure()

	// x = 0.5, expected: x^8 = 0.00390625
	var x float32 = 0.5
	expected := float32(0.00390625)

	ct0 := mult_dep__encrypt__arg0(evaluator, params, ecd, enc, x)

	resultCt := mult_dep(evaluator, params, ecd, ct0)

	result := mult_dep__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.001)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.6f != %.6f", result, expected)
	}
}
