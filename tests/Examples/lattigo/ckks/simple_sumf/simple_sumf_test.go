package simplesumf

import (
	"math"
	"testing"
)

func TestSimpleSumf(t *testing.T) {
	evaluator, params, ecd, enc, dec := simple_sum__configure()

	// Input: 32 values from 0.1 to 3.2
	arg0 := make([]float32, 32)
	expected := float32(0.0)
	for i := 0; i < 32; i++ {
		arg0[i] = 0.1 * float32(i+1)
		expected += arg0[i]
	}

	ct0 := simple_sum__encrypt__arg0(evaluator, params, ecd, enc, arg0)

	resultCt := simple_sum(evaluator, params, ecd, ct0)

	result := simple_sum__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.1)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.6f != %.6f", result, expected)
	}
}
