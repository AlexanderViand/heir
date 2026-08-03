package relucomposite

import (
	"math"
	"testing"
)

func TestReluComposite(t *testing.T) {
	evaluator, params, ecd, enc, dec := Relu_composite__configure()

	// 16 values spanning the calibrated domain [-2.07, 2.05].
	arg0 := make([]float32, 16)
	expected := make([]float32, 16)
	for i := 0; i < 16; i++ {
		x := float32(i)*0.25 - 2.0
		arg0[i] = x
		if x > 0 {
			expected[i] = x
		}
	}

	ct0 := Relu_composite__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Relu_composite(evaluator, params, ecd, ct0)
	result := Relu_composite__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// The composite sign is accurate away from 0 but rounds the kink; the
	// x*step form bounds the kink error by |x|*1 near 0.
	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("index %d: got %.4f want %.4f", i, result[i], expected[i])
		}
	}
}
