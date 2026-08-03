package convrelucomposite

import (
	"math"
	"testing"
)

func TestConvReluComposite(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv_relu_composite__configure()

	arg0 := make([]float32, 16)
	for i := 0; i < 16; i++ {
		arg0[i] = float32(i) * 0.1
	}

	expected := []float32{
		0.0, 0.35, 0.25, 0.15, 0.05, 0, 0, 1.35,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	ct0 := Conv_relu_composite__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Conv_relu_composite(evaluator, params, ecd, ct0)
	result := Conv_relu_composite__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("index %d: got %.4f want %.4f", i, result[i], expected[i])
		}
	}
}
