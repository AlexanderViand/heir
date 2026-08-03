package relu2composite

import (
	"math"
	"testing"
)

func TestRelu2Composite(t *testing.T) {
	btp, evaluator, params, ecd, enc, dec := Relu2_composite__configure()

	arg0 := make([]float32, 16)
	expected := make([]float32, 16)
	for i := 0; i < 16; i++ {
		x := float32(i)*0.25 - 2.0
		arg0[i] = x
		y := x
		if y < 0 {
			y = 0
		}
		y = y - 1.0
		if y < 0 {
			y = 0
		}
		expected[i] = y
	}

	ct0 := Relu2_composite__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Relu2_composite(btp, evaluator, params, ecd, ct0)
	result := Relu2_composite__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("index %d: got %.4f want %.4f", i, result[i], expected[i])
		}
	}
}
