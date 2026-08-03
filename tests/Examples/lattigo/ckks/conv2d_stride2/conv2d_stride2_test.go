package conv2dstride2

import (
	"math"
	"testing"
)

func TestConv2dStride2(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv2d_stride2__configure()

	arg0 := make([]float32, 36)
	for i := 0; i < 36; i++ {
		arg0[i] = float32(i) * 0.1
	}

	// Hand-computed: conv2d(input 1x1x6x6 = 0.1*i, stride 2, filters
	// ch0 [[1,-1],[0.5,0.25]], ch1 [[-0.5,1],[1,-1]], ch2
	// [[0.25,0.5],[-0.25,0.75]], ch3 zeros; bias [0.1,-0.2,0.3,0]).
	expected := []float32{
		0.4750, 0.6250, 0.7750, 1.3750, 1.5250, 1.6750, 2.2750, 2.4250, 2.5750,
		-0.2000, -0.1000, 0.0000, 0.4000, 0.5000, 0.6000, 1.0000, 1.1000, 1.2000,
		0.7250, 0.9750, 1.2250, 2.2250, 2.4750, 2.7250, 3.7250, 3.9750, 4.2250,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
	}

	ct0 := Conv2d_stride2__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Conv2d_stride2(evaluator, params, ecd, ct0)
	result := Conv2d_stride2__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.01)
	nbad := 0
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("index %d (ch %d, pos %d): got %.4f want %.4f", i, i/9, i%9, result[i], expected[i])
			nbad++
		}
	}
	if nbad > 0 {
		t.Logf("%d/36 outputs diverge", nbad)
	}
}
