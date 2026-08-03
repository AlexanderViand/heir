package conv1dncwlintrans

import (
	"math"
	"testing"
)

func TestConv1DLintrans(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv1d_ncw__configure()

	cols := 4 // input elements
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = float32(i)
	}

	expectedSingleChannel := []float32{1, 5}
	expected := make([]float32, 16)
	for c := 0; c < 8; c++ {
		for i := 0; i < 2; i++ {
			expected[c*2+i] = expectedSingleChannel[i]
		}
	}

	ct0 := Conv1d_ncw__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Conv1d_ncw(evaluator, params, ecd, ct0)

	// Debug: dump raw slots of the lintrans output.
	t.Logf("input  ct: level=%d scale=%v", ct0[0].Level(), ct0[0].Scale.Float64())
	t.Logf("output ct: level=%d scale=%v", resultCt[0].Level(), resultCt[0].Scale.Float64())
	rawPt := dec.DecryptNew(resultCt[0])
	rawSlots := make([]float64, 1024)
	if err := ecd.Decode(rawPt, rawSlots); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	t.Logf("slots[0:32]    = %.2f", rawSlots[0:32])
	t.Logf("slots[992:1024] = %.2f", rawSlots[992:1024])

	result := Conv1d_ncw__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.5)
	for i := 0; i < 16; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected[i])
		}
	}
}
