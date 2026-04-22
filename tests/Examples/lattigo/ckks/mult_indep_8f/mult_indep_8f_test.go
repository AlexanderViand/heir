package multindep8f

import (
	"math"
	"testing"
)

func TestMultIndep(t *testing.T) {
	evaluator, params, ecd, enc, dec := mult_indep__configure()

	// All inputs = 0.5, expected: 0.5^8 = 0.00390625
	var arg0, arg1, arg2, arg3 float32 = 0.5, 0.5, 0.5, 0.5
	var arg4, arg5, arg6, arg7 float32 = 0.5, 0.5, 0.5, 0.5
	expected := float32(0.00390625)

	ct0 := mult_indep__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := mult_indep__encrypt__arg1(evaluator, params, ecd, enc, arg1)
	ct2 := mult_indep__encrypt__arg2(evaluator, params, ecd, enc, arg2)
	ct3 := mult_indep__encrypt__arg3(evaluator, params, ecd, enc, arg3)
	ct4 := mult_indep__encrypt__arg4(evaluator, params, ecd, enc, arg4)
	ct5 := mult_indep__encrypt__arg5(evaluator, params, ecd, enc, arg5)
	ct6 := mult_indep__encrypt__arg6(evaluator, params, ecd, enc, arg6)
	ct7 := mult_indep__encrypt__arg7(evaluator, params, ecd, enc, arg7)

	resultCt := mult_indep(evaluator, params, ecd, ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7)

	result := mult_indep__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	errorThreshold := float64(0.001)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.6f != %.6f", result, expected)
	}
}
