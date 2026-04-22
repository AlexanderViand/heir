package mlp

import (
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func mlp(evaluator *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, ct *rlwe.Ciphertext, v0 []float64, v1 []float64, v2 []float64, v3 []float64, v4 []float64, v5 []float64) *rlwe.Ciphertext {
	v6 := int64(128)
	v7 := int64(256)
	v8 := int64(512)
	v9 := int64(1024)
	v10 := int64(2048)
	ct1_diags_idx := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}
	ct1_diags := make(lintrans.Diagonals[float64])
	for i, diagIndex := range ct1_diags_idx {
		ct1_diags[diagIndex] = v0[i*4096 : (i+1)*4096]
	}
	ct1_params := lintrans.Parameters{
		DiagonalsIndexList:        ct1_diags.DiagonalsIndexList(),
		LevelQ:                    5,
		LevelP:                    evaluator.GetRLWEParameters().MaxLevelP(),
		Scale:                     rlwe.NewScale(evaluator.GetRLWEParameters().Q()[5]),
		LogDimensions:             ct.LogDimensions,
		LogBabyStepGiantStepRatio: 0,
	}
	ct1_lt := lintrans.NewTransformation(evaluator.GetRLWEParameters(), ct1_params)
	err0 := lintrans.Encode[float64](encoder, ct1_diags, ct1_lt)
	if err0 != nil {
		panic(err0)
	}
	ct1_lteval := lintrans.NewEvaluator(evaluator)
	ct1, err0 := ct1_lteval.EvaluateNew(ct, ct1_lt)
	if err0 != nil {
		panic(err0)
	}
	ct2, err1 := evaluator.RotateNew(ct1, int(v10))
	if err1 != nil {
		panic(err1)
	}
	ct3, err2 := evaluator.AddNew(ct2, ct1)
	if err2 != nil {
		panic(err2)
	}
	ct4, err3 := evaluator.RotateNew(ct3, int(v9))
	if err3 != nil {
		panic(err3)
	}
	ct5, err4 := evaluator.AddNew(ct4, ct3)
	if err4 != nil {
		panic(err4)
	}
	ct6, err5 := evaluator.RotateNew(ct5, int(v8))
	if err5 != nil {
		panic(err5)
	}
	ct7, err6 := evaluator.AddNew(ct6, ct5)
	if err6 != nil {
		panic(err6)
	}
	ct8, err7 := evaluator.RotateNew(ct7, int(v7))
	if err7 != nil {
		panic(err7)
	}
	ct9, err8 := evaluator.AddNew(ct8, ct7)
	if err8 != nil {
		panic(err8)
	}
	ct10, err9 := evaluator.RotateNew(ct9, int(v6))
	if err9 != nil {
		panic(err9)
	}
	ct11, err10 := evaluator.AddNew(ct10, ct9)
	if err10 != nil {
		panic(err10)
	}
	pt := ckks.NewPlaintext(param, param.MaxLevel())
	pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: 12}
	pt.Scale = param.NewScale(67108864)
	encoder.Encode(v1, pt)
	ct12, err11 := evaluator.AddNew(ct11, pt)
	if err11 != nil {
		panic(err11)
	}
	ct13, err12 := evaluator.MulNew(ct12, ct12)
	if err12 != nil {
		panic(err12)
	}
	ct14, err13 := evaluator.RelinearizeNew(ct13)
	if err13 != nil {
		panic(err13)
	}
	ct15_diags_idx := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}
	ct15_diags := make(lintrans.Diagonals[float64])
	for i, diagIndex := range ct15_diags_idx {
		ct15_diags[diagIndex] = v2[i*4096 : (i+1)*4096]
	}
	ct15_params := lintrans.Parameters{
		DiagonalsIndexList:        ct15_diags.DiagonalsIndexList(),
		LevelQ:                    5,
		LevelP:                    evaluator.GetRLWEParameters().MaxLevelP(),
		Scale:                     rlwe.NewScale(evaluator.GetRLWEParameters().Q()[5]),
		LogDimensions:             ct14.LogDimensions,
		LogBabyStepGiantStepRatio: 0,
	}
	ct15_lt := lintrans.NewTransformation(evaluator.GetRLWEParameters(), ct15_params)
	err14 := lintrans.Encode[float64](encoder, ct15_diags, ct15_lt)
	if err14 != nil {
		panic(err14)
	}
	ct15_lteval := lintrans.NewEvaluator(evaluator)
	ct15, err14 := ct15_lteval.EvaluateNew(ct14, ct15_lt)
	if err14 != nil {
		panic(err14)
	}
	ct16, err15 := evaluator.RotateNew(ct15, int(v10))
	if err15 != nil {
		panic(err15)
	}
	ct17, err16 := evaluator.AddNew(ct16, ct15)
	if err16 != nil {
		panic(err16)
	}
	ct18, err17 := evaluator.RotateNew(ct17, int(v9))
	if err17 != nil {
		panic(err17)
	}
	ct19, err18 := evaluator.AddNew(ct18, ct17)
	if err18 != nil {
		panic(err18)
	}
	ct20, err19 := evaluator.RotateNew(ct19, int(v8))
	if err19 != nil {
		panic(err19)
	}
	ct21, err20 := evaluator.AddNew(ct20, ct19)
	if err20 != nil {
		panic(err20)
	}
	ct22, err21 := evaluator.RotateNew(ct21, int(v7))
	if err21 != nil {
		panic(err21)
	}
	ct23, err22 := evaluator.AddNew(ct22, ct21)
	if err22 != nil {
		panic(err22)
	}
	ct24, err23 := evaluator.RotateNew(ct23, int(v6))
	if err23 != nil {
		panic(err23)
	}
	ct25, err24 := evaluator.AddNew(ct24, ct23)
	if err24 != nil {
		panic(err24)
	}
	pt2 := ckks.NewPlaintext(param, param.MaxLevel())
	pt2.LogDimensions = ring.Dimensions{Rows: 0, Cols: 12}
	pt2.Scale = param.NewScale(4503599627370496)
	encoder.Encode(v3, pt2)
	ct26, err25 := evaluator.AddNew(ct25, pt2)
	if err25 != nil {
		panic(err25)
	}
	ct27 := ct26.CopyNew()
	err26 := evaluator.Rescale(ct26, ct27)
	if err26 != nil {
		panic(err26)
	}
	ct28, err27 := evaluator.MulNew(ct27, ct27)
	if err27 != nil {
		panic(err27)
	}
	ct29, err28 := evaluator.RelinearizeNew(ct28)
	if err28 != nil {
		panic(err28)
	}
	ct30_diags_idx := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095}
	ct30_diags := make(lintrans.Diagonals[float64])
	for i, diagIndex := range ct30_diags_idx {
		ct30_diags[diagIndex] = v4[i*4096 : (i+1)*4096]
	}
	ct30_params := lintrans.Parameters{
		DiagonalsIndexList:        ct30_diags.DiagonalsIndexList(),
		LevelQ:                    4,
		LevelP:                    evaluator.GetRLWEParameters().MaxLevelP(),
		Scale:                     rlwe.NewScale(evaluator.GetRLWEParameters().Q()[4]),
		LogDimensions:             ct29.LogDimensions,
		LogBabyStepGiantStepRatio: 0,
	}
	ct30_lt := lintrans.NewTransformation(evaluator.GetRLWEParameters(), ct30_params)
	err29 := lintrans.Encode[float64](encoder, ct30_diags, ct30_lt)
	if err29 != nil {
		panic(err29)
	}
	ct30_lteval := lintrans.NewEvaluator(evaluator)
	ct30, err29 := ct30_lteval.EvaluateNew(ct29, ct30_lt)
	if err29 != nil {
		panic(err29)
	}
	pt4 := ckks.NewPlaintext(param, param.MaxLevel())
	pt4.LogDimensions = ring.Dimensions{Rows: 0, Cols: 12}
	pt4.Scale = param.NewScale(4503599627370496)
	encoder.Encode(v5, pt4)
	ct31, err30 := evaluator.AddNew(ct30, pt4)
	if err30 != nil {
		panic(err30)
	}
	ct32 := ct31.CopyNew()
	err31 := evaluator.Rescale(ct31, ct32)
	if err31 != nil {
		panic(err31)
	}
	return ct32
}
func mlp__encrypt__arg0(_ *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, encryptor *rlwe.Encryptor, v0 []float64) *rlwe.Ciphertext {
	pt := ckks.NewPlaintext(param, param.MaxLevel())
	pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: 12}
	pt.Scale = param.NewScale(67108864)
	encoder.Encode(v0, pt)
	ct, err32 := encryptor.EncryptNew(pt)
	if err32 != nil {
		panic(err32)
	}
	return ct
}
func mlp__decrypt__result0(_ *ckks.Evaluator, _ ckks.Parameters, encoder *ckks.Encoder, decryptor *rlwe.Decryptor, ct *rlwe.Ciphertext) []float64 {
	v0 := make([]float64, 4096)
	pt := decryptor.DecryptNew(ct)
	v0_float64 := make([]float64, len(v0))
	encoder.Decode(pt, v0_float64)
	v1_converted := make([]float64, len(v0))
	for i := range v0 {
		v1_converted[i] = float64(v0_float64[i])
	}
	v1 := v1_converted
	return v1
}
func mlp__configure() (*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor) {
	param, err34 := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            14,
		Q:               []uint64{1073643521, 67731457, 66813953, 67502081, 67043329, 67239937},
		P:               []uint64{1152921504607338497},
		LogDefaultScale: 26,
	})
	if err34 != nil {
		panic(err34)
	}
	encoder := ckks.NewEncoder(param)
	kgen := rlwe.NewKeyGenerator(param)
	sk, pk := kgen.GenKeyPairNew()
	encryptor := rlwe.NewEncryptor(param, pk)
	decryptor := rlwe.NewDecryptor(param, sk)
	rk := kgen.GenRelinearizationKeyNew(sk)
	gk := kgen.GenGaloisKeyNew(7937, sk)
	gk1 := kgen.GenGaloisKeyNew(24577, sk)
	gk2 := kgen.GenGaloisKeyNew(12589, sk)
	gk3 := kgen.GenGaloisKeyNew(28673, sk)
	gk4 := kgen.GenGaloisKeyNew(30049, sk)
	gk5 := kgen.GenGaloisKeyNew(13409, sk)
	gk6 := kgen.GenGaloisKeyNew(25, sk)
	gk7 := kgen.GenGaloisKeyNew(30721, sk)
	gk8 := kgen.GenGaloisKeyNew(28609, sk)
	gk9 := kgen.GenGaloisKeyNew(20161, sk)
	gk10 := kgen.GenGaloisKeyNew(625, sk)
	gk11 := kgen.GenGaloisKeyNew(31745, sk)
	gk12 := kgen.GenGaloisKeyNew(3361, sk)
	gk13 := kgen.GenGaloisKeyNew(3105, sk)
	gk14 := kgen.GenGaloisKeyNew(20001, sk)
	gk15 := kgen.GenGaloisKeyNew(15873, sk)
	gk16 := kgen.GenGaloisKeyNew(15625, sk)
	gk17 := kgen.GenGaloisKeyNew(28545, sk)
	gk18 := kgen.GenGaloisKeyNew(3713, sk)
	gk19 := kgen.GenGaloisKeyNew(5, sk)
	gk20 := kgen.GenGaloisKeyNew(30177, sk)
	gk21 := kgen.GenGaloisKeyNew(13537, sk)
	gk22 := kgen.GenGaloisKeyNew(125, sk)
	gk23 := kgen.GenGaloisKeyNew(32577, sk)
	gk24 := kgen.GenGaloisKeyNew(24129, sk)
	gk25 := kgen.GenGaloisKeyNew(24641, sk)
	gk26 := kgen.GenGaloisKeyNew(28065, sk)
	gk27 := kgen.GenGaloisKeyNew(3125, sk)
	gk28 := kgen.GenGaloisKeyNew(27809, sk)
	ekset := rlwe.NewMemEvaluationKeySet(rk, gk, gk1, gk2, gk3, gk4, gk5, gk6, gk7, gk8, gk9, gk10, gk11, gk12, gk13, gk14, gk15, gk16, gk17, gk18, gk19, gk20, gk21, gk22, gk23, gk24, gk25, gk26, gk27, gk28)
	evaluator := ckks.NewEvaluator(param, ekset)
	return evaluator, param, encoder, encryptor, decryptor
}
