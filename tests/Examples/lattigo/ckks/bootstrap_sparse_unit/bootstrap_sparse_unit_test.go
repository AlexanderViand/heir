package bootstrapsparseunit

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
)

// Table-driven probe of lattigo bootstrap configurations, mirroring the
// residual parameters HEIR generates (LogN 16, q0 2^55, scale 2^45).
// The HEIR-emitted config (btp LogSlots 13, ct Cols 13) corrupts values;
// these cases discriminate which knob is at fault.
func TestBootstrapConfigs(t *testing.T) {
	cases := []struct {
		name        string
		btpLogSlots *int
		ctCols      int
	}{
		{"sparse13_ct13", utils.Pointy(13), 13}, // HEIR's current emission
		{"full15_ct15", utils.Pointy(15), 15},   // full-slot sanity
		{"default_ct15", nil, 15},               // lattigo defaults, full ct
		{"full15_ct13", utils.Pointy(15), 13},   // full bootstrapper, sparse ct
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			param, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
				LogN:            16,
				LogQ:            []int{55, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
				LogP:            []int{61, 61, 61, 61, 61},
				LogDefaultScale: 45,
			})
			if err != nil {
				t.Fatal(err)
			}

			btpLit := bootstrapping.ParametersLiteral{LogN: utils.Pointy(16)}
			if tc.btpLogSlots != nil {
				btpLit.LogSlots = tc.btpLogSlots
			}
			btpParams, err := bootstrapping.NewParametersFromLiteral(param, btpLit)
			if err != nil {
				t.Fatal(err)
			}

			kgen := rlwe.NewKeyGenerator(param)
			sk, pk := kgen.GenKeyPairNew()
			enc := ckks.NewEncoder(param)
			encryptor := rlwe.NewEncryptor(param, pk)
			decryptor := rlwe.NewDecryptor(param, sk)

			btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
			if err != nil {
				t.Fatal(err)
			}
			btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
			if err != nil {
				t.Fatal(err)
			}

			slots := 1 << tc.ctCols
			values := make([]float64, slots)
			for i := range values {
				values[i] = -0.6 + 0.05*float64(i%16)
			}
			pt := ckks.NewPlaintext(param, 0)
			pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: tc.ctCols}
			if err := enc.Encode(values, pt); err != nil {
				t.Fatal(err)
			}
			ct, err := encryptor.EncryptNew(pt)
			if err != nil {
				t.Fatal(err)
			}

			ctBoot, err := btpEval.Bootstrap(ct)
			if err != nil {
				t.Fatal(err)
			}

			out := make([]float64, slots)
			ptOut := decryptor.DecryptNew(ctBoot)
			if err := enc.Decode(ptOut, out); err != nil {
				t.Fatal(err)
			}

			maxErr := 0.0
			for i := 0; i < 32 && i < slots; i++ {
				if e := math.Abs(out[i] - values[i]); e > maxErr {
					maxErr = e
				}
			}
			t.Logf("%s: max|err| over first 32 slots = %.6f", tc.name, maxErr)
			if maxErr > 0.01 {
				t.Errorf("%s: bootstrap corrupted values, max|err|=%.6f (first slots got %.4f want %.4f)",
					tc.name, maxErr, out[0], values[0])
			}
		})
	}
}
