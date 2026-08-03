package relu2compositedbg

import (
	"fmt"
	"os"
	"strconv"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// __heir_debug is the hook HEIR's Lattigo emitter calls for each inserted
// debug port. Decrypts the ciphertext and prints name/op/level/scale and the
// leading slots to stderr. Prints raw with %v (not json) so NaN/Inf survive.
func __heir_debug(evaluator *ckks.Evaluator, param ckks.Parameters, encoder *ckks.Encoder, decryptor *rlwe.Decryptor, ctObj any, debugAttrMap map[string]string) {
	var ct *rlwe.Ciphertext
	switch v := ctObj.(type) {
	case *rlwe.Ciphertext:
		ct = v
	case []*rlwe.Ciphertext:
		if len(v) == 0 {
			return
		}
		ct = v[0]
	default:
		return
	}
	if ct == nil {
		return
	}
	msgSize := 16
	if s, ok := debugAttrMap["message.size"]; ok {
		if n, err := strconv.Atoi(s); err == nil && n > 0 && n < 64 {
			msgSize = n
		}
	}
	value := make([]float64, msgSize)
	pt := decryptor.DecryptNew(ct)
	if err := encoder.Decode(pt, value); err != nil {
		fmt.Fprintf(os.Stderr, "HEIRDBG %s decode-err %v\n", debugAttrMap["debug.name"], err)
		return
	}
	n := 8
	if len(value) < n {
		n = len(value)
	}
	fmt.Fprintf(os.Stderr, "HEIRDBG name=%s op=%s lvl=%d logscale=%.1f vals=%v\n",
		debugAttrMap["debug.name"], debugAttrMap["asm.op_name"], ct.Level(), ct.Scale.Log2(), value[:n])
}
