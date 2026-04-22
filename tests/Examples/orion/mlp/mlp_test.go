package mlp

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func msSince(t time.Time) int64 {
	return time.Since(t).Milliseconds()
}

func TestMLP(t *testing.T) {
	fmt.Println("=== Lattigo Orion MLP Benchmark ===")

	fmt.Print("[1/4] Configuring Lattigo context...")
	t0 := time.Now()
	evaluator, params, encoder, encryptor, decryptor := mlp__configure()
	fmt.Printf(" done (%d ms)\n", msSince(t0))

	slots := params.MaxSlots()

	inputClear := make([]float64, slots)
	for i := range inputClear {
		inputClear[i] = 0.5
	}

	// Cleartext args: row-major flattened weights/biases (arbitrary values for perf testing).
	arg0 := make([]float64, 128*slots) // fc1 weights
	arg1 := make([]float64, slots)     // fc1 bias
	arg2 := make([]float64, 128*slots) // fc2 weights
	arg3 := make([]float64, slots)     // fc2 bias
	arg4 := make([]float64, 137*slots) // fc3 weights
	arg5 := make([]float64, slots)     // fc3 bias
	for i := range arg0 {
		arg0[i] = 0.01
	}
	for i := range arg2 {
		arg2[i] = 0.01
	}
	for i := range arg4 {
		arg4[i] = 0.01
	}

	fmt.Print("[2/4] Encrypting input...")
	t0 = time.Now()
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	if err := encoder.Encode(inputClear, pt); err != nil {
		t.Fatal(err)
	}
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf(" done (%d ms)\n", msSince(t0))

	// Warmup absorbs one-time costs (Lattigo evaluator caches, GC warmup,
	// any lazy init). Its result is reused as the sink for the timed loop
	// so the last iteration leaves a valid ciphertext for the decrypt phase.
	fmt.Print("[3/4] Running Orion MLP inference (warmup + 5 runs)...")
	resultCt := mlp(evaluator, params, encoder, ct, arg0, arg1, arg2, arg3, arg4, arg5)

	const kReps = 5
	runsMs := make([]int64, 0, kReps)
	for i := 0; i < kReps; i++ {
		s := time.Now()
		resultCt = mlp(evaluator, params, encoder, ct, arg0, arg1, arg2, arg3, arg4, arg5)
		runsMs = append(runsMs, msSince(s))
	}
	sort.Slice(runsMs, func(i, j int) bool { return runsMs[i] < runsMs[j] })
	fmt.Printf(" done (min=%d ms, median=%d ms, max=%d ms)\n",
		runsMs[0], runsMs[kReps/2], runsMs[kReps-1])

	fmt.Print("[4/4] Decrypting result...")
	t0 = time.Now()
	resultPt := decryptor.DecryptNew(resultCt)
	resultFloat64 := make([]float64, slots)
	if err := encoder.Decode(resultPt, resultFloat64); err != nil {
		t.Fatal(err)
	}
	fmt.Printf(" done (%d ms)\n", msSince(t0))

	fmt.Println("\nOutput (first 10 values):")
	for i := 0; i < 10 && i < len(resultFloat64); i++ {
		fmt.Printf("  [%d] = %g\n", i, resultFloat64[i])
	}
	fmt.Println("\n=== Lattigo Orion MLP Complete ===")
}
