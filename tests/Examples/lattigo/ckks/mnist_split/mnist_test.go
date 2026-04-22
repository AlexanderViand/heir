package mnistsplit

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"testing"
	"time"

	"tests/Examples/lattigo/ckks/mnist_split/mnistsplit_utils"
)

const dataBase = "../../../common/mnist/data/"

const (
	modelPath  = dataBase + "/traced_model.pt"
	imagesPath = dataBase + "/t10k-images-idx3-ubyte"
	labelsPath = dataBase + "/t10k-labels-idx1-ubyte"
)

func msSince(t time.Time) int64 {
	return time.Since(t).Milliseconds()
}

func loadWeights(path string) ([][]float32, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	weights := make([][]float32, 4)
	for i := 0; i < 4; i++ {
		f, err := r.Open(fmt.Sprintf("traced_model/data/%d", i))
		if err != nil {
			return nil, err
		}
		defer f.Close()

		data, err := io.ReadAll(f)
		if err != nil {
			return nil, err
		}

		numFloats := len(data) / 4
		weights[i] = make([]float32, numFloats)
		for j := 0; j < numFloats; j++ {
			bits := binary.LittleEndian.Uint32(data[j*4 : (j+1)*4])
			weights[i][j] = math.Float32frombits(bits)
		}
	}
	return weights, nil
}

func loadMNISTImages(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 16)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numImages := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	pixelsPerImage := rows * cols
	images := make([][]float64, numImages)
	for i := 0; i < numImages; i++ {
		imgData := make([]byte, pixelsPerImage)
		if _, err := f.Read(imgData); err != nil {
			return nil, err
		}
		images[i] = make([]float64, pixelsPerImage)
		for j := 0; j < pixelsPerImage; j++ {
			val := float64(imgData[j]) / 255.0
			images[i][j] = (val - 0.1307) / 0.3081
		}
	}
	return images, nil
}

func loadMNISTLabels(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 8)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numLabels := int(binary.BigEndian.Uint32(header[4:8]))
	labels := make([]int, numLabels)
	labelData := make([]byte, numLabels)
	if _, err := f.Read(labelData); err != nil {
		return nil, err
	}
	for i := 0; i < numLabels; i++ {
		labels[i] = int(labelData[i])
	}
	return labels, nil
}

func argmax(vals []float32, n int) int {
	maxVal := float32(-math.MaxFloat32)
	maxIdx := -1
	for j := 0; j < n && j < len(vals); j++ {
		if vals[j] > maxVal {
			maxVal = vals[j]
			maxIdx = j
		}
	}
	return maxIdx
}

func TestMNISTSplit(t *testing.T) {
	fmt.Println("=== Lattigo MNIST Benchmark (split-preprocessing) ===")

	weights, err := loadWeights(modelPath)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}
	images, err := loadMNISTImages(imagesPath)
	if err != nil {
		t.Fatalf("Failed to load images: %v", err)
	}
	labels, err := loadMNISTLabels(labelsPath)
	if err != nil {
		t.Fatalf("Failed to load labels: %v", err)
	}

	fmt.Print("[1/5] Configuring Lattigo context...")
	t0 := time.Now()
	evaluator, params, encoder, encryptor, decryptor := mnist__configure()
	fmt.Printf(" done (%d ms)\n", msSince(t0))

	// Benchmark sample.
	input := images[0]
	inputFloat32 := make([]float32, len(input))
	for j := 0; j < len(input); j++ {
		inputFloat32[j] = float32(input[j])
	}

	fmt.Print("[2/5] Encrypting input...")
	t0 = time.Now()
	ctInput := mnist__encrypt__arg4(evaluator, params, encoder, encryptor, inputFloat32)
	fmt.Printf(" done (%d ms)\n", msSince(t0))

	fmt.Print("[3/5] Preprocessing plaintext weights...")
	t0 = time.Now()
	p0, p1, p2, p3, p4, p5, p6, p7 := mnistsplit_utils.Mnist__preprocessing(
		params, encoder, weights[0], weights[1], weights[2], weights[3],
	)
	fmt.Printf(" done (%d ms)\n", msSince(t0))

	// Warmup absorbs one-time costs (Lattigo evaluator caches, GC warmup,
	// lazy init). Its result is reused as the sink for the timed loop so
	// the last iteration leaves a valid ciphertext for the decrypt phase.
	fmt.Print("[4/5] Running preprocessed MNIST core (warmup + 5 runs)...")
	resCt := mnist__preprocessed(evaluator, params, encoder, ctInput, p0, p1, p2, p3, p4, p5, p6, p7)

	const kReps = 5
	runsMs := make([]int64, 0, kReps)
	for i := 0; i < kReps; i++ {
		s := time.Now()
		resCt = mnist__preprocessed(evaluator, params, encoder, ctInput, p0, p1, p2, p3, p4, p5, p6, p7)
		runsMs = append(runsMs, msSince(s))
	}
	sort.Slice(runsMs, func(i, j int) bool { return runsMs[i] < runsMs[j] })
	fmt.Printf(" done (min=%d ms, median=%d ms, max=%d ms)\n",
		runsMs[0], runsMs[kReps/2], runsMs[kReps-1])

	fmt.Print("[5/5] Decrypting result...")
	t0 = time.Now()
	resValues := mnist__decrypt__result0(evaluator, params, encoder, decryptor, resCt)
	fmt.Printf(" done (%d ms)\n", msSince(t0))

	predicted := argmax(resValues, 10)
	fmt.Printf("\nSample 0: predicted %d, actual %d\n", predicted, labels[0])

	// Quick accuracy check on a few more samples (untimed).
	total := 3
	correct := 0
	if predicted == labels[0] {
		correct++
	}
	for i := 1; i < total; i++ {
		input := images[i]
		inputFloat32 := make([]float32, len(input))
		for j := 0; j < len(input); j++ {
			inputFloat32[j] = float32(input[j])
		}
		ctIn := mnist__encrypt__arg4(evaluator, params, encoder, encryptor, inputFloat32)
		rc := mnist__preprocessed(evaluator, params, encoder, ctIn, p0, p1, p2, p3, p4, p5, p6, p7)
		rv := mnist__decrypt__result0(evaluator, params, encoder, decryptor, rc)
		pred := argmax(rv, 10)
		if pred == labels[i] {
			correct++
		}
		t.Logf("Sample %d: predicted %d, actual %d", i, pred, labels[i])
	}
	accuracy := float64(correct) / float64(total)
	t.Logf("Accuracy: %.2f (%d/%d correct)", accuracy, correct, total)
	if accuracy < 0.6 {
		t.Errorf("Accuracy too low: %.2f (%d/%d correct)", accuracy, correct, total)
	}

	fmt.Println("\n=== Lattigo MNIST Complete ===")
}
