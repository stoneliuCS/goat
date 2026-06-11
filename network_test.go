package goat_test

import (
	nn "goat"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestForwardPass(t *testing.T) {
	// Initialize a 728 x 16 x 16 x 10 Neural Network
	sizes := []uint{728, 16, 16, 10}
	var sampleInput *nn.Vector[float64] = nn.GenerateRandomMatrix(728, 1, func() float64 {
		return rand.Float64() * 255 // grayscale values
	})
	activations := []nn.Activation[float64]{
		nn.SigmoidActivation[float64](),
		nn.SigmoidActivation[float64](),
		nn.SigmoidActivation[float64](),
	}
	network := nn.CreateNetwork(sizes, activations)
	// One forward pass through the network, all values should be squished between 0 and 1
	resultVector := network.Forward(sampleInput)

	for val := range resultVector.GetValues() {
		assert.GreaterOrEqual(t, 1.0, val)
		assert.GreaterOrEqual(t, val, 0.0)
	}
}

func TestXorNetwork(t *testing.T) {
	sizes := []uint{2, 1, 2, 1} // 4 layer neural network
	activations := []nn.Activation[float64]{
		nn.SigmoidActivation[float64](),
		nn.SigmoidActivation[float64](),
		nn.SigmoidActivation[float64](),
	}
	network := nn.CreateNetwork(sizes, activations)
	var input []*nn.Vector[float64] = []*nn.Vector[float64]{
		nn.CreateColumnVector([]float64{1, 1}),
		nn.CreateColumnVector([]float64{1, 0}),
		nn.CreateColumnVector([]float64{0, 1}),
		nn.CreateColumnVector([]float64{0, 0}),
	}
	var targets []*nn.Vector[float64] = []*nn.Vector[float64]{
		nn.CreateColumnVector([]float64{0}),
		nn.CreateColumnVector([]float64{1}),
		nn.CreateColumnVector([]float64{1}),
		nn.CreateColumnVector([]float64{0}),
	}
	network.BatchTrain(input, targets, &nn.MSE[float64]{}, 3)

	// Train the Network
	const epochs, eta = 20000, 5
	for e := 0; e < epochs; e++ {
		network.BatchTrain(input, targets, &nn.MSE[float64]{}, eta)
	}

	for _, i := range input {
		res := network.Forward(i) // The result should be an output of 1 x 1
		n, m := res.GetDimensions()
		assert.Equal(t, uint(1), n)
		assert.Equal(t, uint(1), m)
	}
}

func argmax(v *nn.Vector[float64]) int {
	rows, _ := v.GetDimensions()
	best, bestVal := 0, v.Get(0, 0)
	for i := uint(1); i < rows; i++ {
		if val := v.Get(i, 0); val > bestVal {
			best, bestVal = int(i), val
		}
	}
	return best
}

func TestMNISTDataset(t *testing.T) {
	train, err := nn.LoadMNIST[float64]("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
	if err != nil {
		t.Fatal(err)
	}
	test, err := nn.LoadMNIST[float64]("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		t.Fatal(err)
	}

	sizes := []uint{784, 30, 10} // 28*28 inputs -> hidden -> 10 digit classes
	activations := []nn.Activation[float64]{
		nn.SigmoidActivation[float64](),
		nn.SigmoidActivation[float64](),
	}
	network := nn.CreateNetwork(sizes, activations)

	const epochs, batchSize, eta = 5, 32, 3.0
	loss := &nn.MSE[float64]{}

	for e := 0; e < epochs; e++ {
		rand.Shuffle(len(train), func(i, j int) { train[i], train[j] = train[j], train[i] })
		for start := 0; start < len(train); start += batchSize {
			end := min(start+batchSize, len(train))
			batch := train[start:end]
			inputs := make([]*nn.Vector[float64], len(batch))
			targets := make([]*nn.Vector[float64], len(batch))
			for i, s := range batch {
				inputs[i], targets[i] = s.Image, s.Label
			}
			network.BatchTrain(inputs, targets, loss, eta)
		}

		correct := 0
		for _, s := range test {
			if argmax(network.Forward(s.Image)) == int(s.Digit) {
				correct++
			}
		}
		t.Logf("epoch %d: %d/%d = %.2f%%", e+1, correct, len(test), 100*float64(correct)/float64(len(test)))
	}
}
