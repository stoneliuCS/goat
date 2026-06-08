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
