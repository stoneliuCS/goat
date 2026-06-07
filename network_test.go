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
	network := nn.CreateNetwork(sizes, sampleInput)
	// One forward pass through the network, all values should be squished between 0 and 1
	network.Forward(nn.Sigmoid)
	resultVector := network.GetOutput()

	for val := range resultVector.GetValues() {
		assert.GreaterOrEqual(t, 1.0, val)
		assert.GreaterOrEqual(t, val, 0.0)
	}

}
