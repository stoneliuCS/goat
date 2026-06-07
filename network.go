package goat

import (
	"math"
	"math/rand/v2"
)

type Network[N Number] struct {
	sizes          []uint       // Represents the number of nodes in each layer.
	layers         uint         // Layers not including the input layer.
	biases         []*Vector[N] // Biases for each layer represented as column vectors
	weights        []*Matrix[N]
	input          *Vector[N]
	output         *Vector[N]
	activations    []*Vector[N] // a_l
	preActivations []*Vector[N] // z_l
}

type Seed struct {
	val uint64
}

func (s Seed) Uint64() uint64 {
	return s.val
}

// Creates a network initialized by the number of layers len(sizes), and the number of neurons per layer (sizes[i]).
// The input must be a column vector.
func CreateNetwork[N Number](sizes []uint, input *Vector[N]) *Network[N] {
	if !input.IsColumnVector() {
		panic("Input must be a column vector")
	}
	var layers uint = uint(len(sizes)) - 1
	biases := make([]*Matrix[N], layers)
	weights := make([]*Matrix[N], layers)
	zeroSupplier := func() N { return 0 }

	for i, layer_size := range sizes[1:] {
		idx := i + 1
		fanIn := sizes[idx-1]
		scale := math.Sqrt(1 / float64(fanIn))
		weightSupplier := func() N {
			return N(rand.NormFloat64() * scale)
		}
		biases[i] = GenerateRandomMatrix(1, layer_size, zeroSupplier).Transpose()
		weights[i] = GenerateRandomMatrix(layer_size, fanIn, weightSupplier)
	}

	return &Network[N]{
		sizes:          sizes,
		layers:         layers,
		biases:         biases,
		weights:        weights,
		input:          input,
		output:         CreateMatrix([][]N{make([]N, sizes[len(sizes)-1])}),
		activations:    make([]*Vector[N], sizes[len(sizes)-1]),
		preActivations: make([]*Vector[N], sizes[len(sizes)-1]),
	}
}

func (this *Network[T]) GetOutput() *Vector[T] {
	return this.output
}

// Performs a forward pass on all weights and biases on the current network with the given activation function.
func (this *Network[T]) Forward(activation func(vector *Vector[T]) *Vector[T]) {
	var currentFeatures = this.input
	for layerIdx := range this.layers {
		biasVector := this.biases[layerIdx]
		weights := this.weights[layerIdx]
		z_l := weights.Multiply(currentFeatures).Add(biasVector)
		this.preActivations[layerIdx] = z_l
		a_l := activation(z_l)
		this.activations[layerIdx] = a_l
		currentFeatures = a_l
	}
	this.output = currentFeatures
}

func (this *Network[N]) Backpropogation(output *Vector[N], loss func(groundTruth *Vector[N], output *Vector[N])) {

}
