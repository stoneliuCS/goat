package goat

import (
	"math"
	"math/rand/v2"
)

type Network[N Number] struct {
	sizes   []uint // Represents the number of nodes in each layer.
	layers  uint
	biases  []*Matrix[N]
	weights []*Matrix[N]
	input   *Matrix[N]
}

type Seed struct {
	val uint64
}

func (s Seed) Uint64() uint64 {
	return s.val
}

func Sigmoid[N Number](z *Matrix[N]) *Matrix[N] {
	if z.rows != 1 {
		panic("Input must be a vector")
	}
	res := []N{}
	for num := range z.GetValues() {
		res = append(res, N(1/(1+math.Exp(float64(-1*num)))))
	}
	return &Matrix[N]{
		rows: z.rows,
		cols: z.cols,
		data: [][]N{res},
	}
}

// Creates a network initialized by the number of layers len(sizes), and the number of neurons per layer (sizes[i]).
// The input is represented as a row vector.
func CreateNetwork[N Number](sizes []uint, input *Matrix[N]) *Network[N] {
	if input.cols != uint(sizes[0]) {
		panic("Input of the network must match the dimensions of the sizes")
	}
	biases := make([]*Matrix[N], len(sizes))
	weights := make([]*Matrix[N], len(sizes))
	var layers uint = uint(len(sizes))
	randomSupplier := func() N {
		return N(rand.Float64())
	}
	_, inputLength := input.GetDimensions()

	for idx, layer_size := range sizes {
		biases[idx] = GenerateRandomMatrix(1, layer_size, randomSupplier)
		weights[idx] = GenerateRandomMatrix(inputLength, layer_size, randomSupplier)
	}

	return &Network[N]{
		sizes:   sizes,
		layers:  layers,
		biases:  biases,
		weights: weights,
		input:   input,
	}
}

// Performs a forward pass on all weights and biases on the current network with the given activation function.
func (this *Network[T]) Forward(activation func(vector *Matrix[T]) *Matrix[T]) *Matrix[T] {
	var currentFeatures = this.input
	for layerIdx := range this.layers {
		biasVector := this.biases[layerIdx]
		weightsVector := this.weights[layerIdx].Transpose()
		sumVector := Values(biasVector.rows, biasVector.cols, currentFeatures.Multiply(weightsVector).Get(0, 0)) // This is the dot product
		currentFeatures = activation(sumVector)
	}
	return currentFeatures
}
