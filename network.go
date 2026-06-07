package goat

import (
	"math"
	"math/rand/v2"
)

type Network[N Number] struct {
	sizes                       []uint       // Represents the number of nodes in each layer.
	layers                      uint         // Layers not including the input layer.
	biases                      []*Vector[N] // Biases for each layer represented as column vectors
	weights                     []*Matrix[N]
	activations                 []*Vector[N] // a_l
	preActivations              []*Vector[N] // z_l
	activationFunctionsPerLayer []Activation[N]
}

// Creates a network initialized by the number of layers len(sizes), and the number of neurons per layer (sizes[i]).
// The input must be a column vector.
func CreateNetwork[N Number](sizes []uint, activations []Activation[N]) *Network[N] {
	if len(activations) != len(sizes)-1 {
		panic("Each layer must have an activation associated with it")
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
		sizes:                       sizes,
		layers:                      layers,
		biases:                      biases,
		weights:                     weights,
		activations:                 make([]*Vector[N], len(sizes)-1),
		preActivations:              make([]*Vector[N], len(sizes)-1),
		activationFunctionsPerLayer: activations,
	}
}

// Performs a forward pass on all weights and biases on the current network with the given activation function.
func (this *Network[T]) Forward(input *Vector[T]) *Vector[T] {
	var currentFeatures = input
	for layerIdx := range this.layers {
		biasVector := this.biases[layerIdx]
		weights := this.weights[layerIdx]
		z_l := weights.Multiply(currentFeatures).Add(biasVector)
		this.preActivations[layerIdx] = z_l
		activation := this.activationFunctionsPerLayer[layerIdx]
		a_l := activation.Apply(z_l)
		this.activations[layerIdx] = a_l
		currentFeatures = a_l
	}
	return currentFeatures
}

func (this *Network[N]) Backpropogation(input *Vector[N],
	prediction *Vector[N],
	target *Vector[N],
	loss Cost[N]) ([]*Vector[N], []*Matrix[N]) {
	// First compute the error of the prediction
	layer := int(this.layers - 1)
	derivative_of_cost_with_respect_to_L := loss.Gradient(prediction, target)
	derivative_of_activation := this.activationFunctionsPerLayer[layer].Gradient(this.preActivations[layer])
	delta := derivative_of_cost_with_respect_to_L.HadamardMultiply(derivative_of_activation)
	gradB := make([]*Vector[N], this.layers) // We store the gradients at each layer
	gradW := make([]*Matrix[N], this.layers)

	for layer >= 0 {
		gradB[layer] = delta
		var inActivations *Vector[N]
		if layer-1 < 0 {
			inActivations = input
		} else {
			inActivations = this.activations[layer-1]
		}
		weightGrad := delta.Multiply(inActivations.Transpose())
		gradW[layer] = weightGrad
		if layer > 0 {
			wTransposeDelta := this.weights[layer].Transpose().Multiply(delta)
			activationGradient := this.activationFunctionsPerLayer[layer-1].Gradient(this.preActivations[layer-1])
			delta = wTransposeDelta.HadamardMultiply(activationGradient)
		}
		layer -= 1
	}
	return gradB, gradW
}

func (this *Network[N]) Update(eta N, biases []*Vector[N], weights []*Matrix[N]) {
	for l := range this.layers {
		this.weights[l] = this.weights[l].Subtract(weights[l].Scale(eta))
		this.biases[l] = this.biases[l].Subtract(biases[l].Scale(eta))
	}
}

func (this *Network[N]) BatchTrain(inputs []*Vector[N], targets []*Vector[N], loss Cost[N], eta N) {
	if len(inputs) != len(targets) {
		panic("The length of inputs and targets must match.")
	}
	biasSums := make([]*Vector[N], this.layers)
	weightSums := make([]*Vector[N], this.layers)

	for i, input := range inputs {
		output := this.Forward(input)
		target := targets[i]
		gradB, gradW := this.Backpropogation(input, output, target, loss)
		for l := range this.layers {
			if biasSums[l] == nil || weightSums[l] == nil {
				biasSums[l], weightSums[l] = gradB[l], gradW[l]
			} else {
				biasSums[l] = biasSums[l].Add(gradB[l])
				weightSums[l] = weightSums[l].Add(gradW[l])
			}
		}
	}
	scale := 1 / N(len(inputs))
	for l := range this.layers {
		biasSums[l] = biasSums[l].Scale(scale)
		weightSums[l] = weightSums[l].Scale(scale)
	}
	this.Update(eta, biasSums, weightSums)
}
