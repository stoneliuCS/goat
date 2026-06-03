package goat

type Perceptron struct {
	threshold float64
	inputs    uint // Number of inputs the perceptron is allowed to take.
}

func CreatePerceptron(threshold float64, inputSize uint) *Perceptron {
	return &Perceptron{
		threshold: threshold,
		inputs:    inputSize,
	}

}

// Input is a single binary input together with its weight.
type Input struct {
	X uint8
	W float64
}

// Takes a number of binary inputs and their weights and computes the resulting binary output.
func (this *Perceptron) Forward(inputs []Input) uint8 {
	var sum float64 = 0

	for _, input := range inputs {
		sum += float64(input.X) * input.W
	}

	if sum+this.threshold <= 0 {
		return 0
	} else {
		return 1
	}
}
