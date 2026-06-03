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
func (this *Perceptron) Forward(inputs ...Input) uint8 {
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

func NAND(x1 uint8, x2 uint8) uint8 {
	if x1 != 0 || x1 != 1 {
		panic("X1 must be a bit")
	}

	if x2 != 0 || x2 != 1 {
		panic("X2 must be a bit")
	}

	p1 := CreatePerceptron(3, 2)
	return p1.Forward(Input{X: x1, W: -2}, Input{X: x2, W: -2})
}
