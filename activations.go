package goat

import "math"

type Activation[N Number] interface {
	Apply(z *Vector[N]) *Vector[N]
	Gradient(z *Vector[N]) *Vector[N]
}

// Sigmoid Activation
type Sigmoid[N Number] struct{}

func (this Sigmoid[N]) Apply(z *Vector[N]) *Vector[N] {
	if z.cols != 1 {
		panic("Input must be a vector")
	}
	data := make([][]N, 0, z.rows)
	for num := range z.GetValues() {
		data = append(data, []N{N(1 / (1 + math.Exp(float64(-num))))})
	}
	return &Matrix[N]{
		rows: z.rows,
		cols: z.cols,
		data: data,
	}
}

func (this Sigmoid[N]) Gradient(z *Vector[N]) *Vector[N] {
	// sigma'(x) = sigma(x)(1 - sigma(x))
	n, m := z.GetDimensions()
	s := this.Apply(z)
	return s.HadamardMultiply(Values[N](n, m, 1).Subtract(s))
}

func SigmoidActivation[N Number]() Sigmoid[N] {
	return Sigmoid[N]{}
}
