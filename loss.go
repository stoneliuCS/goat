package goat

import "math"

type Cost[N Number] interface {
	Cost(prediction, target *Vector[N]) N
	Gradient(prediction, target *Vector[N]) *Vector[N]
}

type MSE[N Number] struct{}

func (this *MSE[N]) Cost(prediction, target *Vector[N]) N {
	if !prediction.IsColumnVector() || !target.IsColumnVector() {
		panic("Both vectors must be column vectors")
	}
	if prediction.rows != target.rows {
		panic("Both vectors must have the same shape")
	}
	n := prediction.rows
	var sum float64 = 0

	for i := range n {
		gt := prediction.Get(i, 0)
		pred := target.Get(i, 0)
		sum += math.Pow(float64(gt-pred), 2)
	}
	return N(sum / float64(n))
}

func (this *MSE[N]) Gradient(prediction, target *Vector[N]) *Vector[N] {
	return prediction.Subtract(target)
}
