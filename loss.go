package goat

import "math"

type Cost[N Number] interface {
	Cost(prediction, target *Vector[N]) N
	Gradient(prediction, target *Vector[N]) *Vector[N]
}

type BinaryCrossEntropy[N Number] struct{}

func (this *BinaryCrossEntropy[N]) Cost(prediction, target *Vector[N]) N {
	if !prediction.IsColumnVector() || !target.IsColumnVector() {
		panic("Both vectors must be column vectors")
	}
	if prediction.rows != target.rows {
		panic("Both vectors must have the same shape")
	}
	n := prediction.rows
	bce := 0.0
	for i := range n {
		y_i := float64(target.Get(i, 0))
		p_i := float64(prediction.Get(i, 0))
		log_pi := math.Log(float64(p_i))
		log_1_minus_pi := math.Log(1 - float64(p_i))
		bce += y_i*log_pi + (1-y_i)*log_1_minus_pi
	}
	return N(-1 * (1 / float64(n)) * bce)
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
