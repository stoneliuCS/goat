package goat

import "math"

func Sigmoid[N Number](z *Vector[N]) *Vector[N] {
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
