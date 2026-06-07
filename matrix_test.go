package goat_test

import (
	matrix "goat"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
)

const PROPERTY_RUNS = 1000
const MAX_DIMS = 100
const MAX_VALUE = 1000

func TestCreate(t *testing.T) {
	mat := matrix.CreateMatrix([][]float64{
		[]float64{2, 2},
		[]float64{2, 2},
	})
	n, m := mat.GetDimensions()
	assert.NotNil(t, mat)
	assert.Equal(t, uint(2), n)
	assert.Equal(t, uint(2), m)
}

func TestMultiply(t *testing.T) {
	mat1 := matrix.CreateMatrix([][]float64{
		[]float64{2, 2, 3},
		[]float64{2, 2, 3},
	})

	mat2 := matrix.CreateMatrix([][]float64{
		[]float64{2, 2},
		[]float64{2, 2},
		[]float64{2, 3},
	})

	res := mat1.Multiply(mat2)
	actual_n, actual_m := res.GetDimensions()

	// (2 x 3) x (3 x 2) = 2 x 2
	assert.Equal(t, uint(2), actual_n)
	assert.Equal(t, uint(2), actual_m)

	/*
	 | 14 17 |
	 | 14 17 |
	*/
	assert.Equal(t, float64(14), res.Get(0, 0))
	assert.Equal(t, float64(17), res.Get(0, 1))
	assert.Equal(t, float64(14), res.Get(1, 0))
	assert.Equal(t, float64(17), res.Get(1, 1))
}

func TestMultiplyProp(t *testing.T) {
	for range PROPERTY_RUNS {
		randRow := uint(rand.IntN(MAX_DIMS))
		randCol := uint(rand.IntN(MAX_DIMS))
		randomSupplier := func() float64 {
			if rand.Float32() <= 0.5 {
				return rand.Float64() * MAX_VALUE
			} else {
				return -1 * rand.Float64() * MAX_VALUE
			}
		}
		mat1 := matrix.GenerateRandomMatrix[float64](randRow, randCol, randomSupplier)
		mat2 := matrix.GenerateRandomMatrix[float64](randCol, randRow, randomSupplier)
		res := mat1.Multiply(mat2)
		// Output is a square matrix
		for i := range randRow {
			for j := range randRow {
				expected := matrix.Dot(mat1.GetRow(i), mat2.GetColumn(j))
				actual := res.Get(i, j)
				assert.Equal(t, expected, actual)
			}
		}
	}
}

func TestVectorTransposeMultiplication(t *testing.T) {
	// (1 x 2) x (2 x 1) => 1 x 1 vector
	u := matrix.CreateMatrix([][]float64{[]float64{1, 2}})
	v := matrix.CreateMatrix([][]float64{[]float64{2}, []float64{1}})
	res := u.Multiply(v)
	rows, cols := res.GetDimensions()
	assert.Equal(t, uint(1), rows)
	assert.Equal(t, uint(1), cols)
}
