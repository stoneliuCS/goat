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
	mat := matrix.CreateMatrix([][]int64{
		[]int64{2, 2},
		[]int64{2, 2},
	})
	n, m := mat.GetDimensions()
	assert.NotNil(t, mat)
	assert.Equal(t, uint(2), n)
	assert.Equal(t, uint(2), m)
}

func TestMultiply(t *testing.T) {
	mat1 := matrix.CreateMatrix([][]int64{
		[]int64{2, 2, 3},
		[]int64{2, 2, 3},
	})

	mat2 := matrix.CreateMatrix([][]int64{
		[]int64{2, 2},
		[]int64{2, 2},
		[]int64{2, 3},
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
	assert.Equal(t, int64(14), res.Get(0, 0))
	assert.Equal(t, int64(17), res.Get(0, 1))
	assert.Equal(t, int64(14), res.Get(1, 0))
	assert.Equal(t, int64(17), res.Get(1, 1))
}

func TestMultiplyProp(t *testing.T) {
	for range PROPERTY_RUNS {
		randRow := uint(rand.IntN(MAX_DIMS))
		randCol := uint(rand.IntN(MAX_DIMS))
		randomSupplier := func() int {
			if rand.Float32() <= 0.5 {
				return rand.IntN(MAX_VALUE)
			} else {
				return -1 * rand.IntN(MAX_VALUE)
			}
		}
		mat1 := matrix.GenerateRandomMatrix[int](randRow, randCol, randomSupplier)
		mat2 := matrix.GenerateRandomMatrix[int](randCol, randRow, randomSupplier)
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
