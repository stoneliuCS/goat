package goat_test

import (
	matrix "goat"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCreate(t *testing.T) {
	mat := matrix.Create([][]int64{
		[]int64{2, 2},
		[]int64{2, 2},
	})
	n, m := mat.GetDimensions()
	assert.NotNil(t, mat)
	assert.Equal(t, uint(2), n)
	assert.Equal(t, uint(2), m)
}

func TestMultiply(t *testing.T) {
	mat1 := matrix.Create([][]int64{
		[]int64{2, 2, 3},
		[]int64{2, 2, 3},
	})

	mat2 := matrix.Create([][]int64{
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
