package goat 

type Number interface {
	float64 | int64 | uint64
}

type Matrix[N Number] struct {
	rows uint
	cols uint
	data [][]N
}

func (this *Matrix[T]) GetDimensions() (uint, uint) {
	return this.rows, this.cols
}

func Create[T Number](matrix [][]T) *Matrix[T] {
	if len(matrix) == 0 {
		panic("Cannot create an empty matrix.")
	}
	rows := len(matrix)
	cols := len(matrix[0])
	return &Matrix[T]{
		rows: uint(rows),
		cols: uint(cols),
		data: matrix,
	}
}

func (this *Matrix[T]) GetColumn(idx uint) []T {
	cols := []T{}

	for row := range this.rows {
		cols = append(cols, this.data[row][idx])
	}
	return cols
}

func (this *Matrix[T]) GetRow(idx uint) []T {
	return this.data[idx]
}

func dot[T Number](u []T, v []T) T {
	if len(u) != len(v) {
		panic("Shape of U and V are not the same.")
	}
	var sum T = 0
	n := len(u)
	for i := range n {
		sum += u[i] * v[i]
	}
	return sum
}

func (this *Matrix[T]) Multiply(other *Matrix[T]) *Matrix[T] {
	thisCols := this.cols
	thisRows := this.rows

	otherRows := other.rows
	otherCols := other.cols
	// (m x n) x (n x m) => m x m
	// In other words the cols of this matrix must match the rows of the other matrix
	if thisCols != otherRows {
		panic("Shape mismatch")
	}
	matrix := make([][]T, thisRows)

	for i := range matrix {
		matrix[i] = make([]T, otherCols)
	}

	for i := range len(matrix) {
		for j := range len(matrix[0]) {
			for range len(matrix[0]) {
				col := this.GetColumn(uint(i))
				row := other.GetRow(uint(j))
				matrix[i][j] = dot(col, row)
			}
		}
	}
	return &Matrix[T]{
		rows: thisRows,
		cols: otherCols,
		data: matrix,
	}
}

func (this *Matrix[T]) Get(row uint, col uint) T {
	if row > this.rows-1 {
		panic("Row index out of bounds")
	}

	if col > this.cols-1 {
		panic("Col index out of bounds")
	}
	return this.data[row][col]
}
