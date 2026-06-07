package goat

import (
	"fmt"
	"iter"
	"strings"
)

type Number interface {
	float64 | float32
}

type Vector[N Number] = Matrix[N]

type Matrix[N Number] struct {
	rows uint
	cols uint
	data [][]N
}

func (this *Matrix[T]) GetDimensions() (uint, uint) {
	if len(this.data) != int(this.rows) || len(this.data[0]) != int(this.cols) {
		panic("Invariant violated")
	}
	return this.rows, this.cols
}

func (this *Matrix[N]) IsColumnVector() bool {
	return this.cols == 1
}

func CreateMatrix[T Number](matrix [][]T) *Matrix[T] {
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

func (this *Matrix[T]) GetValues() iter.Seq[T] {
	return func(yield func(T) bool) {
		for i := range this.rows {
			for j := range this.cols {
				if !yield(this.Get(i, j)) {
					return
				}
			}
		}
	}
}

func GenerateRandomMatrix[N Number](rows uint, cols uint, randomSupplier func() N) *Matrix[N] {
	mat := make([][]N, rows)
	for i := range rows {
		mat[i] = make([]N, cols)
	}

	for i := range rows {
		for j := range cols {
			mat[i][j] = randomSupplier()
		}
	}
	return &Matrix[N]{
		rows: rows,
		cols: cols,
		data: mat,
	}
}

func Dot[T Number](u []T, v []T) T {
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

// Transposes this matrix into another
func (this *Matrix[T]) Transpose() *Matrix[T] {
	transposed := make([][]T, this.cols)
	for colIdx := range this.cols {
		transposed[colIdx] = this.GetColumn(colIdx)
	}
	return &Matrix[T]{
		rows: this.cols,
		cols: this.rows,
		data: transposed,
	}
}

func (this *Matrix[T]) Add(other *Matrix[T]) *Matrix[T] {
	if this.rows != other.rows || this.cols != other.cols {
		panic("Matricies must have the same dimensions to add together")
	}
	mat := make([][]T, this.rows)
	for i := range mat {
		mat[i] = make([]T, this.cols)
	}

	for i := range this.rows {
		for j := range this.cols {
			mat[i][j] = this.Get(i, j) + other.Get(i, j)
		}
	}
	return &Matrix[T]{
		rows: this.rows,
		cols: this.cols,
		data: mat,
	}

}

// Creates a constant matrix of the given value
func Values[T Number](rows uint, cols uint, value T) *Matrix[T] {

	matrix := make([][]T, rows)

	for i := range matrix {
		matrix[i] = make([]T, cols)
	}

	for i := range rows {
		for j := range cols {
			matrix[i][j] = value
		}
	}
	return &Matrix[T]{
		rows: rows,
		cols: cols,
		data: matrix,
	}
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

	for i := range thisRows {
		for j := range otherCols {
			row := this.GetRow(uint(i))
			col := other.GetColumn(uint(j))
			matrix[i][j] = Dot(row, col)
		}
	}
	return &Matrix[T]{
		rows: thisRows,
		cols: otherCols,
		data: matrix,
	}
}

func (this *Matrix[T]) String() string {
	if this.rows == 0 || this.cols == 0 {
		return "[]"
	}

	// Format every cell up front and track the widest entry per column.
	cells := make([][]string, this.rows)
	widths := make([]int, this.cols)
	for r := range this.data {
		cells[r] = make([]string, this.cols)
		for c, val := range this.data[r] {
			s := fmt.Sprintf("%v", val)
			cells[r][c] = s
			if len(s) > widths[c] {
				widths[c] = len(s)
			}
		}
	}

	bracket := func(r int) (left, right string) {
		n := len(cells)
		switch {
		case n == 1:
			return "[", "]"
		case r == 0:
			return "⎡", "⎤"
		case r == n-1:
			return "⎣", "⎦"
		default:
			return "⎢", "⎥"
		}
	}

	var b strings.Builder
	for r, row := range cells {
		left, right := bracket(r)
		b.WriteString(left)
		b.WriteByte(' ')
		for c, s := range row {
			if c > 0 {
				b.WriteString("  ")
			}
			fmt.Fprintf(&b, "%*s", widths[c], s)
		}
		b.WriteByte(' ')
		b.WriteString(right)
		if r < len(cells)-1 {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func (this *Matrix[T]) Print() {
	fmt.Println(this.String())
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
