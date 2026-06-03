package goat

import (
	"fmt"
	"strings"
)

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

	for i := range thisRows {
		for j := range otherCols {
			row := this.GetRow(uint(i))
			col := other.GetColumn(uint(j))
			matrix[i][j] = dot(row, col)
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
