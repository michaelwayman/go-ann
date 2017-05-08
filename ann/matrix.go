package ann

import (
	"math/rand"
)

type Matrix [][]float64

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Cols() int {
	return len(m[0])
}

// Add each each element in `a` to `m`
func (m Matrix) Add(a Matrix) Matrix {
	if m.Rows() != a.Rows() || m.Cols() != a.Cols() {
		panic("Can't add 2 different size matrices.")
	}
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			m[i][j] += a[i][j]
		}
	}
	return m
}

// Multiply each element by some constant
func (m Matrix) Mult(x float64) Matrix {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			m[i][j] *= x
		}
	}
	return m
}

// Converts a Matrix to a 1d array
func (m Matrix) ToArray() []float64 {
	a := make([]float64, m.Rows()*m.Cols())
	counter := 0
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			a[counter] = m[i][j]
			counter += 1
		}
	}
	return a
}

// Creates a new Matrix
func NewMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

// Returns the transpose of `m`
func Transpose(m Matrix) Matrix {
	matrix := NewMatrix(m.Cols(), m.Rows())
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			matrix[j][i] = m[i][j]
		}
	}
	return matrix
}

// Converts a 1d array or slice to a Matrix
func ToMatrix(array []float64, rows int, cols int) Matrix {
	if len(array) != rows*cols {
		panic("Array length doesn't match rows and cols specified.")
	}

	matrix := NewMatrix(rows, cols)
	counter := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix[i][j] = array[counter]
			counter += 1
		}
	}
	return matrix
}

// Creates a new matrix with each element (-0.5 <= x <= 0.5)
func RandomWeightMatrix(rows, cols int) Matrix {
	matrix := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix[i][j] = rand.Float64() - 0.5
		}
	}
	return matrix
}

// Returns a new matrix, the dot product of `a` and `b`
func Dot(a, b Matrix) Matrix {
	if a.Cols() != b.Rows() {
		panic("Rows != Cols")
	}

	matrix := NewMatrix(a.Rows(), b.Cols())

	for i := 0; i < a.Rows(); i++ {
		for j := 0; j < b.Cols(); j++ {
			sum := 0.0
			for k := 0; k < b.Rows(); k++ {
				sum = sum + (a[i][k] * b[k][j])
			}
			matrix[i][j] = sum
		}
	}
	return matrix
}

// Subtracts each element in `b` from the corresponding element in `a` and returns a new matrix
func Sub(a, b Matrix) Matrix {
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		panic("Can't subtract 2 different size matrices.")
	}

	m := NewMatrix(a.Rows(), a.Cols())
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			m[i][j] = a[i][j] - b[i][j]
		}
	}
	return m
}

// Calls `fn` on every element in `m` and returns a new Matrix
func Map(m Matrix, fn func(x float64) float64) Matrix {
	matrix := NewMatrix(m.Rows(), m.Cols())
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			matrix[i][j] = fn(m[i][j])
		}
	}
	return matrix
}
