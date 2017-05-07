package main

import (
	// "fmt"
	"math/rand"
)

type Matrix [][]float64

func (m Matrix) rows() int {
	return len(m)
}

func (m Matrix) cols() int {
	return len(m[0])
}

func (m Matrix) add(a Matrix) Matrix {
	if m.rows() != a.rows() || m.cols() != a.cols() {
		panic("Can't add 2 different size matrices.")
	}
	for i := 0; i < m.rows(); i++ {
		for j := 0; j < m.cols(); j++ {
			m[i][j] += a[i][j]
		}
	}
	return m
}

func (m Matrix) mult(x float64) Matrix {
	for i := 0; i < m.rows(); i++ {
		for j := 0; j < m.cols(); j++ {
			m[i][j] *= x
		}
	}
	return m
}

func (m Matrix) ToArray() []float64 {
	a := make([]float64, m.rows()*m.cols())
	counter := 0
	for i := 0; i < m.rows(); i++ {
		for j := 0; j < m.cols(); j++ {
			a[counter] = m[i][j]
			counter += 1
		}
	}
	return a
}

func CreateMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

func transpose(m Matrix) Matrix {
	matrix := CreateMatrix(m.cols(), m.rows())
	for i := 0; i < m.rows(); i++ {
		for j := 0; j < m.cols(); j++ {
			matrix[j][i] = m[i][j]
		}
	}
	return matrix
}

func MakeMatrix(array []float64, rows int, cols int) Matrix {
	if len(array) != rows*cols {
		panic("Array length doesn't match rows and cols specified.")
	}

	matrix := CreateMatrix(rows, cols)
	counter := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix[i][j] = array[counter]
			counter += 1
		}
	}
	return matrix
}

func RandomWeightMatrix(rows, cols int) Matrix {
	matrix := CreateMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix[i][j] = rand.Float64() - 0.5
		}
	}
	return matrix
}

func Dot(a, b Matrix) Matrix {
	if a.cols() != b.rows() {
		panic("Rows != Cols")
	}

	matrix := CreateMatrix(a.rows(), b.cols())

	for i := 0; i < a.rows(); i++ {
		for j := 0; j < b.cols(); j++ {
			sum := 0.0
			for k := 0; k < b.rows(); k++ {
				sum = sum + (a[i][k] * b[k][j])
			}
			matrix[i][j] = sum
		}
	}
	return matrix
}
