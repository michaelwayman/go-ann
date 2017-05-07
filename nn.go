package main

import (
	// "fmt"
	"math"
)

type NeuralNetwork struct {
	numberInputs      int
	numberOutputs     int
	numberHiddenNodes int
	learningRate      float64
	inputWeights      Matrix
	outputWeights     Matrix
}

func NewNeuralNetwork(i int, o int, h int, l float64) NeuralNetwork {
	return NeuralNetwork{
		numberInputs:      i,
		numberOutputs:     o,
		numberHiddenNodes: h,
		learningRate:      l,
		inputWeights:      RandomWeightMatrix(h, i),
		outputWeights:     RandomWeightMatrix(o, h),
	}
}

func activation(x float64) float64 {
	// Sigmoid function
	return 1.0 / (1.0 + math.Exp(-x))
}

func ActivateMatrix(m Matrix) Matrix {
	matrix := CreateMatrix(m.rows(), m.cols())
	for i := 0; i < m.rows(); i++ {
		for j := 0; j < m.cols(); j++ {
			matrix[i][j] = activation(m[i][j])
		}
	}
	return matrix
}

func ErrorMatrix(targets Matrix, finalOut Matrix) Matrix {
	matrix := CreateMatrix(targets.rows(), targets.cols())
	for i := 0; i < targets.rows(); i++ {
		for j := 0; j < targets.cols(); j++ {
			matrix[i][j] = targets[i][j] - finalOut[i][j]
		}
	}
	return matrix
}

func AdjustmentMatrix(a, b Matrix) Matrix {
	m := CreateMatrix(a.rows(), a.cols())
	for i := 0; i < a.rows(); i++ {
		for j := 0; j < a.cols(); j++ {
			m[i][j] = a[i][j] * b[i][j] * (1.0 - b[i][j])
		}
	}
	return m
}

func (nn NeuralNetwork) train(inputs []float64, targets []float64) {
	// Create matrices out of our training data
	inputMatrix := MakeMatrix(inputs, len(inputs), 1)
	targetMatrix := MakeMatrix(targets, len(targets), 1)

	// Calculate signals going into the hidden layer
	hiddenInputs := Dot(nn.inputWeights, inputMatrix)
	// Calculate signals coming out of the hidden layer
	hiddenOutputs := ActivateMatrix(hiddenInputs)

	// Calculate signals going into the output layer
	finalInputs := Dot(nn.outputWeights, hiddenOutputs)
	// Calculate signals coming out of the output layer
	finalOutputs := ActivateMatrix(finalInputs)

	// Output layer error is simple (target - actual)
	outputErrors := ErrorMatrix(targetMatrix, finalOutputs)

	// hidden layer error is output error split by weights
	hiddenError := Dot(transpose(nn.outputWeights), outputErrors)

	// gradient descent
	too := Dot(AdjustmentMatrix(outputErrors, finalOutputs), transpose(hiddenOutputs)).mult(nn.learningRate)
	poo := Dot(AdjustmentMatrix(hiddenError, hiddenOutputs), transpose(inputMatrix)).mult(nn.learningRate)

	nn.outputWeights.add(too)
	nn.inputWeights.add(poo)
}

func (nn NeuralNetwork) query(inputs []float64) Matrix {
	// Create matrices out of our training data
	inputMatrix := MakeMatrix(inputs, len(inputs), 1)

	// Calculate signals going into the hidden layer
	hiddenInputs := Dot(nn.inputWeights, inputMatrix)
	// Calculate signals coming out of the hidden layer
	hiddenOutputs := ActivateMatrix(hiddenInputs)

	// Calculate signals going into the output layer
	finalInputs := Dot(nn.outputWeights, hiddenOutputs)
	// Calculate signals coming out of the output layer
	finalOutputs := ActivateMatrix(finalInputs)

	return finalOutputs
}
