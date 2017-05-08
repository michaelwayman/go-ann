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

/**
 *	Sigmoidal function
 */
func ActivationFunction(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func AdjustmentMatrix(a, b Matrix) Matrix {
	m := NewMatrix(a.rows(), a.cols())
	for i := 0; i < a.rows(); i++ {
		for j := 0; j < a.cols(); j++ {
			m[i][j] = a[i][j] * b[i][j] * (1.0 - b[i][j])
		}
	}
	return m
}

func (nn NeuralNetwork) train(inputs []float64, targets []float64) {
	// Create matrices out of our training data
	inputMatrix := toMatrix(inputs, len(inputs), 1)
	targetMatrix := toMatrix(targets, len(targets), 1)

	// Calculate signals going into the hidden layer
	hiddenInputs := Dot(nn.inputWeights, inputMatrix)
	// Calculate signals coming out of the hidden layer
	hiddenOutputs := Map(hiddenInputs, ActivationFunction)

	// Calculate signals going into the output layer
	finalInputs := Dot(nn.outputWeights, hiddenOutputs)
	// Calculate signals coming out of the output layer
	finalOutputs := Map(finalInputs, ActivationFunction)

	// Output layer error is simple (target - actual)
	outputErrors := Sub(targetMatrix, finalOutputs)

	// hidden layer error is output error split by weights
	hiddenError := Dot(Transpose(nn.outputWeights), outputErrors)

	// gradient descent
	too := Dot(AdjustmentMatrix(outputErrors, finalOutputs), Transpose(hiddenOutputs)).mult(nn.learningRate)
	poo := Dot(AdjustmentMatrix(hiddenError, hiddenOutputs), Transpose(inputMatrix)).mult(nn.learningRate)

	// Adjust our weights
	nn.outputWeights.add(too)
	nn.inputWeights.add(poo)
}

func (nn NeuralNetwork) query(inputs []float64) []float64 {
	// Create matrix out of our training data
	inputMatrix := toMatrix(inputs, len(inputs), 1)

	// Calculate signals going into the hidden layer
	hiddenInputs := Dot(nn.inputWeights, inputMatrix)
	// Calculate signals coming out of the hidden layer
	hiddenOutputs := Map(hiddenInputs, ActivationFunction)

	// Calculate signals going into the output layer
	finalInputs := Dot(nn.outputWeights, hiddenOutputs)
	// Calculate signals coming out of the output layer
	finalOutputs := Map(finalInputs, ActivationFunction)

	return finalOutputs.toArray()
}
