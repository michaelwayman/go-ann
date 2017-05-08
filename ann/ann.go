package ann

import (
	// "fmt"
	"math"
)

type NeuralNetwork struct {
	NumberInputs      int
	NumberOutputs     int
	NumberHiddenNodes int
	LearningRate      float64
	InputWeights      Matrix
	OutputWeights     Matrix
}

// Sigmoidal function
func ActivationFunction(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func AdjustmentMatrix(a, b Matrix) Matrix {
	m := NewMatrix(a.Rows(), a.Cols())
	for i := 0; i < a.Rows(); i++ {
		for j := 0; j < a.Cols(); j++ {
			m[i][j] = a[i][j] * b[i][j] * (1.0 - b[i][j])
		}
	}
	return m
}

func (nn NeuralNetwork) Train(inputs []float64, targets []float64) {
	// Create matrices out of our training data
	inputMatrix := ToMatrix(inputs, len(inputs), 1)
	targetMatrix := ToMatrix(targets, len(targets), 1)

	// Calculate signals going in/out of the hidden layer
	hiddenInputs := Dot(nn.InputWeights, inputMatrix)
	hiddenOutputs := Map(hiddenInputs, ActivationFunction)

	// Calculate signals going in/out of the output layer
	finalInputs := Dot(nn.OutputWeights, hiddenOutputs)
	finalOutputs := Map(finalInputs, ActivationFunction)

	// Output layer error is simple (target - actual)
	outputErrors := Sub(targetMatrix, finalOutputs)

	// hidden layer error is output error split by weights
	hiddenError := Dot(Transpose(nn.OutputWeights), outputErrors)

	// gradient descent
	too := Dot(AdjustmentMatrix(outputErrors, finalOutputs), Transpose(hiddenOutputs)).Mult(nn.LearningRate)
	poo := Dot(AdjustmentMatrix(hiddenError, hiddenOutputs), Transpose(inputMatrix)).Mult(nn.LearningRate)

	// Adjust our weights
	nn.OutputWeights.Add(too)
	nn.InputWeights.Add(poo)
}

func (nn NeuralNetwork) Query(inputs []float64) []float64 {
	// Create matrix out of our training data
	inputMatrix := ToMatrix(inputs, len(inputs), 1)

	// Calculate signals going in/out of the hidden layer
	hiddenInputs := Dot(nn.InputWeights, inputMatrix)
	hiddenOutputs := Map(hiddenInputs, ActivationFunction)

	// Calculate signals going in/out of the output layer
	finalInputs := Dot(nn.OutputWeights, hiddenOutputs)
	finalOutputs := Map(finalInputs, ActivationFunction)

	return finalOutputs.ToArray()
}
