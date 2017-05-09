package ann

import (
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
)

type NeuralNetwork struct {
	NumberInputs      int
	NumberOutputs     int
	NumberHiddenNodes int
	LearningRate      float64
	InputWeights      *mat64.Dense
	OutputWeights     *mat64.Dense
}

func New(numberInputs int, numberOutputs int, numberHiddenNodes int, learningRate float64) NeuralNetwork {
	initInputWeights := make([]float64, numberInputs*numberHiddenNodes)
	initOutputWeights := make([]float64, numberHiddenNodes*numberOutputs)

	for i := range initInputWeights {
		initInputWeights[i] = rand.Float64() - 0.5
	}
	for i := range initOutputWeights {
		initOutputWeights[i] = rand.Float64() - 0.5
	}

	return NeuralNetwork{
		NumberInputs:      numberInputs,
		NumberOutputs:     numberOutputs,
		NumberHiddenNodes: numberHiddenNodes,
		LearningRate:      learningRate,
		InputWeights:      mat64.NewDense(numberHiddenNodes, numberInputs, initInputWeights),
		OutputWeights:     mat64.NewDense(numberOutputs, numberHiddenNodes, initOutputWeights),
	}
}

// Sigmoidal function
func ActivationFunction(i, j int, v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func AdjustmentMatrix(a, b mat64.Matrix) mat64.Matrix {

	rows, cols := a.Dims()
	data := make([]float64, rows*cols)
	counter := 0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[counter] = a.At(i, j) * b.At(i, j) * (1.0 - b.At(i, j))
			counter++
		}
	}

	return mat64.NewDense(rows, cols, data)
}

func (nn NeuralNetwork) Train(inputs []float64, targets []float64) {
	// Create matrices out of our training data
	inputMatrix := mat64.NewDense(len(inputs), 1, inputs)
	targetMatrix := mat64.NewDense(len(targets), 1, targets)

	// Calculate signals going in/out of the hidden layer
	hiddenInputs := mat64.NewDense(nn.NumberHiddenNodes, 1, nil)
	hiddenInputs.Mul(nn.InputWeights, inputMatrix)
	hiddenOutputs := mat64.NewDense(nn.NumberHiddenNodes, 1, nil)
	hiddenOutputs.Apply(ActivationFunction, hiddenInputs)

	// Calculate signals going in/out of the output layer
	finalInputs := mat64.NewDense(nn.NumberOutputs, 1, nil)
	finalInputs.Mul(nn.OutputWeights, hiddenOutputs)
	finalOutputs := mat64.NewDense(nn.NumberOutputs, 1, nil)
	finalOutputs.Apply(ActivationFunction, finalInputs)

	// Output layer error is simple (target - actual)
	outputErrors := mat64.NewDense(nn.NumberOutputs, 1, nil)
	outputErrors.Sub(targetMatrix, finalOutputs)

	// hidden layer error is output error split by weights
	hiddenError := mat64.NewDense(nn.NumberHiddenNodes, 1, nil)
	hiddenError.Mul(nn.OutputWeights.T(), outputErrors)

	// gradient descent
	too := mat64.NewDense(nn.NumberOutputs, nn.NumberHiddenNodes, nil)
	poo := mat64.NewDense(nn.NumberHiddenNodes, nn.NumberInputs, nil)

	too.Mul(AdjustmentMatrix(outputErrors, finalOutputs), hiddenOutputs.T())
	too.Scale(nn.LearningRate, too)
	poo.Mul(AdjustmentMatrix(hiddenError, hiddenOutputs), inputMatrix.T())
	poo.Scale(nn.LearningRate, poo)

	// Adjust our weights
	nn.OutputWeights.Add(nn.OutputWeights, too)
	nn.InputWeights.Add(nn.InputWeights, poo)
}

func (nn NeuralNetwork) Query(inputs []float64) []float64 {
	// Create matrix out of our training data
	inputMatrix := mat64.NewDense(len(inputs), 1, inputs)

	// Calculate signals going in/out of the hidden layer
	hiddenInputs := mat64.NewDense(nn.NumberHiddenNodes, 1, nil)
	hiddenInputs.Mul(nn.InputWeights, inputMatrix)
	hiddenOutputs := mat64.NewDense(nn.NumberHiddenNodes, 1, nil)
	hiddenOutputs.Apply(ActivationFunction, hiddenInputs)

	// Calculate signals going in/out of the output layer
	finalInputs := mat64.NewDense(nn.NumberOutputs, 1, nil)
	finalInputs.Mul(nn.OutputWeights, hiddenOutputs)
	finalOutputs := mat64.NewDense(nn.NumberOutputs, 1, nil)
	finalOutputs.Apply(ActivationFunction, finalInputs)

	results := make([]float64, nn.NumberOutputs)
	return mat64.Col(results, 0, finalOutputs)
}
