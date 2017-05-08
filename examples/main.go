package main

import (
	"fmt"
	"github.com/michaelwayman/go-ann/ann"
)

func TrainNeuralNetwork(nn ann.NeuralNetwork) {
	cleanedTrainingData := CleanedDataForNN("mnist_train.csv")
	for {
		inputs, targets, err := cleanedTrainingData()
		if err != nil {
			break
		}
		nn.Train(inputs, targets)
	}
}

func TestNeuralNetwork(nn ann.NeuralNetwork) float64 {

	// Scoreboard to keep track of how many we got correct
	correct, total := 0.0, 0.0
	scoreboard := func(outputs, targets []float64) float64 {
		maxIndex := 0
		maxVal := outputs[0]
		for i, v := range outputs {
			if v >= maxVal {
				maxIndex = i
				maxVal = v
			}
		}
		total += 1.0
		if targets[maxIndex] >= 0.9 {
			correct += 1.0
		}
		return correct / total
	}

	// Test the Neural Network
	var accuracy float64
	cleanedTestingData := CleanedDataForNN("mnist_test.csv")
	for {
		inputs, targets, err := cleanedTestingData()
		if err != nil {
			break
		}
		accuracy = scoreboard(nn.Query(inputs), targets)
	}
	return accuracy
}

var welcomeMsg = `
Artificial Neural Network to solve the MNIST data set.

 1. Initialize the ANN.
 2. Read the MNIST training data set and train the ANN.
 3. Read the MNIST testing data set and test the ANN.
 4. Print the accuracy of the ANN.
`

func main() {

	fmt.Println(welcomeMsg)

	nn := ann.NeuralNetwork{
		NumberInputs:      784,
		NumberOutputs:     10,
		NumberHiddenNodes: 200,
		LearningRate:      0.1,
		InputWeights:      ann.RandomWeightMatrix(200, 784),
		OutputWeights:     ann.RandomWeightMatrix(10, 200),
	}

	fmt.Println("Training the neural network.")
	TrainNeuralNetwork(nn)

	fmt.Println("Testing the neural network.")
	accuracy := TestNeuralNetwork(nn)

	fmt.Printf("Neural network is %.2f%% accurate at predicting handwritten digits.\n", accuracy*100.0)
}
