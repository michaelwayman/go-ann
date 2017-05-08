package main

import (
	"fmt"
)

func TrainNeuralNetwork(nn NeuralNetwork) {
	cleanedTrainingData := CleanedDataForNN("mnist_train.csv")
	for {
		inputs, targets, err := cleanedTrainingData()
		if err != nil {
			break
		}
		nn.train(inputs, targets)
	}
}

func TestNeuralNetwork(nn NeuralNetwork) float64 {

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
		accuracy = scoreboard(nn.query(inputs), targets)
	}
	return accuracy
}

func main() {
	fmt.Println("Starting neural network to recognize handwritten digits.")

	// 0
	nn := NeuralNetwork{
		numberInputs:      784,
		numberOutputs:     10,
		numberHiddenNodes: 200,
		learningRate:      0.1,
		inputWeights:      RandomWeightMatrix(200, 784),
		outputWeights:     RandomWeightMatrix(10, 200),
	}

	// 1
	fmt.Println("Training the neural network.")
	TrainNeuralNetwork(nn)

	// 2
	fmt.Println("Testing the neural network.")
	accuracy := TestNeuralNetwork(nn)

	// 3
	fmt.Printf("Neural network is %.2f%% accurate at predicting handwritten digits.\n", accuracy*100.0)
}
