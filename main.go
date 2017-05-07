package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
)

func readCSVData(path string) [][]int {
	file, err := os.Open(path)
	if err != nil {
		panic("AHH")
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var csvData [][]int

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}

		var values = make([]int, len(record))
		for i, v := range record {
			intValue, err := strconv.Atoi(v)
			if err != nil {
				break
			}
			values[i] = intValue
		}
		csvData = append(csvData, values)
	}

	return csvData
}

func DataForNN(path string) func() ([]float64, []float64, error) {
	csvData := readCSVData(path)
	counter := 0
	return func() ([]float64, []float64, error) {
		if counter == len(csvData) {
			return nil, nil, errors.New("Done")
		}
		target := csvData[counter][0]
		input := csvData[counter][1:]

		inputs := make([]float64, len(input))
		for i, v := range input {
			inputs[i] = (float64(v) / 255.0 * 0.99) + 0.01
		}

		targets := make([]float64, 10)
		for i := range targets {
			targets[i] = 0.01
		}
		targets[target] = 0.99
		counter += 1
		return inputs, targets, nil
	}
}

func MaxIndex(array []float64) int {
	mi := 0
	mv := array[0]
	for i, v := range array[1:] {
		if v >= mv {
			mi = i
			mv = v
		}
	}
	return mi
}

func main() {
	nn := NewNeuralNetwork(784, 10, 200, 0.1)

	fmt.Println("Beginning NN training.")
	cleanedTrainingData := DataForNN("mnist_train.csv")
	for {
		inputs, targets, err := cleanedTrainingData()
		if err != nil {
			break
		}
		nn.train(inputs, targets)
	}
	fmt.Println("Complete.")

	fmt.Println("Beginning NN testing.")
	right, total := 0, 0
	cleanedTestingData := DataForNN("mnist_test.csv")
	for {
		inputs, targets, err := cleanedTestingData()
		if err != nil {
			break
		}
		if MaxIndex(nn.query(inputs).ToArray()) == MaxIndex(targets) {
			right += 1
		}
		total += 1
	}
	fmt.Println("Complete.")
	fmt.Println("NN is ", float64(right)/float64(total), "% accurate.")
	fmt.Println(right, total)
}
