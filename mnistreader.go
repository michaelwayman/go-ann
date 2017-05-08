package main

import (
	"encoding/csv"
	"errors"
	// "fmt"
	"io"
	"os"
	"strconv"
)

func ReadMNISTCSVData(path string) [][]int {
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

func CleanedDataForNN(path string) func() ([]float64, []float64, error) {
	csvData := ReadMNISTCSVData(path)
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
