## Artificial Neural Network in GO

A simple 3-layer ANN (artificial neural network) written in Go.

**Coming Soon** Configurable layers and CLI tool.

Note: Even when utilizing Go's ability to parallelize, I am only able to "match" the speeds Python gives me implementing the same algorithm. I'm assuming because `numpy` is so effective. I am working on a way to increase the performance of matrix math.


## The example

The example uses the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) to train and test the neural network.

The MNIST (Modified National Institute of Standards and Technology) database contains 60,000 training images and 10,000 testing images of handwritten numbers from 0-9.

#### Download

 - Training Data - [download](https://pjreddie.com/media/files/mnist_train.csv)
 - Testing Data - [download](https://pjreddie.com/media/files/mnist_test.csv)


## Performance

There are 2 branches for this project each with differing performance.

 1. `master` Uses my own brand of matrix and matrix operations
    - takes ~85s
 1. `mat64` Uses the github.com/gonum/matrix library for matrix operations
    - takes ~290s

**Note:** the same setup in *python* can be found [here](https://github.com/michaelwayman/python-ann) and **only takes ~45s**. We can match this number by adding *goroutines* to the `Train()` function but I'm trying to get the matrix math to perform better in general.
