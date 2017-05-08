## Neural Network in GO

You won't find any crazy 3rd party libraries here, just whatever built in functionality comes with GO.
I wrote my own matrix helpers such as dot product O(n^3) and the like, leaving a lot of room for performance improvements.

Right now, (single thread, single core)

 - reading in the training data,
 - training
 - testing for accuracy

takes ~88 seconds on my computer


I want to see if using parallelization or other features of GO, if I can reduce this number pretty significantly.
Right now my first guess, is to parallelize the "train" function of the neural network, and use channels to get the weight adjustments. Stay tuned for more updates.

To run the example you first need to download the CSV MNIST data set and place it in the folder.

[download training data](https://pjreddie.com/media/files/mnist_train.csv)

[download testing data](https://pjreddie.com/media/files/mnist_test.csv)

