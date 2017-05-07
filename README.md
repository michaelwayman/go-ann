## Neural Network to solve the MNIST data set in Google's GoLang

I have the same NN written python in a different repository.
The Python NN obviously uses numpy for its calculations whereas in this Go NN I have numerous O(n^3) and O(n^2) calculations taking place.

They are running at similar speeds at the moment, the Golang might be a little faster but they are close.
My goal is to get them running at about the same speed on single-thread, and then start throwing in all the features that has made Go famous and see how much difference simple parallelizations can make.
