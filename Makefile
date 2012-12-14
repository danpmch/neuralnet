
network_test:
	g++ neuron.h neuralnetwork.cpp network_test.cpp -o network_test -g -Wall -Wextra


perceptron_test:
	g++ neuron.h perceptron.h perceptron_test.cpp -o perceptron_test -g -Wall -Wextra
