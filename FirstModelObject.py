# standard library
import random
# third party library
import numpy as np
# data file
import mnist_loader
# pandas because why not
import pandas as pd

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

"""A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation. It is not optimized, and omits many desirable features"""

# here, I am defining a neural network as a class "network"
# i.e. network([3,3,4]) describes a net with 3 input layers, 3 hidden, and 4 output

class Network(object):
    def __init__(self, sizes):
        # counts layers input into the model
        self.num_layers = len(sizes)
        # amount of neurons in each respective layer.
        # is this called by ...sizes[?]
        self.sizes = sizes
        # initializing biases - returns a random gaussian distribution
        # mean 0 and variance of 1
        # The book will explore better ways of finding initializing conditions in later chapters
        # I am still a bit confused on what rand(y,1) inputs do
        # returns a matrix
        self.biases = [np.random.randn(y ,1) for y in sizes [1:]]
        # returns a matrix of random weights for each layer (y) to start
        # still a bit unsure of what (y,x) means, specifically x
        # what is this zipping, means a tuple right?
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input""" 
        # is for b, w the same thing as two separate loops combined in one
        # is .dot just a multiplication of matrices that contain
        # how is "a" a variable as well as the output
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

# mini-batch stochastic gradient descent was used to train the neural network
# gives inputs and desired outputs in a tuple and each epoch approaches outcome

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                # how is it defining a range k when it hasn't been created
                for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                # to be defined later
                self.update_mini_batch(mini_batches, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
    
# updating w and bs using gradient descent/backpropagation
"""taking the already existing arrays
then seems to be making an identical shape array with 0s
assumedly for transposing the new data on in creation of an updated matrix
eta is the learning rate
mini_batch is a list of tuples, assuming of inputs and desired outputs?"""

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

# returns a tuple representing the gradient vector for the cost function
# this should be fun 

    def backprop(self, x , y): 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    #feedforward
        activation = x
        activations = [x] # list to store all activations, layer by layer
        zs = [] # list to store all z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * \sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # takes advantage of the fact that python can do negative indices
        for l in range(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-1+1].transpose(). delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-1-1].transpose())
        return (nabla_b, nabla_w)
    
# returning amount of instances where the model predicted correctly

    def evaluate(self, test_data): 
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)

#         """Return the vector of partial derivatives partial C_x partial a for the output activations."""

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(z):
        # derivative of the sigmoid function
        return sigmoid(z)*(1-sigmoid(z))