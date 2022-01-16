# standard library
import random
# third party library
import numpy as np
# data file

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


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
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

        

    def update_mini_batch(self, mini_batch, eta):
            
        # updating w and bs using gradient descent/backpropagation
        """taking the already existing arrays
        then seems to be making an identical shape array with 0s
        assumedly for transposing the new data on in creation of an updated matrix
        eta is the learning rate
        mini_batch is a list of tuples, assuming of inputs and desired outputs?"""
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
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
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