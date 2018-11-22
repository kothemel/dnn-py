import numpy as np
import scipy as sc
import pylab
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from matplotlib import cm, figure


def generateData(inputSize):
    '''This function generates a random set of data in 2D space
    given an input number of points.

    Arguments:
        inputSize {integer} -- The number of points in 2D plane
    '''

    np.random.seed(0)
    X, y = datasets.make_moons(inputSize, noise=0.20)
    return(X, y)


def plot_decision_boundary(pred_func):
    ''' Code creadits to Wei Ji in
    https://github.com/dennybritz/nn-from-scratch/blob/master/nn_from_scratch.py
    This functions generates the contour in plot.'''

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with ditsance h between them
    param1 = np.arange(x_min, x_max, h)
    param2 = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(param1, param2)

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)


def predict(model, x):
    '''Code creadits to Wei Ji in
    https://github.com/dennybritz/nn-from-scratch/blob/master/nn_from_scratch.py
    Predicts the output (0 or 1) for the given model.

    Arguments:
        model {class NeuralNetwork} -- Given the model of DNN as input,
        the function can access its Weights.

    Returns:
        integer -- The indices of the class with the higher probability.
    '''

    # Forward propagation
    W1, W2,  = model['W1'], model['W2']
    z1 = x.dot(W1)
    a1 = np.tanh(z1)
    z2 = a1.dot(W2)

    # Use softmax in output layer as an activation function
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


class NeuralNetwork:
    def __init__(self, x, y, inSize, outSize, hidden):
        ''' Constructor for Neural Network class.

        Arguments:
            x {numpy.ndarray}    -- Input vector for NN
            y {numpy.ndarray}    -- The output classes for each input
            inSize {integer}  -- The size of input layer (in nodes)
            outSize {integer} -- The size of output layer (in classes)
            hidden {integer}  -- The size of each hidden layer (in nodes)
        '''

        np.random.seed(0)
        self.input = x
        self.weights1 = np.random.rand(inSize, hidden) / np.sqrt(inSize)
        self.weights2 = np.random.rand(hidden, outSize) / np.sqrt(hidden)
        self.y = y
        self.epsilon = 0.01
        self.reg_lambda = 0.01

    def __str__(self):
        ''' This functions gets all the local variables of a NeuralNetwork object and
        creates a sting representation.

        Returns:
            string -- The object of class NeuralNetwork in string.
        '''

        return ("Input Size: " + str(self.input.shape)+"\n" +
                "Input-Layer1  Weight Size: " + str(self.weights1.shape)+"\n" +
                "Layer1-Output Weight Size: " + str(self.weights2.shape)+"\n")

    def train(self, examplesN):
        '''Forward and back propagation are applied to the NN model.

        Arguments:
            examplesN {integer} -- the size of training data

        Returns:
            class NeuralNetwork -- returns the trained model
        '''

        model = {}
        for _ in range(0, 1000):

            # Feed forward
            # z = X.T*W + b
            z1 = self.input.dot(self.weights1)
            layer1 = np.tanh(z1)
            output_layer = layer1.dot(self.weights2)
            exp_scores = np.exp(output_layer)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            output_delta = probs
            output_delta[range(examplesN), y] -= 1
            dW2 = (layer1.T).dot(output_delta)

            # hidden layer strength on output
            layer1_delta = output_delta.dot(self.weights2.T)
            layer1_delta *= (1 - np.power(layer1, 2))
            dW1 = np.dot(X.T, layer1_delta)

            # Add regularization terms
            dW2 += self.reg_lambda * self.weights2
            dW1 += self.reg_lambda * self.weights1

            # Gradient descent parameter update
            self.weights1 += -self.epsilon * dW1
            self.weights2 += -self.epsilon * dW2

            # Assign new parameters to the model
            model = {'W1': self.weights1,  'W2': self.weights2}
        return model


if __name__ == "__main__":

    # Input section
    inputSize = int(input("Enter the size of training data: "))
    X, y = generateData(inputSize)
    num_examples = len(X)

    # NN construction and training section
    plt.figure(figsize=(16, 32))
    hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50, 100]
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer size %d' % nn_hdim)
        nn = NeuralNetwork(X, y, 2, 2, nn_hdim)
        trainedNN = nn.train(num_examples)
        plot_decision_boundary(lambda x: predict(trainedNN, x))
    pylab.savefig('foo.png')
