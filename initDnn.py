import numpy as np
import scipy as sc
import pylab
import os
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
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    # Use softmax in output layer as an activation function
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def rand_2D(rows, cols):
    '''Gets the number of rows and columns and creates
    a 2-D array of random numbers

    Arguments:
        rows {integer} -- number of rows
        cols {integer} -- number of columns

    Returns:
        [type] -- [description]
    '''

    np.random.seed(0)
    return np.random.rand(rows, cols)


class NeuralNetwork:
    def __init__(self, x, y, input_len, output_len, hidden_len):
        ''' Constructor for Neural Network class.

        Arguments:
            x {numpy.ndarray}    -- Input vector for NN
            y {numpy.ndarray}    -- The output classes for each input
            input_len {integer}  -- The size of input layer (in nodes)
            output_len {integer} -- The size of output layer (in classes)
            hidden_len {integer}  -- The size of each hidden layer (in nodes)
        '''

        self.input = x
        self.weight_1 = rand_2D(input_len, hidden_len) / np.sqrt(input_len)
        self.weight_2 = rand_2D(hidden_len, output_len) / np.sqrt(hidden_len)
        self.bias_1 = np.zeros((1, hidden_len))
        self.bias_2 = np.zeros((1, output_len))
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
                "Input-Layer1  Weight Size: " + str(self.weight_1.shape)+"\n" +
                "Layer1-Output Weight Size: " + str(self.weight_2.shape)+"\n")

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
            z1 = self.input.dot(self.weight_1) + self.bias_1
            layer1 = np.tanh(z1)
            output_layer = layer1.dot(self.weight_2) + self.bias_2
            exp_scores = np.exp(output_layer)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            output_delta = probs
            output_delta[range(examplesN), y] -= 1
            dW2 = (layer1.T).dot(output_delta)
            db2 = np.sum(output_delta, axis=0)

            # hidden layer strength on output
            layer1_delta = output_delta.dot(self.weight_2.T)
            layer1_delta *= (1 - np.power(layer1, 2))
            dW1 = np.dot(X.T, layer1_delta)
            db1 = np.sum(layer1_delta, axis=0)

            # Add regularization terms
            dW2 += self.reg_lambda * self.weight_2
            dW1 += self.reg_lambda * self.weight_1

            # Gradient descent parameter update
            self.weight_1 += -self.epsilon * dW1
            self.weight_2 += -self.epsilon * dW2
            self.bias_1 += -self.epsilon * db1
            self.bias_2 += -self.epsilon * db2

            # Assign new parameters to the model
            model = {'W1': self.weight_1, 'b1': self.bias_1,
                     'W2': self.weight_2, 'b2': self.bias_2}
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
    plt.show()
