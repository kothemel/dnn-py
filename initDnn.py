import numpy as np 

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwrok:
    def __init__ (self, x, y, layer1Size):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], layer1Size)
        self.weights2   = np.random.rand(layer1Size, 1)
        self.y          = y
        self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input,  self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
    
    def backprop(self):
        # error in ouput
        output_error = 2*(self.y - self.output)
        output_delta = output_error * sigmoid_derivative(self.output)

        #how much layer1 weights contribute to output error
        layer1_error = np.dot(output_delta, self.weights2.T)
        layer1_delta = layer1_error * sigmoid_derivative(self.layer1)

        self.d_weights2 = np.dot(self.layer1.T,  output_delta)
        self.d_weights1 = np.dot(self.input.T,   layer1_delta)

        # Adjust weights. First for (input -> hidden), then for (hidden --> output)
        self.weights1 += self.d_weights1
        self.weights2 += self.d_weights2

if  __name__ == "__main__":

    # Input section
    inputN = int(input("Enter the size of input vector: "))
    hiddenLayerSize = int(input("Enter the size of the hidden layer: "))

    # Randomize dnn input and output layer
    X = np.random.rand(inputN,3)
    y = np.random.rand(inputN,1)

    nn = NeuralNetwrok(X, y, 8)
    for i in range (3000):
        nn.feedforward()
        nn.backprop()
    print(nn.output)

    with open('dnn.txt', 'w+') as fileT:
        fileT.write("Input  X: "    +str(X)+"\n\n"+ 
                    "Output Y: "    +str(y)+"\n\n"+
                    "Predictions: " +str(nn.output))
