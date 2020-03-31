# Machine Learning -  Neural Network

import random
import numpy as np


class NeuralNetwork:

    def __init__(self, inputs, targets, hidden, n_outputs, learning_rate, epochs, batch_size):
        """
        Initialise the artificial neural network.
        :param inputs: numpy array with the first axis indicating the size of the input dataset. The second axis indicates number of input
        nodes to the neural network. 60000x728 training dataset. This dataset consists of 60000 images each with 728 input nodes.
        The input data must be normalised!
        :param targets: target outputs corresponding to each input data object. 60000 labels for each of the 60000 input images.
        :param hidden: number of nodes in the hidden layers. [16, 12, 8] would indicate three hidden layers with 16, 12, and 8 nodes
        :param n_outputs: number of outputs. Equivalent to the number of output nodes in the final layer of the network.
        :param learning_rate: neural network learning rate

        There will be n_layers - 1 `output` arrays. The weights between each network connection control the 'strength' of an input
        (how much influence this value should have on the next output). The number and shape of the weight matrices are dependant on
        the network structure (how many layers there are and how many neurons are in each layer).
        e.g. With an input of 3 neurons [x1; x2; x3], and the first hidden layer consisting of 4 neurons, the weight matrix will be
        [[w11, w12, w13]; [w12, w22, w32]; [w13, w23, w33]; [w14, w24, w34]]. This is a (4x3) matrix. This structure holds for other layers
        and therefore the weight matrices for each step in the feedforward propagation can be initialised using the known network structure.

        The input matrix z will store the weighted sum input to node j in layer L before the activation is applied transforming it to the
        output of node j. The matrices will be of size (nx1) where n is the number of nodes in layer L. The input matrices are required for
        the backpopagation algorithm when determining the derivative of the activation output for node j relative to the input to node j.
        """

        # training and testing data
        self.training_inputs = inputs  # training dataset inputs
        self.targets = targets  # target outputs for the training data
        self.n_data = len(self.training_inputs)  # size of training data

        # network structure
        self.n_inputs = len(self.training_inputs[0])  # number of input nodes
        self.hidden_layers = hidden  # list containing the number of neurons in each hidden layer. Layer number is the index.
        self.n_outputs = n_outputs  # number of outputs in the final layer
        n_nuerons = [[self.n_inputs], self.hidden_layers, [self.n_outputs]]
        self.layers = [x for s in n_nuerons for x in s]  # list of the number of neurons in each layer. Layer number is the index
        self.n_layers = len(self.layers)  # number of layers in the network

        # hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size  # batching is not implemented yet.. using online learning

        # initialise matrices
        self.b, self.z = [], []
        for neurons in self.layers[1:]:  # loop through all layers except the input network layer
            self.b.append(np.random.randn(neurons))  # biases
            self.z.append(np.zeros(neurons))  # layer weighted sum inputs
        self.a = [np.zeros(n) for n in self.layers]  # activation outputs - including inputs to network as activation outputs for layer 0
        self.w = [np.random.rand(self.layers[i + 1], self.layers[i]) - 0.5 for i in range(self.n_layers - 1)]  # weights

        self.train()

    def train(self):
        """
        Train the neural network. The training of the network involves first feeding forward the input data though the network of hidden
        layers. The inputs are modified by the weights connecting each node in the subsequent layers i.e. The input to node J is the
        weighted sum of the connections from the previous layer. An activation function is then applied to this input and the process
        continues until the output is obtained. Once the feedforward is complete, backpropagation is used to make adjustments to each weight
        in the previous layers. Once the weights are updated the prcess repeats for the next training data sample. The number of epochs
        is how many passes are made on the entire set of training data.
        """
        for epoch in range(self.epochs):
            random.shuffle(self.training_inputs)
            sum_error = 0
            for index in range(self.n_data):
                # forward pass
                x = self.training_inputs[index]
                self.a[0] = x
                outputs = self.feedforward(x)
                sum_error += sum(self.cost(outputs, index))
                # backward pass
                dw, db = self.backpropagation(index)
                print(dw[0].shape, db[0].shape)
                self.update_weights(dw, db)
            print('>epoch=%d, error=%.3f' % (epoch, sum_error))

    def feedforward(self, output):
        """
        Propagating the inputs through the neural network to get some output.
        The feedforward first requires taking the dot product of the weight matrix with the corresponding
        input array at the current step i.e. dot(W, X). Each element of the resulting vector is then passed
        through the activation function to get the final output array.

        There will be n_layers - 1 output arrays. These output arrays will hold the results
        of the feedforward propagation. Each array will have a number of elements equal to the number
        the size of the dot product result between the weight and input matrices.

        e.g. Moving forward from an input layer of 3 neurons, to a hidden layer with 4 neurons,
        the output of the dot product between these two layers will have 4 elements.
        input matrix = [x1, x2, x3]^T
        weight matrix = [[w11, w21, w31], [w12, w22, w32], [w13, w23, w33], [w14, w24, w34]]
        output = [xh1, xh2, xh3, xh4]
        The output array for the first feedforward step will have 4 outputs to use as inputs in the next step.
        Obviously each output array will have a number of elements equal to the number neurons in that layer.
        :param output: output array for the current step. Used as input array unless it is the final layer in network
        """
        for layer in range(self.n_layers - 1):
            self.z[layer] = np.dot(self.w[layer], self.a[layer]) + self.b[layer]  # layer inputs
            self.a[layer + 1] = self.sigmoid(self.z[layer])  # output activations
        return self.a[-1]

    def feedforward_eval(self, output, layer=0):
        """
        Feedforward through the network. This method is used for testing and validation once the network has been trained.
        It does not include a backward pass of the data and only maps an input to an output which can be used to compare the output
        to the testing target data.
        :param output: output activation from a given layer. NOTE: The first input ``output`` is the input (x) to the network.
        :param layer: the current layer in the feedforward propagation
        :return: the final outputs from the network
        """
        if layer == self.n_layers - 1:
            return output
        return self.feedforward_eval(self.sigmoid(np.dot(self.w[layer], output)), layer + 1)

    def backpropagation(self, index):
        """
        The objective of backpropagation is to update the weight matrices which transition the inputs from one layer
        to the next. Updating the weights is necessary to minimise the loss or cost function (the output error).
        The method used to adjust these weights and minimise the error is gradient descent.
        Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable
        function. To find a local minimum of a function using gradient descent, take steps proportional to the negative
        of the gradient of the function at the current point
        :param index: index of the dataset being feedforward required to get target values
        :return dw, db: gradient vectors for updating weights and biases respectively
        """
        # initialise gradient matrices
        dw = [np.zeros(w.shape) for w in self.w]
        db = [np.zeros(n) for n in self.layers[1:]]

        # loop backwards through network layers
        for i in reversed(range(1, self.n_layers)):
            if i == self.n_layers - 1:
                # output layer
                db[-1] = (self.targets[index] - self.a[-1]) * self.sp(self.z[-1])
                dw[-1] = np.dot(np.asmatrix(db[-1]).transpose(), np.asmatrix(self.a[-2]))
            else:
                # all other layers
                # NOTE: to compute the weight gradient vector for hidden layer 1 the inputs to the network are used as `activation outputs`
                db[i - 1] = np.multiply(np.dot(np.transpose(self.w[i]), db[i]), self.sp(self.z[i - 1]))
                dw[i - 1] = np.dot(np.asmatrix(db[i - 1]).transpose(), np.asmatrix(self.a[i - 2]))

        return dw, db

    def update_weights(self, dw, db):
        """
        Update the weight and bias matrices with the results obtained from the forward and backpropagation algorithms for
        a single input dataset. dw and db are the gradient vectors for the weights and biases. The gradience vectors relate the
        changes in the weights and biases to the changes in the cost function.
        :param dw: gradient of the weight matrix
        :param db: gradient of the bias matrix
        """
        self.w = np.array([w - self.learning_rate * nw for w, nw in zip(self.w, dw)])
        self.b = np.array([b - self.learning_rate * nb for b, nb in zip(self.b, db)])

    def cost(self, outputs, index):
        """
        The quadratic cost function.
        The cost (or error or loss) function is used to analyse the difference between the target outputs and the desired outputs
        during the training of the neural network. The purpose of training the model is to minimise this function. The minimisation
        algorithm used is gradient descent and is implemented in the backpropagation method.

        The cost is the squared difference of the desired output for node j in the output layer L and the activation output of node j in L.
        To calculate the Loss, sum the squared difference for each node j in the output layer L.
        :param outputs: estimated output values
        :param index: training data index
        :return: the error (loss) for node j in layer L. Difference between target and guess
        """
        return (self.targets[index] - outputs) ** 2

    @staticmethod
    def sigmoid(z):
        """
        The activation function is an abstraction representing the rate of action potential firing in the cell.
        Choice of activation function will influence the performance of the neural network. The sigmoid "squishification"
        function is used here.
        param x: input value for sigmoid squishification
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sp(z):
        """
        The derivative of the activation function. In this case, sigmoid'. sigmoid_prime.
        :param z: evaluate the slope of the sigmoid function at z
        :return: the rate of change of the activation function evaluated at z
        """
        return z * (1.0 - z)
