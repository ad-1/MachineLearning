# Machine learning - Neural Network
import numpy as np
from matplotlib import pyplot as plt
from neural_network import nn

# Program driver
if __name__ == '__main__':
    data = np.load('mnist.npz')
    (training_images, test_images) = data['x_train'], data['x_test']
    (training_labels, test_labels) = data['y_train'], data['y_test']
    # display training data
    # print(training_labels[0], np.size(training_images), np.size(training_labels))
    # plt.figure(1)
    # plt.imshow(np.reshape(training_images[0], (28, 28)))
    # display testing data
    # print(test_labels[0], np.size(test_images), np.size(test_labels))
    # plt.figure(2)
    # print(len(np.array(test_images[0]).flatten()))
    # print(np.array(test_images[0]).shape)
    # plt.imshow(np.reshape(test_images[0], (28, 28)))
    # plt.show()
    # training_inputs = [[1, 1], [0, 1], [1, 0], [0, 0]]  # OR
    # training_labels = [1, 1, 1, 0]
    # x_inputs, self.training_labels = inputs
    n_outputs = 10
    training_inputs = np.array([data.flatten() / 255 for data in training_images])
    training_targets = []
    for i in range(len(training_labels)):
        training_targets.append([1 if training_labels[i] == x else 0 for x in range(n_outputs)])
    nn = nn.NeuralNetwork(training_inputs, np.array(training_targets), [16], n_outputs, learning_rate=0.1, epochs=10, batch_size=100)
