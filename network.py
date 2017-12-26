import numpy as np


class Network(object):

    # initialize parameters
    def __init__(self, layer_properties):

        self.num_layers = len(layer_properties)
        self.layer_dims = [int(i) for i in (layer_properties[:,0])]
        self.layer_fcs = layer_properties[:,1]
        self.layer_dropouts = [float(i) for i in (layer_properties[:,2])]
        self.layer_dropouts[-1] = 1.0
        self.parameters = {}
        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * \
                                            np.sqrt(1/self.layer_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    # forward propagation with dropout
    # use dropout in all layers except input layer and output layer
    def forward_propagation_with_dropout(self, minibatch_X):
        caches = {'A0': minibatch_X}
        for l in range(1, self.num_layers):
            caches['Z' + str(l)] = np.dot(self.parameters['W' + str(l)], caches['A' + str(l - 1)]) + \
                                   self.parameters['b' + str(l)]

            if self.layer_fcs[l] == "sigmoid":
                caches['A' + str(l)] = sigmoid(caches['Z' + str(l)])
            elif self.layer_fcs[l] == "tanh":
                caches['A' + str(l)] = np.tanh(caches['Z' + str(l)])
            elif self.layer_fcs[l] == "relu":
                caches['A' + str(l)] = relu(caches['Z' + str(l)])
            elif self.layer_fcs[l] == "softmax":
                caches['A' + str(l)] = softmax(caches['Z' + str(l)])
            else:
                print("\ninvalid or unsupported function!\n")
                exit(1)  # unsuccessful termination

            caches['D' + str(l)] = np.random.rand(caches['A' + str(l)].shape[0], caches['A' + str(l)].shape[1])
            caches['D' + str(l)] = caches['D' + str(l)] < self.layer_dropouts[l]
            caches['A' + str(l)] = np.multiply(caches['A' + str(l)], caches['D' + str(l)])
            caches['A' + str(l)] = caches['A' + str(l)] / self.layer_dropouts[l]

        return caches['A' + str(self.num_layers - 1)], caches

    # cross entropy loss
    @staticmethod
    def compute_cost_cross_entropy(y_hat, y):
        m = y.shape[1]
        cost = (-1. / m) * np.nan_to_num(np.sum(np.multiply(y, np.log(np.maximum(y_hat,10**-10))) +
                                                np.multiply((1 - y), np.log(np.maximum(1 - y_hat,10**-10)))))
        np.squeeze(cost)
        return cost

    # back-propagation with dropout
    def backward_propagation_with_dropout(self, minibatch_Y, y_hat, caches):
        grads = {}
        m = minibatch_Y.shape[1]
        dAL = - (minibatch_Y/(y_hat+10**-10)) + ((1-minibatch_Y)/(1-y_hat+10**-10))

        for l in range(self.num_layers - 1, 0, -1):
            if self.layer_fcs[l] == "sigmoid":
                dZ = dAL * sigmoid_prime(caches['Z' + str(l)])
            elif self.layer_fcs[l] == "tanh":
                dZ = dAL * tanh_prime(caches['Z' + str(l)])
            elif self.layer_fcs[l] == "relu":
                dZ = dAL * relu_prime(caches['Z' + str(l)])
            elif self.layer_fcs[l] == "softmax":
                dZ = dAL * softmax_prime(caches['Z' + str(l)])

            grads['dW' + str(l)] = (1.0 / m) * np.dot(dZ, caches['A' + str(l - 1)].T)
            grads['db' + str(l)] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            dAL = np.dot(self.parameters['W' + str(l)].T, dZ)
            if l > 1:
                dAL = dAL * caches['D' + str(l - 1)]
                dAL = dAL / self.layer_dropouts[l - 1]
        return grads

    # initialization of the adam variables
    def initialize_adam(self):
        v = {}
        s = {}
        for l in range(1, self.num_layers):
            v['dW' + str(l)] = np.zeros(
                (self.parameters['W' + str(l)].shape[0], self.parameters['W' + str(l)].shape[1]))
            v['db' + str(l)] = np.zeros(
                (self.parameters['b' + str(l)].shape[0], self.parameters['b' + str(l)].shape[1]))
            s['dW' + str(l)] = np.zeros(
                (self.parameters['W' + str(l)].shape[0], self.parameters['W' + str(l)].shape[1]))
            s['db' + str(l)] = np.zeros(
                (self.parameters['b' + str(l)].shape[0], self.parameters['b' + str(l)].shape[1]))

        return v, s

    # param update with adam optimization
    def update_parameter_with_adam(self, grads, v, s, adam_counter, learning_rate=0.1,
                                   beta1=0.9, beta2=0.999,  epsilon=1e-8):

        v_corrected = {}
        s_corrected = {}
        for l in range(1, self.num_layers):
            v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
            v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
            v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - (beta1 ** adam_counter))
            v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - (beta1 ** adam_counter))

            s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * np.power(grads['dW' + str(l)], 2)
            s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * np.power(grads['db' + str(l)], 2)
            s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - (beta2 ** adam_counter))
            s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - (beta2 ** adam_counter))
            self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * \
                                                                            (v_corrected['dW' + str(l)] / np.sqrt(
                                                                                s_corrected['dW' + str(l)] + epsilon))
            self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * \
                                                                            (v_corrected['db' + str(l)] / np.sqrt(
                                                                                s_corrected['db' + str(l)] + epsilon))
        return v, s

    # param update with gradient descent
    def update_parameters_with_gd(self, grads, learning_rate):

        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]


# create random mini batch out of data-set
# X -- input features, shape(feature_dim, m)
# Y -- ground truth labels, shape(output_dim, m)
def random_mini_batches(X, Y, mini_batch_size):

    m = X.shape[1]  # number of training examples in a minibatch
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in partitioning.
    num_complete_minibatches = int(m / mini_batch_size)
    k = 0
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handle end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# sigmoid activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-np.nan_to_num(Z)))


# sigmoid derivative w.r.t. Z
def sigmoid_prime(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


# softmax activation function
def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


def softmax_prime(Z):
    Z_new = np.nan_to_num(Z)
    exp_sum = np.sum(np.exp(Z_new), axis=0, keepdims=True)
    return (exp_sum*np.nan_to_num(np.exp(Z_new)) - np.nan_to_num(np.exp(Z_new))** 2)/(exp_sum ** 2 + 10**-10)


def tanh_prime(Z):
    return 1 - np.power(np.tanh(np.nan_to_num(Z)), 2)


# relu activation function
def relu(Z):
    return np.maximum(np.nan_to_num(np.array(Z)), 10**-10)


def relu_prime(Z):
    return np.array(np.nan_to_num(Z)) > 0


# for first and last layer dropout 1 by default
# network_prop = [[786, None, 1.0], [60, "relu", 0.6], [30, "relu", 0.8], [10, "softmax", 1.0]]
# net = Network(network_prop)