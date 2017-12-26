import network
import numpy as np
import data_loader
import matplotlib.pyplot as plt


def predict(X, Y, obj):
    a, _ = obj.forward_propagation_with_dropout(X)
    result = [(np.argmax(a[:,i]),np.argmax(Y[:,i])) for i in range(Y.shape[1])]
    return sum(int(x == y) for (x, y) in result)


def get_dset(data):
    X, Y = zip(*data)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1])).T
    Y = Y.reshape((Y.shape[0], Y.shape[1])).T

    return (X, Y)


def model(layer_properties, learning_rate=0.001, mini_batch_size=64, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=100, learning_decay_rate=0.7, decay_epoch=30, learning_decay=False, print_cost=True):

    # load training, validation, and test data
    training_data, validation_data, test_data = data_loader.load_data_wrapper()

    X, Y = get_dset(training_data)      # getting training data-set, X -- shape(features, m)
    c, d = get_dset(validation_data)    # getting test data-set, Y -- shape(ground-truth-label, m)
    costs = []
    adam_counter = 0
    net = network.Network(layer_properties)
    # v, s = net.initialize_adam()  # --uncomment this line to if you are using the adam optimization

    m_train = Y.shape[1]
    m_val = d.shape[1]

    for i in range(num_epochs):

        cost = 0
        minibatches = network.random_mini_batches(X, Y, mini_batch_size)

        if learning_decay and i%decay_epoch == 0 and i > 0:
            learning_rate *= learning_decay_rate

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a, caches = net.forward_propagation_with_dropout(minibatch_X)
            cost = net.compute_cost_cross_entropy(a, minibatch_Y)
            grads = net.backward_propagation_with_dropout(minibatch_Y, a, caches)
            # UNCOMMENT ONLY NEXT LINE 1 AND 2 IF YOU ARE USING ADAM OPTIMIZATION
            # UNCOMMENT ONLY LINE 3 IF YOU ARE USING GRADIENT DESCENT
            # 1.adam_counter = adam_counter + 1
            # 2.net.update_parameter_with_adam(grads, v, s, adam_counter, learning_rate, beta1, beta2, epsilon)
            # 3.net.update_parameters_with_gd(grads, learning_rate)

        if print_cost and i%10 == 0:
            costs.append(cost)
            print("Cost after epoch %i: %f\t learning rate: %f" % (i, cost, learning_rate))
            print("validation set accuracy: %d/%d" % (predict(c, d, net),m_val))

    A, B = get_dset(test_data)

    m_test = B.shape[1]
    print("test set accuracy: %d/%d" % (predict(A, B, net),m_test))
    print("train set accuracy: %d/10000" % predict(X[:, 0:10000], Y[:, 0:10000], net))

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()


# network_prop_array -- numpy array of network properties containing
# [no. of neurons in each layer, "Activation function to use for a layer", drop-out] for each layer.
# for example to create a 4-layer deep feedforward neural network 'network_properties' is
# defined as
network_properties = [[784, "None", 1.0], [60, "relu", 0.8], [30, "relu", 0.9], [30, "relu", 0.9], [10, "softmax", 1.0]]
# NOTE that in the above line [784, "None", 1.0] corresponds to the 0th layer (i.e, input layer)
# Different Activation functions that are supported are:
# "sigmoid", "tanh", "relu", "softmax"
# TO CREATE DIFFERENT LAYERS OF NEURAL NETWORK JUST CHANGE ABOVE PROPERTIES
# ANY NUMBER OF LAYERS CAN BE CREATED.

network_prop_array = np.array(network_properties)
model(network_prop_array)
