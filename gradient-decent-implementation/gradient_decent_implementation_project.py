import numpy as np
from data_prep import features, targets, features_test, targets_test

"""

Implementing gradient decent to train neural network

"""


# Helper functions
# Activation (sigmoid) function
def sigmoid(inputValue):
    return 1 / (1 + np.exp(-inputValue))


# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# First, you'll need to initialize the weights. We want these to be small such that the input to the sigmoid is in
# the linear region near 0 and not squashed at the high and low ends. It's also important to initialize them randomly
# so that they all have different starting values and diverge, breaking symmetry. So, we'll initialize the weights
# from a normal distribution centered at 0. A good value for the scale is 1/sqrt{n}​ where n is the number
# of input units. This keeps the input to the sigmoid low for increasing numbers of input units.

# TODO: Initialize weights
weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

# Neural Network hyper parameters
epochs = 1000
learnrate = 0.5

# NumPy provides a function np.dot() that calculates the dot product of two arrays, which conveniently calculates hhh
# for us. The dot product multiplies two arrays element-wise, the first element in array 1 is multiplied by the first
# element in array 2, and so on. Then, each product is summed.

# input to the output layer
# output = np.dot(weights, inputs) which is the same as output = x[0] * weights[0] + x[1] * weights[1] ...

print('features', features.shape)
print('targets', targets.shape)

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # TODO: Calculate the output
        # Activation of the output unit
        #   Notice we multiply the inputs and the weights here
        #   rather than storing h as a separate variable
        output = sigmoid(np.dot(x, weights))

        # TODO: Calculate the error
        error = y - output

        # TODO: Calculate the error term
        # Notice we calculate f'(h) here instead of defining a separate
        #   sigmoid_prime function. This just makes it faster because we
        #   can re-use the result of the sigmoid function stored in
        #   the output variable
        error_term = error * output * (1 - output)

        # TODO: Calculate the change in weights for this sample
        #       and add it to the total weight change
        del_w += error_term * x

    # TODO: Update weights using the learning rate and the average change in weights
    # Update the weights here. The learning rate times the
    # change in weights, divided by the number of records to average
    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
