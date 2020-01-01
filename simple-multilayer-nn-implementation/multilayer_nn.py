import numpy as np
from data_prep import features, targets, features_test, targets_test


# Helper functions
# Activation (sigmoid) function
def sigmoid(inputvalue):
    return 1 / (1 + np.exp(-inputvalue))


# Use to same seed to make debugging easier
np.random.seed(21)

# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

# Number of records and input units
n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights_input_to_hidden = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden))
weights_hidden_to_output = np.random.normal(scale=1 / n_features ** .5, size=n_hidden)

print('weights_input_to_hidden', weights_input_to_hidden)
print('weights_hidden_to_output', weights_hidden_to_output)

for e in range(epochs):
    del_w_hidden_output = np.zeros(weights_hidden_to_output.shape)
    del_w_input_hidden = np.zeros(weights_input_to_hidden.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # ----- Forward pass -----
        # TODO: Calculate the hidden layer output
        hidden_layer_input = np.dot(x, weights_input_to_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        # TODO: Calculate the hidden layer output
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
        output = sigmoid(output_layer_input)

        # ----- Backpropagation -----
        # TODO: Calculate the output error
        error = y - output

        # TODO: Calculate error term for output unit
        output_error_term = error * output * (1 - output)

        # Propagate errors to hidden layer
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_to_output)

        # TODO: Calculate error term for hidden layer
        hidden_error_term = hidden_error * hidden_layer_output * (1 - hidden_layer_output)

        # TODO: Calculate change in weights for hidden layer to output layer
        del_w_hidden_output += output_error_term * hidden_layer_output

        # TODO: Calculate change in weights for input layer to hidden layer
        del_w_input_hidden += hidden_error_term * x[:, None]

    # TODO: Update weights using the learning rate and the average change in weights
    weights_input_to_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_to_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(features, weights_input_to_hidden))
        out = sigmoid(np.dot(hidden_output, weights_hidden_to_output))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_to_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_to_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
