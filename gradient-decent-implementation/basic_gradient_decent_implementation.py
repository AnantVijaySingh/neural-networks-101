# Gradient Descent: The Code
#
# From before we saw that one weight update can be calculated as:
#
# Δwi=ηδxi
#
# with the error term δ as
#
# δ=(y−y^)f′(h)=(y−y^)f′(∑wixi) \delta = (y - \hat y) f'(h) = (y - \hat y) f'(\sum w_i x_i)δ=(y−y^​)f′(h)=(y−y^​)f′(
# ∑wi​xi​)
#
# Remember, in the above equation (y−y^)(y - \hat y)(y−y^​) is the output error, and f′(h)f'(h)f′(h) refers to the
# derivative of the activation function, f(h)f(h)f(h). We'll call that derivative the output gradient.
#
# Now I'll write this out in code for the case of only one output unit. We'll also be using the sigmoid as the
# activation function f(h)f(h)f(h).

import numpy as np


# Defining the sigmoid function for activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Input data
x = np.array([1, 2, 3, 4])

# Target
y = np.array(0.5)

# Initial weights
weights = np.array([0.5, -0.5, 0.3, 0.1])

# The learning rate, eta in the weight step equation
learnrate = 0.5

# TODO: Calculate the node's linear combination of inputs and weights
# the linear combination performed by the node (h in f(h) and f'(h))
# h = x[0] * weights[0] + x[1] * weights[1] ...
h = np.dot(x, weights)

# TODO: Calculate output of neural network
# The neural network output (y-hat)
nn_output = sigmoid(h)

# TODO: Calculate error of neural network
# output error (y - y-hat)
error = y - nn_output

# output gradient (f'(h))
output_grad = sigmoid_prime(h)

# TODO: Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.
# error term (lowercase delta)
error_term = error * output_grad

# TODO: Calculate change in weights
# Gradient descent step
# del_w = [learnrate * error_term * x[0],
#          learnrate * error_term * x[1],
#          . . .]
del_w = learnrate * error_term * x


print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
