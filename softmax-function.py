import numpy as np

# Softmax function is used for cases when we have more than 2 possible outcomes such as with image classification of
# different types of animals. When we only need to determine binary outputs such as dog / no dog then we can use
# sigmoid. For n =2 softmax is still valid and gives the same output as sigmoid.

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    return np.divide(expL, expL.sum())

    # Note: alternative
    # sumExpL = sum(expL)
    # result = []
    # for i in expL:
    #     result.append(i * 1.0 / sumExpL)


print(softmax(4))
