import numpy as np

# Cross Entropy: Given the set of sample events, how closely do they match the probability that the given events
# would take place. Thus this is the gap between expected and actual results. The smaller the Cross Entropy,
# the better the model. This is defined mathematically as the sum of the negatives of the probability of the actual
# sample(training) data.

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


print(cross_entropy([1, 1, 0], [0.7, 0.9, 0.1]))  # Low Cross Entropy as expected
print(cross_entropy([0, 0, 1], [0.7, 0.9, 0.1]))  # High Cross Entropy
