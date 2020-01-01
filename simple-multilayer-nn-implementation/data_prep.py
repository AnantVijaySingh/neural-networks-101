"""
Data and data formatting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
print(data[:10])


# Function to help us plot
def plot_points(data):
    X = np.array(data[['gre', 'gpa', 'rank']])
    y = np.array(data['admit'])
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')


# Plotting the points
plot_points(data)
plt.show()

# Data cleanup
#
# You might think there will be three input units, but we actually need to transform the data first. The rank feature
# is categorical, the numbers don't encode any sort of relative values. Rank 2 is not twice as much as rank 1,
# rank 3 is not 1.5 more than rank 2. Instead, we need to use dummy variables to encode rank, splitting the data into
# four new columns encoded with ones or zeros. Rows with rank 1 have one in the rank 1 dummy column, and zeros in all
# other columns. Rows with rank 2 have one in the rank 2 dummy column, and zeros in all other columns. And so on.
#
# We'll also need to standardize the GRE and GPA data, which means to scale the values such that they have zero mean
# and a standard deviation of 1. This is necessary because the sigmoid function squashes really small and really
# large inputs. The gradient of really small and large inputs is zero, which means that the gradient descent step
# will go to zero too. Since the GRE and GPA values are fairly large, we have to be really careful about how we
# initialize the weights or the gradient descent steps will die off and the network won't train. Instead,
# if we standardize the data, we can initialize the weights easily and everyone is happy.


# Separating the ranks
data_rank1 = data[data["rank"] == 1]
data_rank2 = data[data["rank"] == 2]
data_rank3 = data[data["rank"] == 3]
data_rank4 = data[data["rank"] == 4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()

# This looks more promising, as it seems that the lower the rank, the higher the acceptance rate. Let's use the rank
# as one of our inputs. In order to do this, we should one-hot encode it.
# Use the get_dummies function in Pandas in order to one-hot encode the data.

# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)

# Print the first 10 rows of our data
print(one_hot_data[:10])

# Scaling the data
#
# The next step is to scale the data. We notice that the range for grades is 1.0-4.0, whereas the range for test
# scores is roughly 200-800, which is much larger. This means our data is skewed, and that makes it hard for a neural
# network to handle. Let's fit our two features into a range of 0-1, by dividing the grades by 4.0, and the test
# score by 800.

# Making a copy of our data
processed_data = one_hot_data[:]

# Scale the columns / Standarize features
# processed_data['gre'] = processed_data['gre'] / 800
# processed_data['gpa'] = processed_data['gpa'] / 4
for field in ['gre', 'gpa']:
    mean, std = processed_data[field].mean(), processed_data[field].std()
    processed_data.loc[:, field] = (processed_data[field] - mean) / std

# Printing the first 10 rows of our processed data
print(processed_data[:10])


"""

Creating training and testing data sets

"""

# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])


# Split into features and targets
features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(targets[:10])
print(features[:10])
