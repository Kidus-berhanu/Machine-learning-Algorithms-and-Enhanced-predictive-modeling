#Shuffle the data randomly. This can be done using the shuffle function from the random module.

#Split the data into three parts. You can use the floor function from the math module to compute the sizes of the training and validation sets, 
#and the remainder will be the size of the test set.
import math

# Compute the sizes of the three datasets
n_samples = len(data)
n_train = math.floor(n_samples * 0.7)
n_valid = math.floor(n_samples * 0.15)
n_test = n_samples - n_train - n_valid

# Split the data into the three datasets
train_data = data[:n_train]
valid_data = data[n_train:n_train+n_valid]
test_data = data[n_train+n_valid:]

#Check the target rate in each dataset. You can use a Counter from the collections module to count the number of occurrences of each target value in each dataset, 
#and then divide by the size of the dataset to compute the target rate.
from collections import Counter

# Compute the target rate in the training set
train_targets = [example[1] for example in train_data]
train_counts = Counter(train_targets)
train_rates = {k: v / len(train_data) for k, v in train_counts.items()}

# Compute the target rate in the validation set
valid_targets = [example[1] for example in valid_data]
valid_counts = Counter(valid_targets)
valid_rates = {k: v / len(valid_data) for k, v in valid_counts.items()}

# Compute the target rate in the test set
test_targets = [example[1] for example in test_data]
test_counts = Counter(test_targets)
test_rates = {k: v / len(test_data) for k, v in test_counts.items()}
