import pandas as pd
import matplotlib.pyplot as plt

# Load the training and testing data
syn_train_TCP = pd.read_parquet("Syn-training.parquet", engine='pyarrow')
syn_test_TCP = pd.read_parquet("syn-testing.parquet", engine='pyarrow')

# Display the first few rows of the training dataset
print("Training Data Preview:")
print(syn_train_TCP.head())

# Display the first few rows of the testing dataset
print("\nTesting Data Preview:")
print(syn_test_TCP.head())

# Get information about the training dataset
print("\nTraining Data Info:")
syn_train_TCP.info()

# Get information about the testing dataset
print("\nTesting Data Info:")
syn_test_TCP.info()

# Get summary statistics for the training dataset
print("\nTraining Data Description:")
print(syn_train_TCP.describe())

# Get summary statistics for the testing dataset
print("\nTesting Data Description:")
print(syn_test_TCP.describe())

# Plot the number of unique values for each column in the training dataset
plt.figure(figsize=(10, 5))
syn_train_TCP.nunique().plot(kind='bar')
plt.title("Number of Unique Values per Column in Training Data")
plt.xlabel("Columns")
plt.ylabel("Unique Values")
plt.xticks(rotation=45)
plt.show()
