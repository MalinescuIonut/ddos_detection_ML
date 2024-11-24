import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
syn_train_TCP = pd.read_parquet("Syn-training.parquet", engine='pyarrow')
syn_test_TCP = pd.read_parquet("syn-testing.parquet", engine='pyarrow')

# 1. Basic Analysis
print("\n=== Training Dataset Analysis ===")
print(f"Training set shape: {syn_train_TCP.shape}")
print("\nSample of first 5 rows:")
print(syn_train_TCP.head())
print("\nData types of columns:")
print(syn_train_TCP.dtypes)
print("\nBasic statistics:")
print(syn_train_TCP.describe())

# 2. Check for missing values
print("\nMissing values in training set:")
print(syn_train_TCP.isnull().sum())

# 3. If there's a target/label column for DDoS attacks, show class distribution
# Replace 'label' with your actual target column name if different
if 'label' in syn_train_TCP.columns:
    print("\nClass distribution in training set:")
    print(syn_train_TCP['label'].value_counts(normalize=True) * 100)

# Save the output to a text file for easier review
with open('data_analysis_report.txt', 'w') as f:
    f.write("=== Dataset Analysis Report ===\n")
    f.write(f"\nTraining set shape: {syn_train_TCP.shape}\n")
    f.write("\nColumns in the dataset:\n")
    f.write(str(syn_train_TCP.columns.tolist()))
    f.write("\n\nData types:\n")
    f.write(str(syn_train_TCP.dtypes))