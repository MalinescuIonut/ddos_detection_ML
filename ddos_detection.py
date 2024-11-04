# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the training and testing data
syn_train_TCP = pd.read_parquet("Syn-training.parquet", engine='pyarrow')
syn_test_TCP = pd.read_parquet("syn-testing.parquet", engine='pyarrow')

# Define feature columns (X) and target column (y) - replace 'target_column_name' with your actual target column
X_train = syn_train_TCP.drop('target_column_name', axis=1)
y_train = syn_train_TCP['target_column_name']
X_test = syn_test_TCP.drop('target_column_name', axis=1)
y_test = syn_test_TCP['target_column_name']

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()