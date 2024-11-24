import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load the pre-trained model
model = joblib.load("best_model_random_forest.joblib")  # Replace with your model path


def preprocess_data(X):
    """
    Preprocess incoming data (handling missing values).
    """
    # Handle missing values
    X = X.fillna(0)  # Replace missing values with 0
    return X


def predict_ddos(model, X):
    """
    Use the trained model to predict DDoS attacks and probabilities.
    """
    y_pred = model.predict(X)  # Predict class labels
    y_pred_proba = model.predict_proba(X)  # Predict probabilities (if model supports it)
    return y_pred, y_pred_proba


def analyze_and_classify(file_path, model, save_predictions=True, save_ddos_only=True):
    """
    Analyze the testing file and classify DDoS traffic using the pre-trained model.
    """
    print(f"Analyzing and classifying data from: {file_path}")
    data = pd.read_parquet(file_path, engine="pyarrow")
    
    # Basic Analysis
    print(f"Dataset shape: {data.shape}")
    print("\nSample of first 5 rows:")
    print(data.head())
    
    # Check for missing values
    print("\nMissing values in dataset:")
    print(data.isnull().sum())
    
    # Feature selection (numerical columns only)
    X = data.select_dtypes(include=['float64', 'int64']).copy()
    
    # Preprocess data
    X_preprocessed = preprocess_data(X)
    
    # Predict using the pre-trained model
    predictions, probabilities = predict_ddos(model, X_preprocessed)
    
    # Add predictions and probabilities to the dataframe
    data['Predictions'] = predictions
    data['DDoS_Probability'] = probabilities[:, 1]  # Assuming column 1 represents the DDoS class
    
    # Save predictions if required
    if save_predictions:
        output_file = file_path.replace(".parquet", "_predictions.csv")
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
    
    # Save only DDoS-labeled rows if required
    if save_ddos_only:
        ddos_data = data[data['Predictions'] == 1]  # Filter rows where DDoS is detected
        ddos_output_file = file_path.replace(".parquet", "_ddos_only.csv")
        ddos_data.to_csv(ddos_output_file, index=False)
        print(f"DDoS-only data saved to: {ddos_output_file}")
    
    # Visualization
    visualize_predictions(predictions)


def visualize_predictions(predictions):
    """
    Generate a bar chart of DDoS predictions.
    """
    prediction_counts = pd.Series(predictions).value_counts()
    prediction_counts.plot(kind='bar', color='skyblue')
    plt.title("DDoS Detection Results")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    # Path to your testing parquet file
    test_file = "syn-testing.parquet"

    # Analyze and classify the testing dataset
    analyze_and_classify(test_file, model)
