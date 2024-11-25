# Import required libraries
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load the pre-trained Random Forest model from disk
model = joblib.load("best_model_random_forest.joblib")


def preprocess_data(X):
    """
    Preprocess incoming data by handling missing values.
    Args:
        X (pd.DataFrame): Input features dataframe
    Returns:
        pd.DataFrame: Preprocessed dataframe with missing values filled
    """
    # Replace all missing values with 0
    X = X.fillna(0)
    return X


def predict_ddos(model, X):
    """
    Make DDoS attack predictions using the trained model.
    Args:
        model: Trained machine learning model
        X (pd.DataFrame): Preprocessed input features
    Returns:
        tuple: (predictions array, probability scores array)
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)  # Get probability scores for each class
    return y_pred, y_pred_proba


def analyze_and_classify(file_path, model, save_predictions=True, save_ddos_only=True):
    """
    Main function to analyze network traffic and detect DDoS attacks.
    Args:
        file_path (str): Path to the parquet file containing network traffic data
        model: Trained machine learning model
        save_predictions (bool): Whether to save all predictions to CSV
        save_ddos_only (bool): Whether to save only DDoS-flagged traffic to CSV
    """
    # Load and display initial data information
    print(f"Analyzing and classifying data from: {file_path}")
    data = pd.read_parquet(file_path, engine="pyarrow")
    
    # Print basic dataset information
    print(f"Dataset shape: {data.shape}")
    print("\nSample of first 5 rows:")
    print(data.head())
    
    # Display missing value statistics
    print("\nMissing values in dataset:")
    print(data.isnull().sum())
    
    # Select only numerical features for prediction
    X = data.select_dtypes(include=['float64', 'int64']).copy()
    
    # Preprocess the features
    X_preprocessed = preprocess_data(X)
    
    # Make predictions
    predictions, probabilities = predict_ddos(model, X_preprocessed)
    
    # Add prediction results to the original dataframe
    data['Predictions'] = predictions
    data['DDoS_Probability'] = probabilities[:, 1]  # Probability of being a DDoS attack
    
    # Save complete results if requested
    if save_predictions:
        output_file = file_path.replace(".parquet", "_predictions.csv")
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
    
    # Save only DDoS traffic if requested
    if save_ddos_only:
        ddos_data = data[data['Predictions'] == 1]
        ddos_output_file = file_path.replace(".parquet", "_ddos_only.csv")
        ddos_data.to_csv(ddos_output_file, index=False)
        print(f"DDoS-only data saved to: {ddos_output_file}")
    
    # Create visualization of results
    visualize_predictions(predictions)


def visualize_predictions(predictions):
    """
    Create a bar chart showing the distribution of normal vs DDoS traffic.
    Args:
        predictions (array): Array of model predictions (0 for normal, 1 for DDoS)
    """
    # Count occurrences of each class
    prediction_counts = pd.Series(predictions).value_counts()
    
    # Create and display bar chart
    prediction_counts.plot(kind='bar', color='skyblue')
    plt.title("DDoS Detection Results")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


# Execute if run as a script
if __name__ == "__main__":
    # Specify the path to the testing data
    test_file = "syn-testing.parquet"

    # Run the analysis
    analyze_and_classify(test_file, model)
