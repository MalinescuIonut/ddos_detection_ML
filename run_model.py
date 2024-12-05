import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Load the pre-trained Random Forest model from disk
model = joblib.load("model_xgboost.joblib")


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


def analyze_and_compare(file_path, auto_label_file, model):
    """
    Analyze the .parquet file, make predictions, and compare with auto-labeled dataset.
    Args:
        file_path (str): Path to the .parquet file containing network traffic data.
        auto_label_file (str): Path to the auto-labeled CSV file for comparison.
        model: Trained machine learning model.
    """
    # Load and display initial data information
    print(f"Analyzing data from: {file_path}")
    data = pd.read_parquet(file_path, engine="pyarrow")
    
    # Load the auto-labeled dataset
    auto_labeled_data = pd.read_csv(auto_label_file)
    
    # Print basic dataset information
    print(f"Dataset shape: {data.shape}")
    print("\nSample of first 5 rows:")
    print(data.head())
    
    # Preprocess features
    X = data.select_dtypes(include=['float64', 'int64']).copy()
    X_preprocessed = preprocess_data(X)
    
    # Make predictions
    predictions, probabilities = predict_ddos(model, X_preprocessed)
    
    # Add prediction results to the original dataframe
    data['Predictions'] = predictions
    data['DDoS_Probability'] = probabilities[:, 1]
    
    # Map labels for auto-labeled dataset to binary format
    auto_labeled_data['Auto_Label'] = auto_labeled_data['Auto_Label'].map({'DDoS': 1, 'Normal': 0})
    
    # Compare model predictions to auto-labeled dataset
    if 'Auto_Label' in auto_labeled_data:
        y_true = auto_labeled_data['Auto_Label']
        y_pred = data['Predictions']
        
        # Evaluate predictions
        evaluate_predictions(y_true, y_pred)
    
    # Save predictions
    output_file = file_path.replace(".parquet", "_predictions.csv")
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")


def evaluate_predictions(y_true, y_pred):
    """
    Calculate and print various performance metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\nModel Performance on Testing Data:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))


# Execute if run as a script
if __name__ == "__main__":
    # Specify the paths to the testing data and auto-labeled dataset
    test_file = "syn-testing.parquet"  # Replace with your .parquet file path
    auto_label_file = "labeled_dataset.csv"  # Replace with your auto-labeled CSV file path

    # Run the analysis and comparison
    analyze_and_compare(test_file, auto_label_file, model)
