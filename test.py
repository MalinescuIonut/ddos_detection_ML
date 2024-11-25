import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logging.basicConfig(filename='rf_model_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_attack_distribution(y):
    """Analyze and visualize the distribution of attack types"""
    attack_counts = y.value_counts()
    
    # Print attack distribution
    print("\nAttack Type Distribution:")
    for attack_type, count in attack_counts.items():
        percentage = (count / len(y)) * 100
        print(f"{attack_type}: {count} samples ({percentage:.2f}%)")
    
    # Plot attack distribution
    plt.figure(figsize=(12, 6))
    attack_counts.plot(kind='bar')
    plt.title('Distribution of DDoS Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('attack_distribution.png')
    plt.close()
    
    return attack_counts

def train_random_forest(model, X_train, y_train):
    """Train Random Forest model"""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val, attack_types, X_train):
    """Evaluate model and generate metrics"""
    y_pred = model.predict(X_val)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
    
    # Log overall metrics
    logging.info(f"Overall Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"\nOverall Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report by Attack Type:")
    print(classification_report(y_val, y_pred, target_names=attack_types))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, y_pred, attack_types)
    
    # Plot feature importance
    plot_feature_importance(model, X_train.columns)
    
    # Analyze misclassifications
    analyze_misclassifications(y_val, y_pred, attack_types)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def analyze_misclassifications(y_true, y_pred, attack_types):
    """Analyze and report misclassified attacks"""
    print("\nMisclassification Analysis:")
    
    for true_type in attack_types:
        mask_true = (y_true == true_type)
        if not any(mask_true):
            continue
            
        wrong_predictions = y_pred[mask_true] != y_true[mask_true]
        if not any(wrong_predictions):
            continue
            
        misclassified = pd.Series(y_pred[mask_true][wrong_predictions])
        print(f"\nMisclassifications for {true_type}:")
        print(misclassified.value_counts())

def plot_confusion_matrix(y_true, y_pred, attack_types):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot raw counts
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=attack_types, yticklabels=attack_types)
    plt.title('Confusion Matrix\nCounts')
    plt.ylabel('True Attack Type')
    plt.xlabel('Predicted Attack Type')
    plt.tight_layout()
    plt.savefig('confusion_matrix_counts.png')
    plt.close()
    
    # Calculate and plot percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=attack_types, yticklabels=attack_types)
    plt.title('Confusion Matrix\nPercentages')
    plt.ylabel('True Attack Type')
    plt.xlabel('Predicted Attack Type')
    plt.tight_layout()
    plt.savefig('confusion_matrix_percent.png')
    plt.close()

def plot_feature_importance(model, feature_names, top_n=None):
    """Plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Determine number of features to plot
    n_features = len(feature_names)
    if top_n is None or top_n > n_features:
        top_n = n_features
    
    # Plot features
    plt.figure(figsize=(12, 6))
    plt.title(f'Top {top_n} Most Important Features')
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), 
               [feature_names[i] for i in indices[:top_n]], 
               rotation=45, 
               ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Print feature importance rankings
    print("\nFeature Importance Rankings:")
    for i in range(top_n):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def save_analysis_to_file(train_data, X, y, filename="dataset_analysis.txt"):
    """Save detailed analysis to a text file"""
    with open(filename, 'w') as f:
        # Dataset Information
        f.write("Dataset Information:\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nTraining Dataset Shape: {train_data.shape}\n")
        f.write(f"\nColumns in dataset: {train_data.columns.tolist()}\n")
        
        f.write("\nSample of the data (first 5 rows):\n")
        f.write(train_data.head().to_string())
        
        # Label Analysis
        f.write("\n\nLabel Analysis:\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nUnique values in Label column: {train_data['Label'].unique()}\n")
        
        f.write("\nDetailed label counts:\n")
        label_counts = train_data['Label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(train_data)) * 100
            f.write(f"\nLabel: {label}\n")
            f.write(f"Count: {count}\n")
            f.write(f"Percentage: {percentage:.2f}%\n")
            f.write("-" * 30 + "\n")
        
        # Feature Information
        f.write("\nFeature Information:\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nNumber of features: {X.shape[1]}\n")
        f.write("\nFeature names:\n")
        for i, feature in enumerate(X.columns, 1):
            f.write(f"{i}. {feature}\n")
        
        # Attack Type Analysis
        f.write("\nAttack Type Analysis:\n")
        f.write("=" * 50 + "\n")
        attack_types = y.unique()
        f.write(f"\nNumber of unique attack types: {len(attack_types)}\n")
        f.write(f"Attack types detected: {attack_types}\n")
        
        f.write("\nAttack Type Distribution:\n")
        attack_counts = y.value_counts()
        for attack_type, count in attack_counts.items():
            percentage = (count / len(y)) * 100
            f.write(f"{attack_type}: {count} samples ({percentage:.2f}%)\n")

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_data = pd.read_parquet("Syn-training.parquet", engine='pyarrow')
    test_data = pd.read_parquet("syn-testing.parquet", engine='pyarrow')

    # Prepare features and target
    X = train_data.select_dtypes(include=['float64', 'int64']).copy()
    y = train_data['Label']
    
    # Save analysis to file
    save_analysis_to_file(train_data, X, y)
    print("\nDetailed analysis has been saved to 'dataset_analysis.txt'")
    
    # Rest of your existing code...