# [1] Import libraries and setup logging
import pandas as pd
import numpy as np
import logging

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_fscore_support, accuracy_score,
                           roc_curve, auc)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Model persistence
import joblib
import lightgbm as lgb

# [2] Configure logging
logging.basicConfig(filename='model_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# [3] Feature preprocessing function
def analyze_correlation(X, threshold=0.9):
    """Remove highly correlated features"""
    corr_matrix = X.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    print(f"Highly correlated features to drop: {to_drop}")
    return X.drop(columns=to_drop, axis=1)

# [8] Visualization functions - Called during model evaluation
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

# [9] ROC curve visualization - Called during model evaluation
def plot_roc_curve(model, X_val, y_val, model_name):
    """Create ROC curve plot"""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

# [6] Model training functions - Called based on model type
def train_random_forest(model, X_train, y_train):
    """Train Random Forest model"""
    model.fit(X_train, y_train)
    return model

def train_xgboost(model, X_train, X_val, y_train, y_val):
    """Train XGBoost model with early stopping"""
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=10,
        verbose=False
    )
    return model

def train_lightgbm(model, X_train, X_val, y_train, y_val):
    """Train LightGBM model with early stopping"""
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric=['error', 'logloss'],
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )
    return model

# [7] Model evaluation function - Called after training each model
def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model performance and create visualizations"""
    y_pred = model.predict(X_val)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
    accuracy = accuracy_score(y_val, y_pred)

    logging.info(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    plot_confusion_matrix(y_val, y_pred, model_name)
    plot_roc_curve(model, X_val, y_val, model_name)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# [5] Main training and evaluation function
def train_and_evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Main function to train and evaluate each model"""
    if isinstance(model, RandomForestClassifier):
        trained_model = train_random_forest(model, X_train, y_train)
    elif isinstance(model, XGBClassifier):
        trained_model = train_xgboost(model, X_train, X_val, y_train, y_val)
    elif isinstance(model, LGBMClassifier):
        trained_model = train_lightgbm(model, X_train, X_val, y_train, y_val)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    metrics = evaluate_model(trained_model, X_val, y_val, model_name)
    return trained_model, metrics

# [10] Machine Learning Workflow
if __name__ == "__main__":
    # [10.1] Load data
    print("Loading data...")
    train_data = pd.read_parquet("Syn-training.parquet", engine='pyarrow')
    test_data = pd.read_parquet("syn-testing.parquet", engine='pyarrow')

    # [10.2] Extract features and encode target
    X = train_data.select_dtypes(include=['float64', 'int64']).copy()
    y = train_data['Label']
    le = LabelEncoder()
    y = le.fit_transform(y)

    # [10.3] Preprocess data
    X = analyze_correlation(X)  # First preprocessing step
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_balanced, y_train_balanced = X_train, y_train

    # [10.4] Define models with hyperparameters
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            class_weight='balanced',
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric=['error', 'logloss'],
            random_state=42,
            n_estimators=1000
        ),
        "LightGBM": LGBMClassifier(
            random_state=42,
            n_estimators=1000
        )
    }

    # [10.5] Train and evaluate all models
    models_metrics = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_model, metrics = train_and_evaluate_model(
            model, 
            X_train_balanced, 
            X_val, 
            y_train_balanced, 
            y_val, 
            name
        )
        models_metrics[name] = metrics
        trained_models[name] = trained_model

    # [10.6] Save all models
    for model_name, trained_model in trained_models.items():
        model_filename = f"model_{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(trained_model, model_filename)
        print(f"\nSaved {model_name} model to {model_filename}")
    
    # Still print the best model for reference
    best_model_name = max(models_metrics.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest performing model was: {best_model_name}")
