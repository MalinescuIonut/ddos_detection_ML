import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
from shap import TreeExplainer
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import lightgbm as lgb

logging.basicConfig(filename='model_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Feature selection using correlation analysis
def analyze_correlation(X, threshold=0.9):
    corr_matrix = X.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    print(f"Highly correlated features to drop: {to_drop}")
    return X.drop(columns=to_drop, axis=1)

# Feature selection using RFE
def select_features(X, y, model, num_features):
    selector = RFE(model, n_features_to_select=num_features, step=1)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    print(f"Selected features: {selected_features}")
    return X[selected_features]

# Balance data using SMOTE
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print("Data balanced with SMOTE.")
    return X_balanced, y_balanced

# Scale features for models requiring scaling
def scale_features(X_train, X_val, scale_models=False):
    if scale_models:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        print("Data scaled for applicable models.")
    else:
        print("No scaling applied for tree-based models.")
    return X_train, X_val

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

def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model and generate metrics"""
    y_pred = model.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
    accuracy = accuracy_score(y_val, y_pred)

    # Log metrics
    logging.info(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Generate all plots
    plot_confusion_matrix(y_val, y_pred, model_name)
    plot_roc_curve(model, X_val, y_val, model_name)
    plot_precision_recall_curve(model, X_val, y_val, model_name)
    plot_feature_importance(model, X_train, model_name)
    
    if isinstance(model, (XGBClassifier, LGBMClassifier)):
        plot_learning_curves(model, X_train, X_val, y_train, y_val, model_name)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def train_and_evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Main function to train and evaluate models"""
    # Train model based on type
    if isinstance(model, RandomForestClassifier):
        trained_model = train_random_forest(model, X_train, y_train)
    elif isinstance(model, XGBClassifier):
        trained_model = train_xgboost(model, X_train, X_val, y_train, y_val)
    elif isinstance(model, LGBMClassifier):
        trained_model = train_lightgbm(model, X_train, X_val, y_train, y_val)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Evaluate model
    metrics = evaluate_model(trained_model, X_val, y_val, model_name)
    
    return trained_model, metrics

# Train ensemble model
def train_ensemble(X_train, X_val, y_train, y_val):
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5, class_weight='balanced', random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    lgbm_model = LGBMClassifier(random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft'
    )

    model, metrics = train_and_evaluate_model(ensemble, X_train, X_val, y_train, y_val, "Ensemble")
    return model, metrics

# Hyperparameter optimization with Optuna
def optimize_with_optuna(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
        rf = RandomForestClassifier(random_state=42, **params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        return f1_score(y_val, y_pred, average='weighted')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

# Explain model
def explain_model(model, X_val):
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    # Plot SHAP summary

def plot_learning_curves(model, X_train, X_val, y_train, y_val, model_name):
    """Plot learning curves for models that support it (XGBoost and LightGBM)"""
    if hasattr(model, 'evals_result'):
        results = model.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot error
        ax1.plot(x_axis, results['validation_0']['error'], label='Train')
        ax1.plot(x_axis, results['validation_1']['error'], label='Validation')
        ax1.set_title(f'{model_name} Error')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Error')
        ax1.legend()
        
        # Plot log loss
        ax2.plot(x_axis, results['validation_0']['logloss'], label='Train')
        ax2.plot(x_axis, results['validation_1']['logloss'], label='Validation')
        ax2.set_title(f'{model_name} Log Loss')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Log Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'learning_curves_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def plot_feature_importance(model, X, model_name):
    """Plot feature importance for tree-based models"""
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f'Feature Importances ({model_name})')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def plot_roc_curve(model, X_val, y_val, model_name):
    """Plot ROC curve"""
    y_pred_proba = model.predict_proba(X_val)
    
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_val))):
        fpr, tpr, _ = roc_curve(y_val == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_precision_recall_curve(model, X_val, y_val, model_name):
    """Plot Precision-Recall curve"""
    y_pred_proba = model.predict_proba(X_val)
    
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_val))):
        precision, recall, _ = precision_recall_curve(y_val == i, y_pred_proba[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'precision_recall_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}\nCounts')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_counts_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Plot percentage version
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}\nPercentages')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_percent_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

# Add a function to create a summary report
def create_summary_report(models_metrics):
    """Create a summary plot comparing all models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(models_metrics.keys())
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        # Extract metric values in the correct order
        metric_values = [models_metrics[model_name][metric] for metric in metrics]
        plt.bar(x + i * width, metric_values, width, label=model_name)
    
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width * (len(model_names) - 1) / 2, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_data = pd.read_parquet("Syn-training.parquet", engine='pyarrow')
    test_data = pd.read_parquet("syn-testing.parquet", engine='pyarrow')

    # Prepare features and target
    X = train_data.select_dtypes(include=['float64', 'int64']).copy()
    y = train_data['Label']
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Correlation analysis
    X = analyze_correlation(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balance data
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    # Initialize models with their parameters
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

    # Train and evaluate all models
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

    # Create summary comparison
    create_summary_report(models_metrics)

    # Save best model (you might want to select based on specific metric)
    best_model_name = max(models_metrics.items(), key=lambda x: x[1]['f1'])[0]
    joblib.dump(trained_models[best_model_name], f"best_model_{best_model_name.lower().replace(' ', '_')}.joblib")
    print(f"\nBest model ({best_model_name}) saved!")
