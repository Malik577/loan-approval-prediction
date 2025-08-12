"""Loan Approval Prediction Model.

This script:
1. Loads and preprocesses the loan dataset
2. Handles missing values and encodes categorical features
3. Trains a model to predict loan approval
4. Evaluates performance with focus on imbalanced metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath):
    """Load and clean the dataset."""
    # Read data efficiently
    df = pd.read_csv(filepath, engine='c')
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    """Preprocess the data including handling missing values and encoding."""
    # Identify numeric and categorical columns more efficiently
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features.remove('loan_id')
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    categorical_features.remove('loan_status')
    
    # Create preprocessing pipelines with optimized settings
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(copy=False))  # Avoid copying data
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        n_jobs=-1  # Use all CPU cores
    )
    
    # Prepare target
    le = LabelEncoder()
    y = le.fit_transform(df['loan_status'].values)  # Convert to numpy array for speed
    
    # Remove target and ID from features efficiently
    X = df.drop(['loan_status', 'loan_id'], axis=1)
    
    return X, y, preprocessor

def train_evaluate_model(X, y, preprocessor):
    """Train model and evaluate performance."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with optimized settings
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=200,  # Increase max iterations
            n_jobs=-1,  # Use all CPU cores
            solver='saga'  # Faster solver for large datasets
        ))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, X_test, y_test, y_pred, y_prob

def plot_evaluation_metrics(y_test, y_pred, y_prob):
    """Plot evaluation metrics including PR curve and confusion matrix."""
    # Calculate metrics once
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create single figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot PR curve
    ax1.plot(recall, precision, label=f'PR curve (AP = {pr_auc:.2f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the loan prediction pipeline."""
    # Load data
    print("Loading data...")
    df = load_data('loan_approval_dataset.csv')
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y, preprocessor = preprocess_data(df)
    
    # Train and evaluate model
    print("\nTraining model...")
    model, X_test, y_test, y_pred, y_prob = train_evaluate_model(X, y, preprocessor)
    
    # Print classification report
    print("\nModel Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot evaluation metrics
    print("\nPlotting evaluation metrics...")
    plot_evaluation_metrics(y_test, y_pred, y_prob)

if __name__ == "__main__":
    main() 