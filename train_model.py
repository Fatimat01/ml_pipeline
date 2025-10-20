# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
# Add the necessary imports for the starter code.
import pandas as pd
import os
import joblib
from ml.model import train_model, compute_model_metrics, inference
# Add code to load in the data.
from ml.data import process_data    


##################################################
# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def train_and_evaluate_fold(train_data, test_data):
    """
    Train and evaluate model on a single fold.

    Inputs
    ------
    train_data : pd.DataFrame
        Training data for this fold
    test_data : pd.DataFrame
        Test data for this fold
        
    Returns
    -------
    model : RandomForestClassifier
        Trained model
    encoder : OneHotEncoder
        Fitted encoder
    lb : LabelBinarizer
        Fitted label binarizer
    precision : float
        Precision score
    recall : float
        Recall score
    f1 : float
        F1 score
    """
    # Process data for this fold
    X_train, y_train, encoder, lb = process_data(
        train_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Process the test data with the process_data function
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train and evaluate
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    
    return model, encoder, lb, precision, recall, f1


def perform_cross_validation(data, k=5):
    """
    Perform k-fold cross-validation and return best model.
    
    Inputs
    ------
    data : pd.DataFrame
        Full dataset
    k : int
        Number of folds for cross-validation
        
    Returns
    -------
    best_model : RandomForestClassifier
        Best performing model
    best_encoder : OneHotEncoder
        Encoder from best model
    best_lb : LabelBinarizer
        Label binarizer from best model
    precision_scores : list
        Precision scores for each fold
    recall_scores : list
        Recall scores for each fold
    f1_scores : list
        F1 scores for each fold
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    best_model = None
    best_f1 = -np.inf  # Initialize lowest possible score
    best_encoder = None
    best_lb = None
    fold = 1
    
    for train_index, test_index in kf.split(data):
        print(f"\n--- k = {fold} ---")
        
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        
        model, encoder, lb, precision, recall, f1 = train_and_evaluate_fold(train, test)
        
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Store metrics
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        # Track the best model
        if f1 > best_f1:
            print(f"New best model found with F1 = {f1:.3f}")
            best_f1 = f1
            best_model = model
            best_encoder = encoder
            best_lb = lb
        
        fold += 1
    
    return best_model, best_encoder, best_lb, precision_scores, recall_scores, best_f1


# Load the data
print("Loading data...")
data = pd.read_csv("data/census.csv")

# Perform k-fold cross validation
k = 5
best_model, best_encoder, best_lb, precision_scores, recall_scores, f1_scores = perform_cross_validation(
    data, k=k
)

# Save per-fold metrics
metrics_df = pd.DataFrame({
    "Fold": range(1, k + 1),
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1": f1_scores
})
metrics_df.to_csv("model/kfold_metrics.csv", index=False)

print("Per-fold metrics saved to model/kfold_metrics.csv")

# Print average metrics
print("\n--- Cross-Validation Results ---")
print(f"Average Precision: {np.mean(precision_scores):.3f}")
print(f"Average Recall: {np.mean(recall_scores):.3f}")
print(f"Average F1: {np.mean(f1_scores):.3f}")

# Save the best model, encoder, and label binarizer.
if best_model is not None:
    MODEL_DIR = "model"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(best_encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(best_lb, os.path.join(MODEL_DIR, "label_binarizer.pkl"))
    print(f"Best model saved with F1 = {f1_scores:.3f}")


def compute_model_metrics_on_slices(data, model, encoder, lb, feature_name):
    """
    Compute model metrics on slices of data for a given categorical feature.
    
    Inputs
    ------
    data : pd.DataFrame
        The full dataset
    model : RandomForestClassifier
        Trained model
    encoder : OneHotEncoder
        Fitted encoder
    lb : LabelBinarizer
        Fitted label binarizer
    categorical_features : list
        List of categorical feature names
    feature_name : str
        The feature to slice on
        
    Returns
    -------
    results : dict
        Dictionary with feature values as keys and metrics as values
    """
    results = {}
    
    for value in data[feature_name].unique():
        # Filter data for this slice
        slice_data = data[data[feature_name] == value]
        
        # Process the slice
        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )
        
        # Get predictions and compute metrics
        preds = inference(model, X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, preds)
        
        results[value] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': len(slice_data)
        }
    
    return results


# Compute performance on slices for all categorical features
print("\n--- Computing Model Performance on Slices ---")
with open("model/slice_output.txt", "w", encoding="utf-8") as f:
    for feature in cat_features:
        f.write(f"\n{'='*80}\n")
        f.write(f"Feature: {feature}\n")
        f.write(f"{'='*80}\n")
        
        slice_results = compute_model_metrics_on_slices(
            data, best_model, best_encoder, best_lb, feature
        )
        
        for value, metrics in slice_results.items():
            f.write(f"\n{feature} = {value}\n")
            f.write(f"  Count: {metrics['count']}\n")
            f.write(f"  Precision: {metrics['precision']:.3f}\n")
            f.write(f"  Recall: {metrics['recall']:.3f}\n")
            f.write(f"  F1 Score: {metrics['f1']:.3f}\n")
        
        print(f"Computed metrics for {feature} with {len(slice_results)} unique values")

print("\nSlice metrics saved to model/slice_output.txt")

