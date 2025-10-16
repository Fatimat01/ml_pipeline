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


# load the data
print("Loading data...")
data = pd.read_csv("data/census.csv")


# k-fold cross validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


precision_scores = []
recall_scores = []
f1_scores = []

best_model = None
best_f1 = -np.inf  # Initialize lowest possible score
best_encoder = None
best_lb = None
fold = 1

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

for train_index, test_index in kf.split(data):
    print(f"\n--- k = {fold} ---")

    train = data.iloc[train_index]
    test = data.iloc[test_index]

    # Process data for this fold
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test,
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
    print(f"Best model saved with F1 = {best_f1:.3f}")

