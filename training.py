"""
train.py
--------
Customer Churn Prediction System — Model Training Module

Trains and compares three classifiers:
- Logistic Regression
- Random Forest
- XGBoost

Applies SMOTE on the TRAINING SET ONLY (after preprocessing, before fitting)
to handle class imbalance without leaking synthetic samples into the test set.

Saves each trained model to models/ as .pkl, and prints/saves a comparison
of evaluation metrics.

Author: Kalsoom
"""

import os
import joblib
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocessing import preprocess_data


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_PATH = "data/telco_churn.csv"   # update to match your actual file
MODELS_DIR = "models"
RANDOM_STATE = 42


# ------------------------------------------------------------------
# 1. APPLY SMOTE (TRAIN SET ONLY)
# ------------------------------------------------------------------
def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Balance the training set using SMOTE.
    Must only be applied AFTER the train/test split and AFTER preprocessing,
    and NEVER on the test set (that would leak synthetic info into evaluation).
    """
    print(f"[INFO] Before SMOTE -> class distribution: {np.bincount(y_train)}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"[INFO] After SMOTE  -> class distribution: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


# ------------------------------------------------------------------
# 2. DEFINE MODELS
# ------------------------------------------------------------------
def get_models(random_state: int = 42) -> dict:
    """Return a dictionary of model name -> initialized (untrained) model."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


# ------------------------------------------------------------------
# 3. TRAIN + EVALUATE A SINGLE MODEL
# ------------------------------------------------------------------
def train_and_evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """Fit a model, evaluate it on the test set, and return metrics as a dict."""
    print(f"\n[INFO] Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    print(f"[RESULT] {name}")
    print(f"  Accuracy : {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")
    print(f"  F1 Score : {metrics['F1 Score']:.4f}")
    print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, metrics


# ------------------------------------------------------------------
# 4. SAVE MODEL
# ------------------------------------------------------------------
def save_model(model, name: str, save_dir: str = "models"):
    """Save trained model to disk using joblib."""
    os.makedirs(save_dir, exist_ok=True)
    filename_map = {
        "Logistic Regression": "logistic.pkl",
        "Random Forest": "randomforest.pkl",
        "XGBoost": "xgboost.pkl"
    }
    filename = filename_map.get(name, f"{name.lower().replace(' ', '_')}.pkl")
    path = os.path.join(save_dir, filename)
    joblib.dump(model, path)
    print(f"[INFO] Saved {name} to: {path}")


# ------------------------------------------------------------------
# 5. MAIN TRAINING PIPELINE
# ------------------------------------------------------------------
def main():
    # Step 1: preprocess data (loads, cleans, splits, encodes, scales)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        filepath=DATA_PATH,
        target_col="Churn",
        test_size=0.2,
        random_state=RANDOM_STATE,
        save_dir=MODELS_DIR
    )

    # Step 2: balance training set only
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, RANDOM_STATE)

    # Step 3: train + evaluate each model
    models = get_models(RANDOM_STATE)
    results = []
    trained_models = {}

    for name, model in models.items():
        trained_model, metrics = train_and_evaluate(
            name, model,
            X_train_balanced, y_train_balanced,
            X_test, y_test
        )
        trained_models[name] = trained_model
        results.append(metrics)
        save_model(trained_model, name, MODELS_DIR)

    # Step 4: comparison table
    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    results_df.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Comparison table saved to: {results_path}")

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n[DONE] Best performing model: {best_model_name}")

    return trained_models, results_df


if __name__ == "__main__":
    main()