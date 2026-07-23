"""
train.py
Loan Approval Prediction System - Model Training

What this does:
1. Loads the processed data from preprocessing.py
2. Splits into train/test (test set is held out and NEVER balanced/touched)
3. Balances the TRAINING set only, using SMOTE (fixes class imbalance)
4. Trains 3 models with hyperparameter tuning via GridSearchCV:
       - Decision Tree
       - Random Forest
       - XGBoost
5. Evaluates all 3 on the untouched test set
6. Saves the best model (by ROC-AUC) to models/best_model.pkl

Run from terminal (after preprocessing.py):
    python train.py

Requires: imbalanced-learn, xgboost (see requirements.txt)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

PROCESSED_PATH = DATA_DIR / "processed_data.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RESULTS_PATH = MODELS_DIR / "model_comparison.csv"

RANDOM_STATE = 42


def load_processed_data():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {PROCESSED_PATH}. Run preprocessing.py first."
        )
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*50}")
    print(f"{name} Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def train_decision_tree(X_train, y_train):
    print("\nTuning Decision Tree...")
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    print("\nTuning Random Forest...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_


def train_xgboost(X_train, y_train):
    print("\nTuning XGBoost...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    grid = GridSearchCV(
        XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
        param_grid, cv=5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_


def main():
    X, y = load_processed_data()
    print(f"Loaded processed data: {X.shape[0]} rows, {X.shape[1]} features")

    # Split BEFORE balancing - test set must reflect real-world distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Balance the training data only
    print("\nClass distribution before SMOTE:", dict(y_train.value_counts()))
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:", dict(pd.Series(y_train_bal).value_counts()))

    # Train all three models
    dt_model = train_decision_tree(X_train_bal, y_train_bal)
    rf_model = train_random_forest(X_train_bal, y_train_bal)
    xgb_model = train_xgboost(X_train_bal, y_train_bal)

    # Evaluate all three on the untouched test set
    results = []
    results.append(evaluate_model("Decision Tree", dt_model, X_test, y_test))
    results.append(evaluate_model("Random Forest", rf_model, X_test, y_test))
    results.append(evaluate_model("XGBoost", xgb_model, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\n{'='*50}")
    print("MODEL COMPARISON (sorted by ROC-AUC)")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))

    # Pick the best model by ROC-AUC
    best_name = results_df.iloc[0]["Model"]
    best_model = {"Decision Tree": dt_model, "Random Forest": rf_model, "XGBoost": xgb_model}[best_name]

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nBest model: {best_name} (ROC-AUC: {results_df.iloc[0]['ROC_AUC']:.4f})")
    print(f"Saved to {BEST_MODEL_PATH}")
    print("\nTraining complete. Next: run the Streamlit app -> streamlit run app.py")


if __name__ == "__main__":
    main()