"""
tune_models.py
--------------
Customer Churn Prediction System — Hyperparameter Tuning Module

Improves on train.py by:
1. Searching for better hyperparameters via RandomizedSearchCV (cross-validated)
2. Scoring on ROC-AUC during tuning (not plain accuracy) so tuning doesn't
   just reward the majority class
3. Re-evaluating tuned models on the untouched test set

Run this AFTER preprocessing.py has already produced models/preprocessor.pkl.

Author: Kalsoom
"""

import os
import joblib
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from preprocessing import preprocess_data


DATA_PATH = "data/telco_churn.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42


# ------------------------------------------------------------------
# PARAMETER GRIDS — the ranges RandomizedSearchCV will search over
# ------------------------------------------------------------------
PARAM_GRIDS = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "params": {
            "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [4, 6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5]
        }
    }
}


# ------------------------------------------------------------------
# TUNE A SINGLE MODEL
# ------------------------------------------------------------------
def tune_model(name, config, X_train, y_train, n_iter=30, cv=5):
    """
    Run RandomizedSearchCV for one model, scored on ROC-AUC via
    stratified cross-validation (keeps class ratio consistent per fold).
    """
    print(f"\n[INFO] Tuning {name} ({n_iter} candidate combinations, {cv}-fold CV)...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"[RESULT] Best CV ROC-AUC for {name}: {search.best_score_:.4f}")
    print(f"[RESULT] Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


# ------------------------------------------------------------------
# EVALUATE ON TEST SET
# ------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
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

    print(f"\n[TEST RESULTS] {name}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        filepath=DATA_PATH,
        target_col="Churn",
        test_size=0.2,
        random_state=RANDOM_STATE,
        save_dir=MODELS_DIR
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    results = []
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, config in PARAM_GRIDS.items():
        best_model, best_params, best_cv_score = tune_model(
            name, config, X_train_bal, y_train_bal, n_iter=30, cv=5
        )

        metrics = evaluate(name, best_model, X_test, y_test)
        results.append(metrics)

        filename_map = {
            "Logistic Regression": "logistic_tuned.pkl",
            "Random Forest": "randomforest_tuned.pkl",
            "XGBoost": "xgboost_tuned.pkl"
        }
        path = os.path.join(MODELS_DIR, filename_map[name])
        joblib.dump(best_model, path)
        print(f"[INFO] Saved tuned {name} to: {path}")

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    results_df.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 60)
    print("TUNED MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(MODELS_DIR, "tuned_model_comparison.csv"), index=False)
    print(f"\n[DONE] Best tuned model: {results_df.iloc[0]['Model']}")


if __name__ == "__main__":
    main()