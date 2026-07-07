"""
training.py
------------
Week 2 Project: Student Performance Prediction

What this script does (in plain steps):
1. Loads the processed data saved by preprocessing.py
2. Trains a Logistic Regression model
3. Trains a Random Forest model
4. Evaluates BOTH models: Accuracy, Confusion Matrix, Classification Report
5. Plots and saves a Feature Importance chart (from Random Forest)
6. Saves both trained models to disk so app.py can load and use them

Run this AFTER preprocessing.py.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Load processed data (created by preprocessing.py)
# ---------------------------------------------------------
print("Step 1: Loading processed data...")

X_train = pd.read_csv(f"{MODELS_DIR}/X_train.csv")
X_test = pd.read_csv(f"{MODELS_DIR}/X_test.csv")
y_train = pd.read_csv(f"{MODELS_DIR}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{MODELS_DIR}/y_test.csv").values.ravel()

selected_features = joblib.load(f"{MODELS_DIR}/selected_features.pkl")

print(f"   Loaded {X_train.shape[0]} training rows, {X_test.shape[0]} test rows.")
print(f"   Using {len(selected_features)} features: {selected_features}")

# ===========================================================
# MODEL 1: LOGISTIC REGRESSION
# ===========================================================
print("\nStep 2: Training Logistic Regression...")

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

print(f"   Logistic Regression Accuracy: {acc_lr:.4f}")
print("\n   Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr, target_names=["Fail", "Pass"]))

# ===========================================================
# MODEL 2: RANDOM FOREST
# ===========================================================
print("\nStep 3: Training Random Forest...")

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"   Random Forest Accuracy: {acc_rf:.4f}")
print("\n   Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=["Fail", "Pass"]))

# ---------------------------------------------------------
# STEP 4: Compare models & pick the best one
# ---------------------------------------------------------
print("\nStep 4: Comparing models...")
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [acc_lr, acc_rf]
})
print(comparison.to_string(index=False))
comparison.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)

best_model_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
print(f"\n   Best performing model: {best_model_name}")

# ---------------------------------------------------------
# STEP 5: Confusion matrices (saved as an image)
# ---------------------------------------------------------
print("\nStep 5: Plotting confusion matrices...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=axes[0])
axes[0].set_title(f"Logistic Regression (Acc: {acc_lr:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=axes[1])
axes[1].set_title(f"Random Forest (Acc: {acc_rf:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/confusion_matrices.png", dpi=150)
plt.close()
print(f"   Saved to {MODELS_DIR}/confusion_matrices.png")

# ---------------------------------------------------------
# STEP 6: Feature Importance (from Random Forest)
# ---------------------------------------------------------
print("\nStep 6: Plotting feature importance (Random Forest)...")

importances = rf_clf.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df.to_string(index=False))
feature_importance_df.to_csv(f"{MODELS_DIR}/feature_importance.csv", index=False)

plt.figure(figsize=(9, 6))
sns.barplot(data=feature_importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/feature_importance.png", dpi=150)
plt.close()
print(f"   Saved to {MODELS_DIR}/feature_importance.png")

# ---------------------------------------------------------
# STEP 7: Save trained models
# ---------------------------------------------------------
print("\nStep 7: Saving trained models...")

joblib.dump(log_reg, f"{MODELS_DIR}/logistic_regression.pkl")
joblib.dump(rf_clf, f"{MODELS_DIR}/random_forest.pkl")

print("\nTraining complete! Saved in 'models/':")
print("   - logistic_regression.pkl")
print("   - random_forest.pkl")
print("   - model_comparison.csv, feature_importance.csv")
print("   - confusion_matrices.png, feature_importance.png")
print("\nNext step: run 'streamlit run app.py' to launch the web app.")