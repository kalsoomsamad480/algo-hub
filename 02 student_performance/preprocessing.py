"""
preprocessing.py
-----------------
Week 2 Project: Student Performance Prediction

What this script does (in plain steps):
1. Loads student-mat.csv (the raw dataset)
2. Creates a target column "pass" (1 = pass, 0 = fail) from the G3 grade
3. Encodes all categorical (text) columns into numbers
4. Selects the TOP 15 most useful features using SelectKBest (Feature Selection)
5. Scales numeric features
6. Splits data into train/test sets
7. Saves everything (processed data + encoders + scaler + feature list) to disk
   so training.py and app.py can reuse the exact same preprocessing.

Run this FIRST, before training.py.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# STEP 0: Setup paths
# ---------------------------------------------------------
DATA_PATH = "data/student-mat.csv"   # change this if your CSV is somewhere else
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# How many top features to keep (Feature Selection step)
NUM_FEATURES_TO_SELECT = 15

# ---------------------------------------------------------
# STEP 1: Load the dataset
# ---------------------------------------------------------
print("Step 1: Loading dataset...")

# IMPORTANT: this UCI dataset uses semicolons (;) as separators, not commas!
df = pd.read_csv(DATA_PATH, sep=";")

print(f"   Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"   Columns: {list(df.columns)}")

# ---------------------------------------------------------
# STEP 2: Create the classification target
# ---------------------------------------------------------
print("\nStep 2: Creating target column 'pass' from G3 (final grade)...")

# G3 is the final grade, out of 20.
# Rule: pass = 1 if G3 >= 10, else 0 (fail)
df["pass"] = (df["G3"] >= 10).astype(int)

print(f"   Pass count: {df['pass'].sum()} | Fail count: {(df['pass'] == 0).sum()}")

# We drop G3 itself so the model can't "cheat" by just reading the answer.
# We KEEP G1 and G2 (earlier period grades) as legitimate historical academic features.
df = df.drop(columns=["G3"])

# ---------------------------------------------------------
# STEP 3: Encode categorical columns
# ---------------------------------------------------------
print("\nStep 3: Encoding categorical (text) columns...")

categorical_cols = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

label_encoders = {}  # we save these so app.py can encode new user input the same way

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   Encoded '{col}': {list(le.classes_)} -> {list(range(len(le.classes_)))}")

# ---------------------------------------------------------
# STEP 4: Separate features (X) and target (y)
# ---------------------------------------------------------
X = df.drop(columns=["pass"])
y = df["pass"]

all_feature_names = list(X.columns)

# ---------------------------------------------------------
# STEP 5: Feature Selection (SelectKBest)
# ---------------------------------------------------------
print(f"\nStep 4: Selecting top {NUM_FEATURES_TO_SELECT} features using SelectKBest...")

selector = SelectKBest(score_func=f_classif, k=NUM_FEATURES_TO_SELECT)
X_selected = selector.fit_transform(X, y)

# Get the names of the features that were selected
selected_mask = selector.get_support()
selected_features = [f for f, keep in zip(all_feature_names, selected_mask) if keep]

print(f"   Selected features: {selected_features}")

X = pd.DataFrame(X_selected, columns=selected_features)

# ---------------------------------------------------------
# STEP 6: Train/test split
# ---------------------------------------------------------
print("\nStep 5: Splitting into train/test sets (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ---------------------------------------------------------
# STEP 7: Scale features
# ---------------------------------------------------------
print("\nStep 6: Scaling features with StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

# ---------------------------------------------------------
# STEP 8: Save everything for training.py and app.py
# ---------------------------------------------------------
print("\nStep 7: Saving processed data and preprocessing objects...")

X_train_scaled.to_csv(f"{MODELS_DIR}/X_train.csv", index=False)
X_test_scaled.to_csv(f"{MODELS_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{MODELS_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{MODELS_DIR}/y_test.csv", index=False)

joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
joblib.dump(label_encoders, f"{MODELS_DIR}/label_encoders.pkl")
joblib.dump(selected_features, f"{MODELS_DIR}/selected_features.pkl")
joblib.dump(categorical_cols, f"{MODELS_DIR}/categorical_cols.pkl")

print("\nPreprocessing complete! Files saved in the 'models/' folder:")
print("   - X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("   - scaler.pkl, label_encoders.pkl, selected_features.pkl, categorical_cols.pkl")
print("\nNext step: run training.py")