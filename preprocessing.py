"""
data_preprocessing.py
----------------------
Customer Churn Prediction System — Data Preprocessing Module

Handles: loading, cleaning, feature engineering, encoding, scaling,
train/test splitting, and saving the fitted preprocessor for reuse
at inference time (predict.py / app.py).

Dataset assumed: Telco Customer Churn (IBM/Kaggle)
Target column: 'Churn' (Yes/No)

Author: Kalsoom
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw churn dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    return df


# ------------------------------------------------------------------
# 2. CLEAN DATA
# ------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning steps specific to the Telco churn dataset:
    - Drop customerID (identifier, not predictive)
    - Fix TotalCharges (loaded as object due to blank strings)
    - Drop rows where TotalCharges is missing (usually tenure == 0)
    """
    df = df.copy()

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        missing_count = df["TotalCharges"].isna().sum()
        if missing_count > 0:
            print(f"[INFO] Dropping {missing_count} rows with missing TotalCharges")
            df.dropna(subset=["TotalCharges"], inplace=True)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


# ------------------------------------------------------------------
# 3. ENCODE TARGET
# ------------------------------------------------------------------
def encode_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Encode target variable Yes/No -> 1/0."""
    df = df.copy()
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    return df


# ------------------------------------------------------------------
# 4. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------------
def get_feature_types(df: pd.DataFrame, target_col: str = "Churn"):
    """Split columns into numerical and categorical feature lists."""
    features = df.drop(columns=[target_col])

    numerical_cols = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object"]).columns.tolist()

    print(f"[INFO] Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"[INFO] Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    return numerical_cols, categorical_cols


# ------------------------------------------------------------------
# 5. BUILD PREPROCESSING PIPELINE
# ------------------------------------------------------------------
def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes and scales numerical features
    - Imputes and one-hot encodes categorical features
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


# ------------------------------------------------------------------
# 6. FULL PREPROCESSING PIPELINE
# ------------------------------------------------------------------
def preprocess_data(
    filepath: str,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "models"
):
    """
    Full preprocessing pipeline:
    1. Load raw data
    2. Clean it
    3. Encode target
    4. Split into train/test (before fitting transformers -> no data leakage)
    5. Fit preprocessor on TRAIN ONLY, transform both sets
    6. Save fitted preprocessor for use in predict.py / app.py

    Returns: X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    os.makedirs(save_dir, exist_ok=True)

    # Step 1-3
    df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)
    df = encode_target(df, target_col)

    # Step 4: split BEFORE fitting any transformer
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # important: churn datasets are usually imbalanced
    )

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[INFO] Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")

    # Step 5: build + fit preprocessor on training data only
    numerical_cols, categorical_cols = get_feature_types(
        pd.concat([X_train, y_train], axis=1), target_col
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get final feature names (after one-hot encoding)
    cat_feature_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)

    # Step 6: save preprocessor
    preprocessor_path = os.path.join(save_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[INFO] Preprocessor saved to: {preprocessor_path}")

    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor, feature_names


# ------------------------------------------------------------------
# MAIN (standalone test run)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Adjust path to wherever your raw CSV lives, e.g. data/telco_churn.csv
    DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        filepath=DATA_PATH,
        target_col="Churn",
        test_size=0.2,
        random_state=42,
        save_dir="models"
    )

    print("\n[DONE] Preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of final features: {len(feature_names)}")