"""
preprocessing.py
Loan Approval Prediction System - Data Preprocessing

What this does:
1. Loads the raw Kaggle "Loan Prediction Problem Dataset" (train_ csv)
2. Cleans missing values (mode for categorical, median for numeric)
3. Encodes categorical columns
4. Engineers a couple of useful features (TotalIncome, LoanAmount_log)
5. Saves the cleaned/encoded dataset + the encoders so app.py can reuse them

Run from terminal:
    python preprocessing.py

Expected input file (download from Kaggle and place here):
    data/train.csv
Kaggle dataset: altruistdelhite04/loan-prediction-problem-dataset
The Kaggle file is usually named something like "train_u6lujuX_CVtuZ9i.csv" -
just rename it to train.csv and drop it in the data/ folder.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# Paths (use pathlib so this works regardless of where you run it from)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "train.csv"
PROCESSED_PATH = DATA_DIR / "processed_data.csv"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}. Download the dataset from Kaggle "
            f"(altruistdelhite04/loan-prediction-problem-dataset), rename the "
            f"training file to train.csv, and place it in the data/ folder."
        )
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop the ID column - it carries no predictive signal
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Dependents has a "3+" category - convert to numeric-friendly "3"
    # NOTE: astype(str) turns real NaN into the literal string "nan", so we
    # convert that back to an actual NaN afterward - otherwise fillna() below
    # won't catch it and astype(int) will crash on the string "nan".
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(str).str.replace("3+", "3", regex=False)
        df["Dependents"] = df["Dependents"].replace("nan", np.nan)

    # --- Fill missing values ---
    categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    numeric_cols = ["LoanAmount", "Loan_Amount_Term"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Dependents back to int now that NaNs are gone
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(int)

    # Credit_History is really categorical (0/1) even though it's numeric
    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].astype(int)

    print("Missing values after cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else "None ✅")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Combined household income is usually more predictive than either alone
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Loan amount is right-skewed - log transform helps tree models less than
    # linear models, but it's cheap, standard practice, and keeps this reusable
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

    # Loan amount relative to income - a classic risk signal
    df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)

    return df


def encode_data(df: pd.DataFrame):
    df = df.copy()
    encoders = {}

    # Simple binary categorical columns -> LabelEncoder
    binary_cols = ["Gender", "Married", "Education", "Self_Employed"]
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Property_Area has 3 categories with no natural order -> one-hot encode
    if "Property_Area" in df.columns:
        df = pd.get_dummies(df, columns=["Property_Area"], prefix="Property_Area")
        # cast the new dummy columns to int (0/1) instead of bool
        dummy_cols = [c for c in df.columns if c.startswith("Property_Area_")]
        df[dummy_cols] = df[dummy_cols].astype(int)

    # Target column: Y/N -> 1/0
    if "Loan_Status" in df.columns:
        le_target = LabelEncoder()
        df["Loan_Status"] = le_target.fit_transform(df["Loan_Status"].astype(str))
        encoders["Loan_Status"] = le_target

    return df, encoders


def main():
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_data(df)

    # Save processed dataset
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved processed data to {PROCESSED_PATH} -> shape {df.shape}")

    # Save encoders so app.py can transform raw user input the same way
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Saved encoders to {ENCODERS_PATH}")

    # Save the final feature column order (excluding target) - critical so
    # app.py builds prediction rows with columns in the exact same order
    feature_columns = [c for c in df.columns if c != "Loan_Status"]
    with open(FEATURE_COLUMNS_PATH, "wb") as f:
        pickle.dump(feature_columns, f)
    print(f"Saved feature column order to {FEATURE_COLUMNS_PATH}")

    print("\nClass balance in Loan_Status (0=Rejected, 1=Approved):")
    print(df["Loan_Status"].value_counts(normalize=True).round(3))

    print("\nPreprocessing complete. Next: run train.py")


if __name__ == "__main__":
    main()