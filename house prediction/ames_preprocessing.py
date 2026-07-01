"""
Ames Housing Dataset - Preprocessing + Feature Engineering
------------------------------------------------------------
Run this in VS Code terminal (not the Play button), like:
    python ames_preprocessing.py

Expects a CSV file named "AmesHousing.csv" (or "train.csv" if you got it
from the Kaggle competition) in the same folder as this script.
Adjust DATA_PATH below if your file is named differently.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------
# STEP 1: Load the data
# ----------------------------------------------------------------
DATA_PATH = "AmesHousing.csv"   # change if needed, e.g. "train.csv"
TARGET_COL = "SalePrice"

df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# Ames Housing column names sometimes have spaces (e.g. "Lot Area").
# This standardizes them to no-space, consistent naming so code below
# works regardless of which version of the file you downloaded.
df.columns = [c.strip().replace(" ", "") for c in df.columns]
TARGET_COL = TARGET_COL.replace(" ", "")

# ----------------------------------------------------------------
# STEP 2: Drop irrelevant / identifier columns
# ----------------------------------------------------------------
# "Order" and "PID" are just row/parcel IDs in the Ames dataset - they
# carry no predictive information and would only confuse the model.
id_cols = [c for c in ["Order", "PID", "Id"] if c in df.columns]
df = df.drop(columns=id_cols)

# ----------------------------------------------------------------
# STEP 3: Separate target and features
# ----------------------------------------------------------------
df = df.dropna(subset=[TARGET_COL])  # rows with no target are useless
y = df[TARGET_COL].copy()
X = df.drop(columns=[TARGET_COL])

# ----------------------------------------------------------------
# STEP 4: Handle missing values intelligently
# ----------------------------------------------------------------
# In Ames Housing, many "NA" values are NOT actually missing - they mean
# "feature does not exist" (e.g. NA in PoolQC means "no pool").
# We fix this BEFORE generic imputation, because filling these with the
# column mean/mode would be wrong.

none_means_missing_feature = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]
for col in none_means_missing_feature:
    if col in X.columns:
        X[col] = X[col].fillna("None")

# Numeric columns where NA also means "0 of this feature" (e.g. no garage
# means GarageArea / GarageYrBlt / GarageCars should be 0, not averaged).
zero_means_missing_feature = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
]
for col in zero_means_missing_feature:
    if col in X.columns:
        X[col] = X[col].fillna(0)

# LotFrontage: fill with the median frontage of houses in the same
# neighborhood, since lot size is strongly tied to location.
if "LotFrontage" in X.columns and "Neighborhood" in X.columns:
    X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )

# ----------------------------------------------------------------
# STEP 5: Feature engineering
# ----------------------------------------------------------------
# These new features often boost model performance noticeably on Ames
# Housing, because raw columns don't always capture what buyers care about.

# 5a. Total square footage (basement + 1st floor + 2nd floor)
X["TotalSF"] = (
    X.get("TotalBsmtSF", 0) + X.get("1stFlrSF", 0) + X.get("2ndFlrSF", 0)
)

# 5b. House age at time of sale, and years since remodel
if "YrSold" in X.columns and "YearBuilt" in X.columns:
    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
if "YrSold" in X.columns and "YearRemodAdd" in X.columns:
    X["YearsSinceRemodel"] = X["YrSold"] - X["YearRemodAdd"]

# 5c. Total bathrooms (full baths count as 1, half baths as 0.5)
X["TotalBathrooms"] = (
    X.get("FullBath", 0) + 0.5 * X.get("HalfBath", 0)
    + X.get("BsmtFullBath", 0) + 0.5 * X.get("BsmtHalfBath", 0)
)

# 5d. Total porch square footage (combines multiple porch types into one)
porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
porch_cols = [c for c in porch_cols if c in X.columns]
X["TotalPorchSF"] = X[porch_cols].sum(axis=1)

# 5e. Simple yes/no flags - sometimes "having a feature at all" matters
# more to price than its exact size.
X["HasPool"] = (X.get("PoolArea", 0) > 0).astype(int)
X["HasGarage"] = (X.get("GarageArea", 0) > 0).astype(int)
X["HasBasement"] = (X.get("TotalBsmtSF", 0) > 0).astype(int)
X["HasFireplace"] = (X.get("Fireplaces", 0) > 0).astype(int)
X["Has2ndFloor"] = (X.get("2ndFlrSF", 0) > 0).astype(int)
X["IsRemodeled"] = (X.get("YearBuilt", 0) != X.get("YearRemodAdd", 0)).astype(int)

# 5f. Quality x Condition interaction (overall quality matters a LOT for
# price, so combining it with condition can capture nonlinear effects)
if "OverallQual" in X.columns and "OverallCond" in X.columns:
    X["OverallQual_x_Cond"] = X["OverallQual"] * X["OverallCond"]

print("Shape after feature engineering:", X.shape)

# ----------------------------------------------------------------
# STEP 6: Identify column types for preprocessing pipeline
# ----------------------------------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# ----------------------------------------------------------------
# STEP 7: Train/test split (done BEFORE scaling/encoding to avoid
# data leakage - the scaler/encoder must only "learn" from training data)
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------
# STEP 8: Build preprocessing pipeline
# ----------------------------------------------------------------
# Numeric pipeline: fill any remaining missing values with median,
# then scale so all numeric features are on a comparable range.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline: fill missing with the most frequent category,
# then one-hot encode (turns categories into 0/1 columns).
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ----------------------------------------------------------------
# STEP 9: Fit on training data, transform both train and test
# ----------------------------------------------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Final processed training shape:", X_train_processed.shape)
print("Final processed test shape:", X_test_processed.shape)

# ----------------------------------------------------------------
# STEP 10 (optional but common for SalePrice): log-transform the target
# ----------------------------------------------------------------
# House prices are right-skewed (a few very expensive houses pull the
# distribution). Many models perform better predicting log(price).
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
# Remember: to get back real dollar predictions later, use np.expm1(prediction)

print("\nPreprocessing complete. You now have:")
print("  X_train_processed, X_test_processed  -> ready for model.fit()")
print("  y_train, y_test                      -> original SalePrice")
print("  y_train_log, y_test_log               -> log-transformed target (optional)")