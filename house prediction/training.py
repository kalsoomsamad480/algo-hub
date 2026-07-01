"""
Ames Housing - Model Training
------------------------------------------------------------
Trains a Linear Regression and a Random Forest Regressor on the
preprocessed Ames Housing data, evaluates both, and saves the BEST
model + preprocessing pipeline to disk so app.py can use them later.

Run from terminal:
    python training.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------
# STEP 1: Load data
# ----------------------------------------------------------------
DATA_PATH = "AmesHousing.csv"   # change if your file is named differently
TARGET_COL = "SalePrice"

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace(" ", "") for c in df.columns]
TARGET_COL = TARGET_COL.replace(" ", "")

id_cols = [c for c in ["Order", "PID", "Id"] if c in df.columns]
df = df.drop(columns=id_cols)
df = df.dropna(subset=[TARGET_COL])

y = df[TARGET_COL].copy()
X = df.drop(columns=[TARGET_COL])

# ----------------------------------------------------------------
# STEP 2: Missing value handling (same logic as preprocessing script)
# ----------------------------------------------------------------
none_means_missing_feature = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]
for col in none_means_missing_feature:
    if col in X.columns:
        X[col] = X[col].fillna("None")

zero_means_missing_feature = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
]
for col in zero_means_missing_feature:
    if col in X.columns:
        X[col] = X[col].fillna(0)

if "LotFrontage" in X.columns and "Neighborhood" in X.columns:
    X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )

# ----------------------------------------------------------------
# STEP 3: Feature engineering (same as preprocessing script)
# ----------------------------------------------------------------
X["TotalSF"] = (
    X.get("TotalBsmtSF", 0) + X.get("1stFlrSF", 0) + X.get("2ndFlrSF", 0)
)

if "YrSold" in X.columns and "YearBuilt" in X.columns:
    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
if "YrSold" in X.columns and "YearRemodAdd" in X.columns:
    X["YearsSinceRemodel"] = X["YrSold"] - X["YearRemodAdd"]

X["TotalBathrooms"] = (
    X.get("FullBath", 0) + 0.5 * X.get("HalfBath", 0)
    + X.get("BsmtFullBath", 0) + 0.5 * X.get("BsmtHalfBath", 0)
)

porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
porch_cols = [c for c in porch_cols if c in X.columns]
X["TotalPorchSF"] = X[porch_cols].sum(axis=1)

X["HasPool"] = (X.get("PoolArea", 0) > 0).astype(int)
X["HasGarage"] = (X.get("GarageArea", 0) > 0).astype(int)
X["HasBasement"] = (X.get("TotalBsmtSF", 0) > 0).astype(int)
X["HasFireplace"] = (X.get("Fireplaces", 0) > 0).astype(int)
X["Has2ndFloor"] = (X.get("2ndFlrSF", 0) > 0).astype(int)
X["IsRemodeled"] = (X.get("YearBuilt", 0) != X.get("YearRemodAdd", 0)).astype(int)

if "OverallQual" in X.columns and "OverallCond" in X.columns:
    X["OverallQual_x_Cond"] = X["OverallQual"] * X["OverallCond"]

# ----------------------------------------------------------------
# STEP 4: Identify column types
# ----------------------------------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# ----------------------------------------------------------------
# STEP 5: Train/test split
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Log-transform target (house prices are right-skewed)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# ----------------------------------------------------------------
# STEP 6: Preprocessing pipeline
# ----------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ----------------------------------------------------------------
# STEP 7: Define models inside full pipelines
# (preprocessing + model bundled together so app.py can call .predict()
# directly on raw input without redoing preprocessing manually)
# ----------------------------------------------------------------
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# ----------------------------------------------------------------
# STEP 8: Train both models on log-transformed target
# ----------------------------------------------------------------
print("Training Linear Regression...")
linear_model.fit(X_train, y_train_log)

print("Training Random Forest Regressor...")
rf_model.fit(X_train, y_train_log)

# ----------------------------------------------------------------
# STEP 9: Evaluate both models
# ----------------------------------------------------------------
def evaluate(model, X_test, y_test_log, y_test_actual, name):
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)  # convert back to actual dollar scale

    mae = mean_absolute_error(y_test_actual, preds)
    rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
    r2 = r2_score(y_test_actual, preds)

    print(f"\n{name} Performance (on actual SalePrice scale):")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R2   : {r2:.4f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}

results = []
results.append(evaluate(linear_model, X_test, y_test_log, y_test, "Linear Regression"))
results.append(evaluate(rf_model, X_test, y_test_log, y_test, "Random Forest"))

# ----------------------------------------------------------------
# STEP 10: Pick the best model (highest R2) and save it
# ----------------------------------------------------------------
best = max(results, key=lambda r: r["r2"])
best_model = linear_model if best["name"] == "Linear Regression" else rf_model

print(f"\nBest model: {best['name']} (R2 = {best['r2']:.4f})")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(linear_model, "linear_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")

# Save the list of columns the model expects, plus a sample row, so
# app.py knows exactly what inputs to collect from the user.
joblib.dump(list(X.columns), "model_columns.pkl")
joblib.dump(X.head(1), "sample_row.pkl")

print("\nSaved files:")
print("  best_model.pkl       -> best performing pipeline (preprocessing + model)")
print("  linear_model.pkl     -> Linear Regression pipeline")
print("  rf_model.pkl         -> Random Forest pipeline")
print("  model_columns.pkl    -> list of expected input columns")
print("  sample_row.pkl       -> one example row (used as a template in app.py)")