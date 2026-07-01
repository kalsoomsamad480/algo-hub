"""
Ames Housing - Streamlit Deployment App
------------------------------------------------------------
Loads the trained model from training.py and lets users enter house
details to get a predicted SalePrice.

Run from terminal:
    streamlit run app.py

Requires that training.py has already been run (so best_model.pkl,
model_columns.pkl, and sample_row.pkl exist in the same folder).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 Ames Housing Price Predictor")
st.write("Enter the key details of a house below to get an estimated sale price.")

# ----------------------------------------------------------------
# Load trained model + supporting files
# ----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "best_model.pkl"))
    model_columns = joblib.load(os.path.join(base_dir, "model_columns.pkl"))
    sample_row = joblib.load(os.path.join(base_dir, "sample_row.pkl"))
    return model, model_columns, sample_row

try:
    model, model_columns, sample_row = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python training.py` first "
        "to generate best_model.pkl, model_columns.pkl, and sample_row.pkl."
    )
    st.stop()

# ----------------------------------------------------------------
# Build input form
# ----------------------------------------------------------------
# We only ask the user for the features that matter most to price.
# Every other column is filled in automatically using the sample_row
# defaults saved during training, so the model still gets a full,
# correctly-shaped row to predict on.

st.subheader("House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=200, max_value=6000, value=1500)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=800)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
    year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=2000)

with col2:
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    half_bath = st.slider("Half Bathrooms", 0, 2, 0)
    bedroom_abvgr = st.slider("Bedrooms Above Ground", 0, 8, 3)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=50000, value=9000)
    neighborhood = st.selectbox(
        "Neighborhood",
        sorted(sample_row["Neighborhood"].astype(str).unique().tolist())
        if "Neighborhood" in sample_row.columns else ["NA"]
    )

year_sold = st.number_input("Year Sold", min_value=1870, max_value=2024, value=2024)

# ----------------------------------------------------------------
# Build the full input row: start from the saved sample row (so every
# column the model expects is present), then overwrite with the
# user's inputs.
# ----------------------------------------------------------------
input_row = sample_row.copy()

user_inputs = {
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "TotalBsmtSF": total_bsmt_sf,
    "GarageCars": garage_cars,
    "YearBuilt": year_built,
    "FullBath": full_bath,
    "HalfBath": half_bath,
    "BedroomAbvGr": bedroom_abvgr,
    "LotArea": lot_area,
    "Neighborhood": neighborhood,
    "YrSold": year_sold,
}

for col, value in user_inputs.items():
    if col in input_row.columns:
        input_row[col] = value

# Recompute engineered features so they stay consistent with the
# manually-entered values above (otherwise TotalSF/HouseAge etc. would
# still reflect the original sample_row, not the user's inputs).
input_row["TotalSF"] = (
    input_row.get("TotalBsmtSF", 0) + input_row.get("1stFlrSF", 0) + input_row.get("2ndFlrSF", 0)
)
if "YrSold" in input_row.columns and "YearBuilt" in input_row.columns:
    input_row["HouseAge"] = input_row["YrSold"] - input_row["YearBuilt"]
if "YrSold" in input_row.columns and "YearRemodAdd" in input_row.columns:
    input_row["YearsSinceRemodel"] = input_row["YrSold"] - input_row["YearRemodAdd"]
input_row["TotalBathrooms"] = (
    input_row.get("FullBath", 0) + 0.5 * input_row.get("HalfBath", 0)
    + input_row.get("BsmtFullBath", 0) + 0.5 * input_row.get("BsmtHalfBath", 0)
)
input_row["HasGarage"] = (input_row.get("GarageArea", 0) > 0).astype(int)
input_row["HasBasement"] = (input_row.get("TotalBsmtSF", 0) > 0).astype(int)
if "OverallQual" in input_row.columns and "OverallCond" in input_row.columns:
    input_row["OverallQual_x_Cond"] = input_row["OverallQual"] * input_row["OverallCond"]

# Make sure column order matches what the model was trained on
input_row = input_row[model_columns]

# ----------------------------------------------------------------
# Currency conversion (USD -> PKR)
# ----------------------------------------------------------------
# The Ames dataset's SalePrice is in US Dollars, so the model's raw
# output is in USD. We convert it to PKR for display using a fixed
# exchange rate. Update USD_TO_PKR_RATE below to keep it current,
# since exchange rates change over time.
USD_TO_PKR_RATE = 280  # update this to today's rate as needed

# ----------------------------------------------------------------
# Predict
# ----------------------------------------------------------------
if st.button("Predict Sale Price", type="primary"):
    pred_log = model.predict(input_row)[0]
    pred_price_usd = np.expm1(pred_log)  # undo the log1p transform from training
    pred_price_pkr = pred_price_usd * USD_TO_PKR_RATE

    st.success(f"### Estimated Sale Price: PKR {pred_price_pkr:,.0f}")
    st.caption(f"(≈ ${pred_price_usd:,.0f} USD, converted at 1 USD = {USD_TO_PKR_RATE} PKR)")
    st.caption(
        "This estimate is based on a model trained on the Ames Housing dataset "
        "(originally in USD) and should be used as a rough guide, not an official appraisal."
    )

st.divider()
st.caption("Model: trained via training.py (Linear Regression vs Random Forest, best one selected automatically).")
