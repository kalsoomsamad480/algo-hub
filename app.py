"""
app.py
Loan Approval Prediction System - Streamlit App

Run from terminal:
    streamlit run app.py

Loads the trained model + encoders saved by preprocessing.py / train.py,
takes applicant details from the user, and predicts loan approval.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    missing = [p.name for p in [MODEL_PATH, ENCODERS_PATH, FEATURE_COLUMNS_PATH] if not p.exists()]
    if missing:
        st.error(
            f"Missing model file(s): {', '.join(missing)}. "
            f"Run preprocessing.py then train.py before launching the app."
        )
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    return model, encoders, feature_columns


model, encoders, feature_columns = load_artifacts()

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details below to predict whether a loan will be approved.")

st.divider()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=150, step=10)
    loan_amount_term = st.selectbox(
        "Loan Amount Term (days)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12]
    )
    credit_history = st.selectbox(
        "Credit History", ["Good (1)", "Bad (0)"]
    )

st.divider()

if st.button("Predict Loan Approval", type="primary", use_container_width=True):
    # 1. Build a raw input row matching the original dataset schema
    raw_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": int(dependents),
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": 1 if credit_history.startswith("Good") else 0,
        "Property_Area": property_area,
    }
    df = pd.DataFrame([raw_input])

    # 2. Feature engineering - must match preprocessing.py exactly
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])
    df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)

    # 3. Encode categorical columns using the SAME encoders from training
    binary_cols = ["Gender", "Married", "Education", "Self_Employed"]
    for col in binary_cols:
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))

    # 4. One-hot encode Property_Area, then align to training columns
    df = pd.get_dummies(df, columns=["Property_Area"], prefix="Property_Area")
    dummy_cols = [c for c in df.columns if c.startswith("Property_Area_")]
    if dummy_cols:
        df[dummy_cols] = df[dummy_cols].astype(int)

    # 5. Reindex to match the exact training feature order (fills any
    #    missing one-hot columns with 0)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # 6. Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Loan Approved (confidence: {probability:.1%})")
    else:
        st.error(f"❌ Loan Rejected (confidence: {1 - probability:.1%})")

    st.progress(float(probability))
    st.caption(f"Model's estimated probability of approval: {probability:.1%}")

    with st.expander("See what was sent to the model"):
        st.dataframe(df)

st.divider()
st.caption("Loan Approval Prediction System · Decision Tree / Random Forest / XGBoost · Built with Streamlit")