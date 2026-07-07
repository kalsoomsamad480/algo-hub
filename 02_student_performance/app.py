"""
app.py
------
Week 2 Project: Student Performance Prediction
Streamlit deployment app.

What this app does:
1. Loads the saved models, scaler, encoders, and selected feature list
2. Lets the user choose a model (Logistic Regression or Random Forest)
3. Shows input widgets ONLY for the features that were selected during preprocessing
4. Encodes + scales the user's input exactly like preprocessing.py did
5. Predicts Pass/Fail with a probability score
6. Shows model accuracy comparison and feature importance charts

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import pathlib

# Build the path relative to THIS script's location, not the working directory.
# This makes it work the same whether you run it locally or on Streamlit Cloud,
# where the working directory is always the repo root, not this file's folder.
BASE_DIR = pathlib.Path(__file__).parent
MODELS_DIR = str(BASE_DIR / "models")

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

# ---------------------------------------------------------
# STEP 1: Load everything saved by preprocessing.py / training.py
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    label_encoders = joblib.load(f"{MODELS_DIR}/label_encoders.pkl")
    selected_features = joblib.load(f"{MODELS_DIR}/selected_features.pkl")
    log_reg = joblib.load(f"{MODELS_DIR}/logistic_regression.pkl")
    rf_clf = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
    return scaler, label_encoders, selected_features, log_reg, rf_clf

scaler, label_encoders, selected_features, log_reg, rf_clf = load_artifacts()

# ---------------------------------------------------------
# STEP 2: Describe every possible feature (used to build the right widget)
# Only the ones present in `selected_features` will actually be shown.
# ---------------------------------------------------------
# categorical options (yes/no fields share the same options)
YES_NO = ["no", "yes"]

FEATURE_CONFIG = {
    "school":     {"type": "categorical", "options": ["GP", "MS"], "label": "School"},
    "sex":        {"type": "categorical", "options": ["F", "M"], "label": "Sex"},
    "age":        {"type": "numeric", "min": 15, "max": 22, "default": 17, "label": "Age"},
    "address":    {"type": "categorical", "options": ["U", "R"], "label": "Home Address Type (Urban/Rural)"},
    "famsize":    {"type": "categorical", "options": ["LE3", "GT3"], "label": "Family Size (<=3 or >3)"},
    "Pstatus":    {"type": "categorical", "options": ["T", "A"], "label": "Parents Cohabitation (Together/Apart)"},
    "Medu":       {"type": "numeric", "min": 0, "max": 4, "default": 2, "label": "Mother's Education (0=none - 4=higher)"},
    "Fedu":       {"type": "numeric", "min": 0, "max": 4, "default": 2, "label": "Father's Education (0=none - 4=higher)"},
    "Mjob":       {"type": "categorical", "options": ["teacher", "health", "services", "at_home", "other"], "label": "Mother's Job"},
    "Fjob":       {"type": "categorical", "options": ["teacher", "health", "services", "at_home", "other"], "label": "Father's Job"},
    "reason":     {"type": "categorical", "options": ["home", "reputation", "course", "other"], "label": "Reason for Choosing School"},
    "guardian":   {"type": "categorical", "options": ["mother", "father", "other"], "label": "Guardian"},
    "traveltime": {"type": "numeric", "min": 1, "max": 4, "default": 1, "label": "Travel Time to School (1=short - 4=long)"},
    "studytime":  {"type": "numeric", "min": 1, "max": 4, "default": 2, "label": "Weekly Study Time (1=low - 4=high)"},
    "failures":   {"type": "numeric", "min": 0, "max": 3, "default": 0, "label": "Past Class Failures"},
    "schoolsup":  {"type": "categorical", "options": YES_NO, "label": "Extra School Support"},
    "famsup":     {"type": "categorical", "options": YES_NO, "label": "Family Educational Support"},
    "paid":       {"type": "categorical", "options": YES_NO, "label": "Extra Paid Classes"},
    "activities": {"type": "categorical", "options": YES_NO, "label": "Extra-curricular Activities"},
    "nursery":    {"type": "categorical", "options": YES_NO, "label": "Attended Nursery School"},
    "higher":     {"type": "categorical", "options": YES_NO, "label": "Wants Higher Education"},
    "internet":   {"type": "categorical", "options": YES_NO, "label": "Internet Access at Home"},
    "romantic":   {"type": "categorical", "options": YES_NO, "label": "In a Romantic Relationship"},
    "famrel":     {"type": "numeric", "min": 1, "max": 5, "default": 4, "label": "Family Relationship Quality (1=bad - 5=excellent)"},
    "freetime":   {"type": "numeric", "min": 1, "max": 5, "default": 3, "label": "Free Time After School (1=low - 5=high)"},
    "goout":      {"type": "numeric", "min": 1, "max": 5, "default": 3, "label": "Going Out with Friends (1=low - 5=high)"},
    "Dalc":       {"type": "numeric", "min": 1, "max": 5, "default": 1, "label": "Workday Alcohol Consumption (1=low - 5=high)"},
    "Walc":       {"type": "numeric", "min": 1, "max": 5, "default": 1, "label": "Weekend Alcohol Consumption (1=low - 5=high)"},
    "health":     {"type": "numeric", "min": 1, "max": 5, "default": 4, "label": "Current Health Status (1=bad - 5=very good)"},
    "absences":   {"type": "numeric", "min": 0, "max": 93, "default": 4, "label": "Number of School Absences"},
    "G1":         {"type": "numeric", "min": 0, "max": 20, "default": 10, "label": "First Period Grade (G1)"},
    "G2":         {"type": "numeric", "min": 0, "max": 20, "default": 10, "label": "Second Period Grade (G2)"},
}

# ---------------------------------------------------------
# STEP 3: Sidebar - model choice
# ---------------------------------------------------------
st.title("🎓 Student Performance Prediction")
st.write(
    "Predict whether a student will **Pass** or **Fail** based on academic, "
    "social, and family background data. Built with Logistic Regression and "
    "Random Forest classifiers, trained on the UCI Student Performance dataset."
)

st.sidebar.header("Settings")
model_choice = st.sidebar.radio("Choose a model:", ["Random Forest", "Logistic Regression"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Accuracy")
if os.path.exists(f"{MODELS_DIR}/model_comparison.csv"):
    comparison_df = pd.read_csv(f"{MODELS_DIR}/model_comparison.csv")
    st.sidebar.dataframe(comparison_df, hide_index=True)

# ---------------------------------------------------------
# STEP 4: Build input form dynamically (only selected features)
# ---------------------------------------------------------
st.header("Enter Student Details")

user_input = {}
cols = st.columns(3)

for i, feature in enumerate(selected_features):
    config = FEATURE_CONFIG.get(feature)
    col = cols[i % 3]

    if config is None:
        continue  # safety fallback, shouldn't happen

    with col:
        if config["type"] == "categorical":
            choice = st.selectbox(config["label"], config["options"], key=feature)
            user_input[feature] = choice
        else:
            value = st.number_input(
                config["label"],
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                key=feature
            )
            user_input[feature] = value

# ---------------------------------------------------------
# STEP 5: Predict button
# ---------------------------------------------------------
if st.button("Predict Performance", type="primary"):

    # Build a single-row dataframe in the exact column order used during training
    input_row = {}
    for feature in selected_features:
        value = user_input[feature]
        if feature in label_encoders:
            # Encode categorical text back into the number the model expects
            encoder = label_encoders[feature]
            value = encoder.transform([value])[0]
        input_row[feature] = value

    input_df = pd.DataFrame([input_row], columns=selected_features)

    # Scale exactly like training data was scaled
    input_scaled = scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=selected_features)

    # Pick model
    model = rf_clf if model_choice == "Random Forest" else log_reg

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Predicted Result: **PASS** (Confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"❌ Predicted Result: **FAIL** (Confidence: {probability[0]*100:.1f}%)")

    st.progress(float(probability[1]))
    st.caption(f"Probability of Pass: {probability[1]*100:.1f}% | Probability of Fail: {probability[0]*100:.1f}%")
